"""IsoFLOP rollout evaluation for one ERA5 run (the ERA5 analogue of 2d_turbulence/evaluation/eval_network.py).

For each test initial condition (IC) the truth frames are read once and reused for every checkpoint:

  final checkpoint (end of cooldown), out to --horizon-hours:
      e_k   = x_k - x*_k                        autoregressive error
      eps_k = step(x*_{k-1}) - x*_k             one-step (teacher-forced) error, eps_0 = 0
      I_k   = sum_{j<=k} eps_j                  accumulated one-step error ("injection")
      R_k   = e_k - I_k                         error not explained by the injected one-step errors
    per IC, step and channel we store the mean square of e_k, I_k, eps_k and R_k (normalized units),
    both latitude-weighted and unweighted. Norms are taken in the plotting script.
  intermediate checkpoints (every --traj-every iterations, plus the end of the main phase), out to
  --traj-horizon-hours on a subset of the ICs: mean square of e_k only (for error vs training FLOPs).
  Hutchinson estimate of Tr(dH/du)/N at the truth IC states, where H = (x_{k+1} - x_k) / dt is the
  Euler tendency in normalized units per hour (the 2D code uses the tendency network, not the step map).

The model runs with a latitude split of --sp1 GPUs (any split works: the positional embedding is
coordinate based, so weights do not depend on the layout); ICs are split over the data-parallel groups. Results go to one npz per run (rank 0).

Launch (inside the training Shifter image, mounts /data /expts /registry, one task per GPU):
    python eval_isoflop.py --run-tag 6T_dt4p0_e320_d5 --sweep-yaml scripts_6T_.../swin_dt_scale_4p0_main.yaml
"""

import argparse
import glob
import logging
import os
import re
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml
from omegaconf import OmegaConf

from distributed.mappings import allreduce_forward_identity_backward
from models.helpers import get_model, load_model
from utils import comm
from utils.data_utils import get_dataset, get_preprocessor

_ITER_RE = re.compile(r"ckpt_iter(\d+)\.tar$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-tag", required=True, help="main-phase run tag, e.g. 6T_dt4p0_e320_d5")
    p.add_argument("--run-name", default="era5_isoflop_v2")
    p.add_argument("--expts-root", default="/expts")
    p.add_argument("--sweep-yaml", required=True, help="the run's _main.yaml (FLOP and parameter metadata)")
    p.add_argument("--out", default=None, help="output npz (default <run dir>/eval/isoflop_eval.npz)")
    p.add_argument("--year", type=int, default=2020, help="test year for the ICs")
    p.add_argument("--n-ics", type=int, default=32)
    p.add_argument("--ic-spacing-hours", type=int, default=264, help="11 days; odd ICs are shifted by 12 h")
    p.add_argument("--horizon-hours", type=int, default=72)
    p.add_argument("--traj-every", type=int, default=2000, help="intermediate checkpoint spacing (iterations)")
    p.add_argument("--traj-n-ics", type=int, default=16, help="ICs used for intermediate checkpoints")
    p.add_argument("--traj-horizon-hours", type=int, default=24)
    p.add_argument("--trace-n-ics", type=int, default=8)
    p.add_argument("--trace-k", type=int, default=4, help="Rademacher probes per trace IC")
    p.add_argument("--no-traj", action="store_true")
    p.add_argument("--no-trace", action="store_true")
    p.add_argument("--max-ics", type=int, default=None, help="debug: only evaluate the first N ICs")
    p.add_argument("--sp1", type=int, default=2,
                   help="latitude split for eval (weights do not depend on it: coordinate positional embeddings)")
    return p.parse_args()


def log(msg):
    if comm.get_world_rank() == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ckpt_iters(ckpt_dir):
    out = {}
    for path in glob.glob(os.path.join(ckpt_dir, "ckpt_iter*.tar")):
        m = _ITER_RE.search(path)
        if m:
            out[int(m.group(1))] = path
    return out


def ic_times(year, n_ics, spacing_hours):
    t0 = np.datetime64(f"{year}-01-01T00", "h")
    return np.array([t0 + np.timedelta64(k * spacing_hours + 12 * (k % 2), "h") for k in range(n_ics)])


def load_state(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    if all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state, int(ckpt.get("iters", -1))


class Evaluator:
    def __init__(self, args):
        self.args = args
        main_dir = os.path.join(args.expts_root, args.run_name, args.run_tag)
        cool_dir = main_dir + "-cool"
        self.main_dir, self.cool_dir = main_dir, cool_dir

        cfg = OmegaConf.load(os.path.join(main_dir, "hyperparameters.yaml"))
        # inference only: one sample per forward, no multistep rollout in the dataset windows
        cfg.parallelism.micro_batch_size = 1
        cfg.train.num_rollout_steps = 1
        assert cfg.model.get("coord_pos_embed", True), "learned pos_embed is sharded by sp1; eval needs the training layout"
        cfg.parallelism.sp1, cfg.parallelism.sp2 = args.sp1, 1
        self.cfg = cfg

        comm.init(cfg)
        self.world_rank = comm.get_world_rank()
        self.local_rank = comm.get_local_rank()
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")

        with open(args.sweep_yaml) as f:
            self.sweep_meta = yaml.safe_load(f)["metadata"]

        self.dataset = get_dataset(cfg, mode="test")
        self.metadata = self.dataset.metadata
        self.channels = list(self.metadata["channels"])
        self.n_ch = len(self.channels)
        self.dt_hours = int(cfg.data.dt_scale) * int(self.metadata["dt_hours"] / np.timedelta64(1, "h"))
        self.time_unit_hours = float(cfg.model.get("time_unit_hours", 1.0))

        self.model = get_model(cfg, domain_metadata=self.metadata).to(self.device)
        self.model = comm.all_model_groups_exist(self.model)
        self.model.eval()
        self.preprocessor = get_preprocessor(cfg, self.dataset).to(self.device)
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[cfg.train.amp_dtype]
        self.enable_amp = bool(cfg.train.enable_amp)

        # latitude weights (mean 1 over the global grid) and global point counts
        self.w = torch.as_tensor(self.metadata["area_weights"]).to(self.device, torch.float32)  # [nlat_local, 1]

    # ----- reductions over the spatial-parallel group -----
    def _sp_sum(self, t):
        return allreduce_forward_identity_backward(t, comm_name="sp1-sp2")

    def channel_ms(self, x):
        """x: [B, C, h, w] local shard -> (weighted MS [B, C], unweighted MS [B, C]) over the global grid."""
        x = x.float()
        sq = x * x
        sums = torch.stack([(sq * self.w).sum(dim=(-1, -2)), sq.sum(dim=(-1, -2))], dim=0)  # [2, B, C]
        count = torch.tensor(float(x.shape[-1] * x.shape[-2]), device=x.device)
        sums = self._sp_sum(sums)
        count = self._sp_sum(count)
        return sums[0] / count, sums[1] / count

    # ----- model calls -----
    def step(self, x):
        """x: [B, C, h, w] normalized state -> next state [B, C, h, w] (fp32)."""
        inp = self.preprocessor(x[:, None], device=self.device, mode="inference")
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
            out = self.model.step(inp)
        return out[:, 0].float()

    def tendency_per_hour(self, u):
        """Euler tendency H(u) = (step(u) - u) / dt in normalized units per hour."""
        inp = self.preprocessor(u[:, None], device=self.device, mode="inference")
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
            out = self.model.model(inp)
        if self.model.step_method == "euler":
            h = self.model.tendency_mean + self.model.tendency_std * out.float()
            return h[:, 0] / self.time_unit_hours
        # direct stepper: finite-difference tendency of the step map
        return (out[:, 0].float() - u.float()) / float(self.dt_hours)

    def trace_estimate(self, u0, k_probes):
        """Hutchinson Tr(dH/du)/N at u0 [1, C, h, w]; returns k_probes estimates."""
        n_global = self._sp_sum(torch.tensor(float(u0.numel()), device=self.device))
        ests = []
        # Transformer Engine layers free their saved tensors after one backward (no retain_graph),
        # so each probe redoes the forward pass.
        for _ in range(k_probes):
            u = u0.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                h = self.tendency_per_hour(u)
                v = torch.randint(0, 2, h.shape, device=h.device, dtype=torch.float32) * 2 - 1
                (g,) = torch.autograd.grad(h, u, grad_outputs=v)
            local = (g.float() * v).sum()
            ests.append((self._sp_sum(local.detach()) / n_global).item())
        return np.array(ests, dtype=np.float64)

    def rollout(self, frames, n_steps, teacher_forced):
        """frames: [S+1, C, h, w] normalized truth on CPU. Returns dict of [n_steps+1, C] numpy arrays."""
        keys = ["e_w", "e_u"] + (["I_w", "I_u", "eps_w", "eps_u", "R_w", "R_u"] if teacher_forced else [])
        res = {k: np.zeros((n_steps + 1, self.n_ch), dtype=np.float64) for k in keys}
        with torch.inference_mode():
            x = frames[0:1].to(self.device, non_blocking=True).float()
            eps_sum = torch.zeros_like(x) if teacher_forced else None
            for k in range(1, n_steps + 1):
                truth_prev = frames[k - 1 : k].to(self.device, non_blocking=True).float()
                truth_k = frames[k : k + 1].to(self.device, non_blocking=True).float()
                if teacher_forced:
                    nxt = self.step(torch.cat([x, truth_prev], dim=0))  # one batched forward
                    x, tf = nxt[0:1], nxt[1:2]
                    eps = tf - truth_k
                    eps_sum += eps
                    err = x - truth_k
                    ms_w, ms_u = self.channel_ms(torch.cat([err, eps_sum, eps, err - eps_sum], dim=0))
                    res["e_w"][k], res["I_w"][k], res["eps_w"][k], res["R_w"][k] = ms_w.cpu().numpy()
                    res["e_u"][k], res["I_u"][k], res["eps_u"][k], res["R_u"][k] = ms_u.cpu().numpy()
                else:
                    x = self.step(x)
                    ms_w, ms_u = self.channel_ms(x - truth_k)
                    res["e_w"][k] = ms_w[0].cpu().numpy()
                    res["e_u"][k] = ms_u[0].cpu().numpy()
                if not torch.isfinite(x).all():
                    logging.warning(f"non-finite state at step {k}")
        return res

    def read_frames(self, t0, n_steps):
        ts = [t0 + np.timedelta64(k * self.dt_hours, "h") for k in range(n_steps + 1)]
        frames = self.dataset.get_samples(ts)  # [S+1, C, h, w] normalized
        # drop the page cache for what we just read (cgroup direct reclaim otherwise stalls reads)
        for handle in getattr(self.dataset, "ds_era5", []):
            if handle is not None:
                try:
                    os.posix_fadvise(handle.id.get_vfd_handle(), 0, 0, os.POSIX_FADV_DONTNEED)
                except Exception:
                    pass
        return frames.float().pin_memory()

    def run(self):
        a = self.args
        # ----- checkpoints -----
        main_ck = ckpt_iters(os.path.join(self.main_dir, "checkpoints"))
        cool_ck = ckpt_iters(os.path.join(self.cool_dir, "checkpoints"))
        total_iters = int(self.sweep_meta["total_iterations"])
        main_end = int(self.sweep_meta["cooldown_from_iter"])
        assert total_iters in cool_ck, f"final checkpoint ckpt_iter{total_iters}.tar missing in {self.cool_dir}"
        final_path = cool_ck[total_iters]
        traj = []  # (iter, path)
        if not a.no_traj:
            targets = list(range(a.traj_every, main_end, a.traj_every)) + [main_end]
            for t in targets:
                avail = [i for i in main_ck if i <= t]
                if not avail:
                    continue
                it = max(avail)
                if t - it > a.traj_every // 2:
                    log(f"no checkpoint near iter {t} (closest below: {it}); skipped")
                    continue
                if it not in [x[0] for x in traj]:
                    traj.append((it, main_ck[it]))
        log(f"final: {final_path}; trajectory iters: {[t[0] for t in traj]}")

        final_state, final_iter = load_state(final_path, self.device)
        assert final_iter in (-1, total_iters), f"{final_path} holds iters={final_iter}"
        traj_states = [(it, load_state(p, self.device)[0]) for it, p in traj]

        # ----- ICs -----
        ics = ic_times(a.year, a.n_ics, a.ic_spacing_hours)
        if a.max_ics:
            ics = ics[: a.max_ics]
        n_ics = len(ics)
        traj_ics = set(range(0, n_ics, max(1, n_ics // max(1, a.traj_n_ics)))[: a.traj_n_ics])
        trace_ics = set(range(0, n_ics, max(1, n_ics // max(1, a.trace_n_ics)))[: a.trace_n_ics])
        dp_rank, dp_size = comm.get_rank("dp"), comm.get_size("dp")
        my_ics = list(range(dp_rank, n_ics, dp_size))

        assert a.horizon_hours % self.dt_hours == 0 and a.traj_horizon_hours % self.dt_hours == 0
        S = a.horizon_hours // self.dt_hours
        S_traj = a.traj_horizon_hours // self.dt_hours
        n_tr = len(traj_states)

        final = {k: np.zeros((n_ics, S + 1, self.n_ch)) for k in ["e_w", "e_u", "I_w", "I_u", "eps_w", "eps_u", "R_w", "R_u"]}
        traj_e = {k: np.zeros((n_tr, n_ics, S_traj + 1, self.n_ch)) for k in ["e_w", "e_u"]}
        trace = np.zeros((n_ics, a.trace_k))
        done = np.zeros(n_ics)

        log(f"{a.run_tag}: dt={self.dt_hours}h, {n_ics} ICs from {ics[0]}, dp={dp_size}, "
            f"S={S}, traj ckpts={n_tr} x {len(traj_ics)} ICs, trace {len(trace_ics)} ICs x {a.trace_k}")
        t_start = time.time()
        for j, i in enumerate(my_ics):
            t_ic = time.time()
            frames = self.read_frames(ics[i], S)
            t_read = time.time() - t_ic

            self.model.load_state_dict(final_state)
            res = self.rollout(frames, S, teacher_forced=True)
            for k in final:
                final[k][i] = res[k]
            if i in trace_ics and not a.no_trace:
                trace[i] = self.trace_estimate(frames[0:1].to(self.device).float(), a.trace_k)
            if i in traj_ics:
                for c, (it, state) in enumerate(traj_states):
                    self.model.load_state_dict(state)
                    res = self.rollout(frames[: S_traj + 1], S_traj, teacher_forced=False)
                    traj_e["e_w"][c, i] = res["e_w"]
                    traj_e["e_u"][c, i] = res["e_u"]
            done[i] = 1
            log(f"IC {j + 1}/{len(my_ics)} ({ics[i]}): read {t_read:.0f}s, total {time.time() - t_ic:.0f}s, "
                f"e(24h)={np.sqrt(final['e_w'][i, min(S, 24 // self.dt_hours)].mean()):.4f}, "
                f"elapsed {(time.time() - t_start) / 60:.1f} min")

        # ----- gather: only sp-rank-0 of each dp group contributes, then sum over the world -----
        contrib = 1.0 if comm.get_rank("sp1-sp2") == 0 else 0.0

        def reduce_np(arr):
            t = torch.as_tensor(arr * contrib, dtype=torch.float64, device=self.device)
            dist.all_reduce(t)
            return t.cpu().numpy()

        final = {k: reduce_np(v) for k, v in final.items()}
        traj_e = {k: reduce_np(v) for k, v in traj_e.items()}
        trace = reduce_np(trace)
        done = reduce_np(done)

        if self.world_rank == 0:
            assert np.all(done == 1), f"ICs not evaluated: {np.where(done != 1)[0]}"
            out = a.out or os.path.join(self.main_dir, "eval", "isoflop_eval.npz")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fpi = float(self.sweep_meta["train_flops_per_iteration"])
            traj_iters = np.array([it for it, _ in traj_states] + [total_iters], dtype=np.int64)
            np.savez(
                out,
                run_tag=a.run_tag,
                budget_label=str(self.sweep_meta["budget_label"]),
                dt_hours=float(self.dt_hours),
                lead_hours=np.arange(S + 1, dtype=np.float64) * self.dt_hours,
                traj_lead_hours=np.arange(S_traj + 1, dtype=np.float64) * self.dt_hours,
                channels=np.array(self.channels),
                stds=np.asarray(self.dataset.stds, dtype=np.float64).reshape(-1)[: self.n_ch],
                ic_times=ics.astype("datetime64[h]").astype(str),
                traj_ic_index=np.array(sorted(traj_ics), dtype=np.int64),
                trace_ic_index=np.array(sorted(trace_ics), dtype=np.int64),
                final_iter=total_iters,
                final_checkpoint=final_path,
                **{f"final_{k}": v for k, v in final.items()},
                # trajectory: intermediate checkpoints, then the final checkpoint (from the final rollout)
                traj_iters=traj_iters,
                traj_train_flops=traj_iters.astype(np.float64) * fpi,
                traj_e_w=np.concatenate(
                    [traj_e["e_w"], final["e_w"][None, :, : S_traj + 1]], axis=0)[:, sorted(traj_ics)],
                traj_e_u=np.concatenate(
                    [traj_e["e_u"], final["e_u"][None, :, : S_traj + 1]], axis=0)[:, sorted(traj_ics)],
                trace=trace[sorted(trace_ics)],
                train_flops_per_iteration=fpi,
                total_train_flops=float(self.sweep_meta["total_train_flops"]),
                num_parameters=int(self.sweep_meta["num_parameters"]),
                embed_dim=int(self.sweep_meta["embed_dim"]),
                depth=int(self.sweep_meta["depth"]),
                time_unit_hours=self.time_unit_hours,
            )
            log(f"wrote {out} ({(time.time() - t_start) / 60:.1f} min)")
        dist.barrier()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    Evaluator(args).run()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
