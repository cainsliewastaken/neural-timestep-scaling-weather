import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# networks
from models.swin import SwinTransformer
from collections import OrderedDict


def _cfg_get(cfg, dotted_key, default=None):
    """Read ``a.b.c`` from an OmegaConf/SimpleNamespace cfg, returning ``default`` if missing."""
    node = cfg
    for key in dotted_key.split("."):
        if node is None:
            return default
        node = node.get(key, None) if isinstance(node, dict) else getattr(node, key, None)
    return default if node is None else node


class TimeStepper(nn.Module):
    """Wrapper for time stepping during training.

    step_method:
        'direct': x_{k+1} = net(x_k)  (upstream behaviour)
        'euler':  x_{k+1} = x_k + dt * (mu_y + sigma_y * net(x_k)) / sigma_x
                  where y = (x(t+dt) - x(t)) / dt is the tendency over one model step,
                  mu_y / sigma_y are its per-channel mean / std, and sigma_x is the state
                  std used to normalize the data. All state arithmetic is in normalized
                  units, so the network predicts a unit-variance tendency.

    The rollout backpropagates through every step (no detach), like Loss_Multistep in
    the 1d/2d repos; the loss is averaged over the returned steps.
    """

    def __init__(
        self,
        cfg,
        model_handle,
        step_method="direct",
        dt=1.0,
        tendency_mean=None,
        tendency_std=None,
    ):
        super(TimeStepper, self).__init__()
        self.model = model_handle
        self.num_rollout_steps = cfg.train.num_rollout_steps
        self.num_invariants = len(cfg.data.invariants) if cfg.data.invariants else 0
        self.rollout_checkpointing = bool(
            _cfg_get(cfg, "train.rollout_activation_checkpointing", False)
        )
        temporal_context_window = _cfg_get(cfg, "train.temporal_context_window", 1)
        assert self.num_rollout_steps == 1 or temporal_context_window == 1, (
            "multistep rollout feeds back a single predicted frame; "
            "requires train.temporal_context_window == 1"
        )

        self.step_method = step_method
        if step_method == "euler":
            assert tendency_mean is not None and tendency_std is not None, (
                "euler step needs tendency stats (model.tendency_stats)"
            )
            self.dt = float(dt)
            # per-channel scales in normalized-state units, shaped [1, 1, C, 1, 1]
            self.register_buffer(
                "tendency_mean", torch.as_tensor(tendency_mean, dtype=torch.float32).view(1, 1, -1, 1, 1)
            )
            self.register_buffer(
                "tendency_std", torch.as_tensor(tendency_std, dtype=torch.float32).view(1, 1, -1, 1, 1)
            )
        elif step_method != "direct":
            raise NotImplementedError(f"step_method {step_method} not implemented")

    def _net(self, inpt):
        if self.rollout_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self.model, inpt, use_reentrant=False)
        return self.model(inpt)

    def step(self, inpt):
        out = self._net(inpt)
        if self.step_method == "direct":
            return out
        # euler: state stays fp32 so small increments are not lost under AMP
        c_out = out.shape[2]
        state = inpt[:, -1:, :c_out].float()
        return state + self.dt * (self.tendency_mean + self.tendency_std * out.float())

    def forward(self, inp):
        result = []
        inpt = inp
        invars = inp[:, :, inp.shape[2] - self.num_invariants :] if self.num_invariants > 0 else None

        # stepper
        for step in range(self.num_rollout_steps):
            pred = self.step(inpt)
            c_out = pred.shape[2]  # can be different if there are invariants
            assert inp.shape[2] - c_out == self.num_invariants, "number of invariants does not match"
            result.append(pred)
            if step == self.num_rollout_steps - 1:
                break
            # add back invariants at every step
            inpt = torch.cat([pred, invars], dim=2) if invars is not None else pred

        result = torch.cat(result, dim=1)
        return result


def load_tendency_scales(cfg, domain_metadata):
    """Per-channel Euler scales (mu_y / sigma_x, sigma_y / sigma_x) and dt for this dt_scale.

    ``model.tendency_stats`` is an h5 file laid out like upstream's stats_<tag>.h5
    (channel-wise, ``temp_diff_*`` naming) with datasets
        channel         [C]          channel names (must match the data)
        dt_scales       [n_dt]       increments in native data steps
        temp_diff_mean  [n_dt, C]    mean of x(t + dt_scale * dt_hours) - x(t), physical units
        temp_diff_std   [n_dt, C]    std of the same increment, physical units
    ``model.time_unit_hours`` sets the time unit: dt = dt_scale * dt_hours / time_unit_hours.
    """
    stats_path = _cfg_get(cfg, "model.tendency_stats")
    assert stats_path is not None, "model.step_method=euler requires model.tendency_stats"
    dt_scale = int(cfg.data.dt_scale)
    dt_hours = domain_metadata["dt_hours"] / np.timedelta64(1, "h")
    dt = dt_scale * dt_hours / float(_cfg_get(cfg, "model.time_unit_hours", 1.0))

    state_stats_path = os.path.join("/data", cfg.data.name, "stats", f"stats_{cfg.data.tag}.h5")
    with h5py.File(state_stats_path, "r") as f:
        state_std = f["global_std"][:].astype(np.float64)
    with h5py.File(stats_path, "r") as f:
        channels = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in f["channel"][:]]
        dt_scales = [int(d) for d in f["dt_scales"][:]]
        assert dt_scale in dt_scales, f"dt_scale={dt_scale} not in {stats_path} (has {dt_scales})"
        idx = dt_scales.index(dt_scale)
        diff_mean = f["temp_diff_mean"][idx].astype(np.float64)
        diff_std = f["temp_diff_std"][idx].astype(np.float64)
    assert channels == list(domain_metadata["channels"]), "tendency stats channel order does not match data"
    state_std = state_std.reshape(-1)[: len(channels)]

    # y = dx / dt  ->  mu_y = diff_mean / dt, sigma_y = diff_std / dt; divide by sigma_x for normalized units
    tendency_mean = diff_mean / dt / state_std
    tendency_std = diff_std / dt / state_std
    return dt, tendency_mean, tendency_std


def get_model(cfg, domain_metadata=None):
    name = cfg.model.arch
    if name == "swin":
        model = SwinTransformer.instantiate_from_cfg(cfg, domain_metadata=domain_metadata)
    else:
        raise NotImplementedError(f"model type {name} not implemented")

    # wrap the model into a time stepper to deal with
    # multistep finetuning etc
    step_method = _cfg_get(cfg, "model.step_method", "direct")
    if step_method == "euler":
        dt, tendency_mean, tendency_std = load_tendency_scales(cfg, domain_metadata)
        model = TimeStepper(
            cfg, model, step_method="euler", dt=dt,
            tendency_mean=tendency_mean, tendency_std=tendency_std,
        )
    else:
        model = TimeStepper(cfg, model, step_method=step_method)
    return model


def load_model(model, checkpoint_file, local_rank):
    map_location = "cuda:{}".format(local_rank) if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        checkpoint_file,
        map_location=map_location,
        weights_only=False,
    )
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError:
        if not all(key.startswith("module.") for key in checkpoint["model_state"]):
            raise
        new_state_dict = OrderedDict()
        for key, val in checkpoint["model_state"].items():
            name = key.removeprefix("module.")
            new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    return model
