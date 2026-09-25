"""Build Swin + TimeStepper and count forward FLOPs for isoflop sweeps."""

from __future__ import annotations

import gc
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from distributed.helpers import compute_split_shapes_for_patching
from models.helpers import TimeStepper
from models.swin import SwinTransformer
from utils.flops_utils import FlopsCalculator

# ERA5 0.25 deg grid and channel counts from config/data/latlon_025deg.yaml
DEFAULT_H = 720
DEFAULT_W = 1440
# Reduced grid for PyTorch flop counting (1/16 area); scaled back to full grid in the generator.
DEFAULT_FLOP_COUNT_H = 180
DEFAULT_FLOP_COUNT_W = 360
DEFAULT_N_CHANNELS = 71
DEFAULT_INVARIANTS = ("lsm", "orog", "coslat")


def dict_to_cfg(d: dict[str, Any]) -> SimpleNamespace:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_cfg(v) for k, v in d.items()})
    return d


def get_local_slice(sizes: list[int], rank: int) -> slice:
    start = sum(sizes[:rank])
    end = start + sizes[rank]
    return slice(start, end)


def get_spatial_coords(locations: tuple[np.ndarray, np.ndarray]) -> torch.Tensor:
    lat = torch.tensor(locations[0], dtype=torch.float32)
    lon = torch.tensor(locations[1], dtype=torch.float32)
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    lat_rad, lon_rad = torch.deg2rad(lat), torch.deg2rad(lon)
    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    return torch.stack((x, y, z), dim=-1)


def get_window_size(
    patch_size: int,
    sp1: int = 1,
    sp2: int = 1,
    img_size: tuple[int, int] = (DEFAULT_H, DEFAULT_W),
    base_patch_size: int = 4,
    base_window_size: tuple[int, int] = (9, 18),
) -> tuple[int, int]:
    """Match batchsub.py window sizing for a given patch size."""
    h, w = img_size
    base_wh, _ = base_window_size
    scaled_wh = max(1, math.ceil(base_wh * base_patch_size / patch_size))
    h_grid, w_grid = h // patch_size, w // patch_size
    local_h_grid = (h // sp1) // patch_size
    local_w_grid = (w // sp2) // patch_size
    possible_wh = [height for height in range(1, h_grid + 1) if h_grid % height == 0]
    valid_wh = [
        height
        for height in possible_wh
        if (2 * height) <= w_grid
        and w_grid % (2 * height) == 0
        and local_h_grid % height == 0
        and local_w_grid % (2 * height) == 0
    ]
    if not valid_wh:
        raise ValueError(
            "No valid Swin window size for "
            f"img_size={img_size}, patch_size={patch_size}, sp1={sp1}, sp2={sp2}. "
            "Try a grid whose height and width remain divisible by patch_size and "
            "by sp1/sp2 after patching (e.g. 360x720 or 720x1440 for patch_size=4, sp2=2)."
        )
    wh = max([height for height in valid_wh if height <= scaled_wh], default=valid_wh[0])
    ww = wh * 2
    assert (h // patch_size) % wh == 0
    assert (w // patch_size) % ww == 0
    assert ((h // sp1) // patch_size) % wh == 0
    assert ((w // sp2) // patch_size) % ww == 0
    return wh, ww


def make_domain_metadata(
    h: int = DEFAULT_H,
    w: int = DEFAULT_W,
    sp1: int = 1,
    sp2: int = 1,
    patch_size: int = 4,
    n_channels: int = DEFAULT_N_CHANNELS,
) -> dict[str, Any]:
    sp1_shapes = compute_split_shapes_for_patching(h, sp1, patch_size)
    sp2_shapes = compute_split_shapes_for_patching(w, sp2, patch_size)
    lat_slice = get_local_slice(sp1_shapes, 0)
    lon_slice = get_local_slice(sp2_shapes, 0)
    lats = np.linspace(90.0, -90.0, h)
    lons = np.linspace(0.0, 360.0, w, endpoint=False)
    lat_local = lats[lat_slice]
    lon_local = lons[lon_slice]
    coords = get_spatial_coords((lat_local, lon_local))
    return {
        "sp_shapes": [sp1_shapes, sp2_shapes],
        "sp_slices": [lat_slice, lon_slice],
        "spatial_dims": [len(lat_local), len(lon_local)],
        "channels": [f"ch{i}" for i in range(n_channels)],
        "n_channels": n_channels,
        "coords": coords,
        "global_latitudes": lats,
        "global_longitudes": lons,
    }


def build_training_stack(
    *,
    embed_dim: int,
    depth: int,
    num_heads: int,
    patch_size: int,
    window_size: tuple[int, int],
    temporal_context_window: int = 1,
    num_rollout_steps: int = 1,
    invariants: tuple[str, ...] = DEFAULT_INVARIANTS,
    use_transformer_engine: bool = False,
    h: int = DEFAULT_H,
    w: int = DEFAULT_W,
    device: torch.device | str = "cpu",
) -> TimeStepper:
    cfg = dict_to_cfg(
        {
            "model": {
                "arch": "swin",
                "embed_dim": embed_dim,
                "patch_size": patch_size,
                "depth": depth,
                "window_size": list(window_size),
                "num_heads": num_heads,
                "dropout": 0.0,
                "mlp_ratio": 4,
                "coord_pos_embed": True,
            },
            "train": {
                "temporal_context_window": temporal_context_window,
                "num_rollout_steps": num_rollout_steps,
            },
            "data": {"invariants": list(invariants)},
            "parallelism": {"use_transformer_engine": use_transformer_engine},
        }
    )
    metadata = make_domain_metadata(h=h, w=w, patch_size=patch_size)
    model = SwinTransformer.instantiate_from_cfg(cfg, domain_metadata=metadata)
    stepper = TimeStepper(cfg, model)
    return stepper.to(device).eval()


def make_sample(
    metadata: dict[str, Any],
    *,
    batch_size: int = 1,
    temporal_context_window: int = 1,
    n_invariants: int = len(DEFAULT_INVARIANTS),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    h, w = metadata["spatial_dims"]
    c_in = metadata["n_channels"] + n_invariants
    return torch.randn(
        batch_size,
        temporal_context_window,
        c_in,
        h,
        w,
        dtype=torch.float32,
        device=device,
    )


def resolve_device(device: str, *, for_flop_count: bool = True) -> str:
    """Normalize device strings; flop counting defaults to CPU to avoid GPU OOM."""
    normalized = device.strip().lower()
    if normalized in {"gpu", "cuda"}:
        if for_flop_count:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    return device


def free_memory(device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_flops_cfg(
    *,
    embed_dim: int,
    depth: int,
    num_heads: int,
    patch_size: int,
    window_size: tuple[int, int],
    temporal_context_window: int = 1,
) -> SimpleNamespace:
    return dict_to_cfg(
        {
            "model": {
                "embed_dim": embed_dim,
                "depth": depth,
                "num_heads": num_heads,
                "patch_size": patch_size,
                "window_size": list(window_size),
            },
            "train": {"temporal_context_window": temporal_context_window},
            "parallelism": {"micro_batch_size": 1},
        }
    )


def count_forward_flops_analytical(
    *,
    embed_dim: int,
    depth: int,
    num_heads: int,
    patch_size: int,
    window_size: tuple[int, int],
    h: int = DEFAULT_H,
    w: int = DEFAULT_W,
    n_channels: int = DEFAULT_N_CHANNELS,
    n_invariants: int = len(DEFAULT_INVARIANTS),
    temporal_context_window: int = 1,
) -> float:
    """Full-model forward FLOPs using the same formulas as ``FlopsCalculator``.

    ``FlopsCalculator.flops()`` returns TFLOPs; convert back to raw FLOPs so the
    isoflop search matches ``trainer.py`` (which uses this calculator, not
    ``FlopCounterMode``). PyTorch/fvcore miss ``scaled_dot_product_attention``.
    """
    c_in = n_channels + n_invariants
    cfg = make_flops_cfg(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        window_size=window_size,
        temporal_context_window=temporal_context_window,
    )
    calculator = FlopsCalculator(cfg, h=h, w=w, c_in=c_in, c_out=n_channels)
    return float(calculator.flops() * 1e12)


def missing_sdpa_flops(
    *,
    embed_dim: int,
    depth: int,
    num_heads: int,
    patch_size: int,
    window_size: tuple[int, int],
    h: int,
    w: int,
    temporal_context_window: int = 1,
) -> float:
    """QK^T + softmax + AV FLOPs omitted by PyTorch/fvcore flop counters.

    ``FlopCounterMode`` and fvcore count QKV/proj matmuls but not
    ``F.scaled_dot_product_attention``.
    """
    wh, ww = window_size
    h_patch = h // patch_size
    w_patch = w // patch_size
    num_windows = (h_patch // wh) * (w_patch // ww)
    seq = temporal_context_window * wh * ww
    head_dim = embed_dim // num_heads
    batch_windows = num_windows  # micro-batch 1
    logits = 2 * batch_windows * num_heads * seq * head_dim * seq
    softmax = 5 * batch_windows * num_heads * seq * seq
    attend = 2 * batch_windows * num_heads * seq * seq * head_dim
    return float((logits + softmax + attend) * depth)


def count_parameters_analytical(
    *,
    embed_dim: int,
    depth: int,
    num_heads: int,
    patch_size: int,
    window_size: tuple[int, int],
    h: int = DEFAULT_H,
    w: int = DEFAULT_W,
    n_channels: int = DEFAULT_N_CHANNELS,
    n_invariants: int = len(DEFAULT_INVARIANTS),
) -> int:
    c_in = n_channels + n_invariants
    cfg = make_flops_cfg(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        window_size=window_size,
    )
    calculator = FlopsCalculator(cfg, h=h, w=w, c_in=c_in, c_out=n_channels)
    return int(calculator.param_count() * 1e6)


def count_forward_flops(
    model: torch.nn.Module,
    sample: torch.Tensor,
    *,
    backend: str = "pytorch",
    verbose: bool = False,
) -> float:
    """Count one forward pass through the training stack."""
    model.eval()
    with torch.no_grad():
        if backend == "pytorch":
            from torch.utils.flop_counter import FlopCounterMode

            with FlopCounterMode(display=verbose) as fcm:
                model(sample)
            return float(fcm.get_total_flops())
        if backend == "fvcore":
            from fvcore.nn import FlopCountAnalysis

            fca = FlopCountAnalysis(model, sample)
            if not verbose:
                fca.unsupported_ops_warnings(False)
                fca.uncalled_modules_warnings(False)
            return float(fca.total())
        raise ValueError(
            f"Unknown flop backend {backend!r}; use 'analytical', 'pytorch', or 'fvcore'."
        )


def measure_forward_flops_pytorch(
    *,
    embed_dim: int,
    depth: int,
    num_heads: int,
    patch_size: int,
    window_size: tuple[int, int],
    h: int,
    w: int,
    device: str = "cpu",
    backend: str = "analytical",
    verbose: bool = False,
) -> float:
    """Forward FLOPs for isoflop sizing.

    Default ``analytical`` uses ``FlopsCalculator`` (same as training). The
    PyTorch/fvcore backends build the model and add the SDPA terms those
    counters skip.
    """
    if backend == "analytical":
        flops = count_forward_flops_analytical(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            window_size=window_size,
            h=h,
            w=w,
        )
        if verbose:
            print(f"  [analytical] forward_flops={flops:.6e}", flush=True)
        return flops

    flop_device = resolve_device(device, for_flop_count=True)
    model = build_training_stack(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        window_size=window_size,
        h=h,
        w=w,
        device=flop_device,
        use_transformer_engine=False,
    )
    metadata = make_domain_metadata(h=h, w=w, patch_size=patch_size)
    sample = make_sample(metadata, device=flop_device)
    try:
        counted = count_forward_flops(model, sample, backend=backend, verbose=verbose)
        counted += missing_sdpa_flops(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            window_size=window_size,
            h=h,
            w=w,
        )
        return counted
    finally:
        del model, sample, metadata
        free_memory(flop_device)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def valid_embed_dim(embed_dim: int, head_dim: int) -> bool:
    return embed_dim >= head_dim and embed_dim % head_dim == 0
