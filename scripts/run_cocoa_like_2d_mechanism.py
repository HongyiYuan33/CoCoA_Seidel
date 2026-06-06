"""Run a CoCoA-like object/loss mechanism with the 2-D Seidel ring forward.

This script keeps the Neural_RIng_AO forward model intact:

* 2-D coordinate input/output.
* Seidel ring PSF synthesis and ring convolution.

The object-side mechanism is changed to mirror the CoCoA demo as closely as a
2-D target permits:

* positive MLP output via Softplus, capped by max_val during rendering;
* 5x blurred-measurement pretraining with global SSIM loss;
* reciprocal standard-deviation contrast prior (RSD) during joint training;
* no 2-D TV by default.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_ring_cocoa import (  # noqa: E402
    NeuralObject2D,
    load_baboon_gt,
    synthesize_measurement,
)
from hybrid_ring_cocoa.metrics import compute_nrmse, msssim, ssim_window  # noqa: E402
from hybrid_ring_cocoa.object.coords import make_coord_grid_2d  # noqa: E402
from hybrid_ring_cocoa.object.encoding import radial_fourier_encoding  # noqa: E402
from hybrid_ring_cocoa.optics.ring_forward import (  # noqa: E402
    blur_ring_trace_trainable,
    blur_ring_trainable,
    blur_ring_with_psfs,
)
from hybrid_ring_cocoa.optics.seidel_psf import (  # noqa: E402
    build_sys_params,
    compress_trace_seidel,
    get_trainable_ring_psfs,
    get_trainable_trace_ring_psfs,
)
from hybrid_ring_cocoa.training.data import synthesize_trace_measurement  # noqa: E402
from hybrid_ring_cocoa.training.losses import (  # noqa: E402
    single_mode_control,
    ssim_loss,
    tv_2d,
)


UCLA_NP = np.array([0.9157, 0.3318, 0.0081, 0.2914, 0.0, 0.0], dtype=np.float32)
IKSUNG_NP = np.array([0.5, 0.3, 0.15, 0.1, 0.0, 0.2], dtype=np.float32)
T3_SMALL_NP = np.array([0.2, 0.1, 0.05, 0.05, 0.0, 0.1], dtype=np.float32)
NORMAL_BALANCED_NP = np.array([0.24, -0.08, 0.07, 0.06, 0.0, 0.08], dtype=np.float32)
NORMAL_COCOA_LIKE_NP = np.array(
    [0.30, -0.10, 0.05, 0.08, 0.0, 0.08], dtype=np.float32
)
SEIDEL_PRESETS = {
    "ucla": UCLA_NP,
    "iksung": IKSUNG_NP,
    "normal_mild": T3_SMALL_NP,
    "normal_balanced": NORMAL_BALANCED_NP,
    "normal_cocoa_like": NORMAL_COCOA_LIKE_NP,
}
SYS_PARAMS = {"NA": 0.45, "lamb": 0.55e-6}
TRACE_SEPARATED_CONVENTIONS = ("trace5", "trace4", "trace3")
CLASSICAL_CONVENTIONS = ("classical4d", "classical5d", "classical6d", "backend6")
SEIDEL_CONVENTIONS = (
    "classical4d",
    "classical5d",
    "classical6d",
    "backend6",
)


def fixed_seidel_indices_for_convention(convention: str) -> list[int]:
    """Return hard-fixed backend indices for classical/backend conventions."""

    if convention == "classical4d":
        return [4, 5]
    if convention == "classical5d":
        return [5]
    if convention in {"classical6d", "backend6"}:
        return []
    if convention == "trace5":
        return [5]
    if convention in {"trace4", "trace3"}:
        return [4, 5]
    raise ValueError(f"Unknown seidel convention {convention!r}")


def trace_model_dim(convention: str) -> int | None:
    if convention == "trace5":
        return 5
    if convention == "trace4":
        return 4
    if convention == "trace3":
        return 3
    if convention in CLASSICAL_CONVENTIONS:
        return None
    raise ValueError(f"Unknown seidel convention {convention!r}")


def active_backend_dim(convention: str) -> int:
    return 6 - len(fixed_seidel_indices_for_convention(convention))


def coerce_classical_backend(values: np.ndarray, convention: str, *, atol: float = 1e-8) -> np.ndarray:
    """Coerce compact or full classical backend vectors to full backend-6D."""

    fixed = fixed_seidel_indices_for_convention(convention)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 6:
        if fixed and np.max(np.abs(arr[fixed])) > atol:
            raise ValueError(
                f"{convention} fixes backend indices {fixed}; got nonzero values "
                f"{arr[fixed].tolist()} in a full backend vector."
            )
        out = arr.copy()
        if fixed:
            out[fixed] = 0.0
        return out

    expected = active_backend_dim(convention)
    if arr.size != expected:
        raise ValueError(
            f"{convention} expects either a compact {expected}D active backend "
            f"vector or a full 6D backend vector, got {arr.size} values"
        )
    out = np.zeros(6, dtype=np.float32)
    active = [idx for idx in range(6) if idx not in fixed]
    out[active] = arr
    return out


def convention_metadata(convention: str) -> dict[str, Any]:
    if convention in {"classical4d", "classical5d", "classical6d", "backend6"}:
        fixed = fixed_seidel_indices_for_convention(convention)
        return {
            "theta_convention": convention,
            "fixed_seidel_indices": fixed,
            "no_defocus": 5 in fixed,
            "no_w311_no_defocus": 4 in fixed and 5 in fixed,
            "distortion_forward_model": "disabled_W311_zero" if 4 in fixed else "backend_W311",
            "distortion_warp": False,
            "per_field_recenter": False,
            "trace_separated_status": "not_used_by_default",
        }
    if convention == "trace5":
        return {
            "theta_convention": "trace5",
            "fixed_seidel_indices": [5],
            "no_defocus": True,
            "no_w311_no_defocus": False,
            "distortion_forward_model": "frozen_backend_W311",
            "distortion_warp": False,
            "per_field_recenter": False,
            "trace_separated_status": "paused_internal_reproduction_only",
        }
    if convention in {"trace4", "trace3"}:
        return {
            "theta_convention": convention,
            "fixed_seidel_indices": [4, 5],
            "no_defocus": True,
            "no_w311_no_defocus": True,
            "distortion_forward_model": "disabled_W311_zero",
            "distortion_warp": False,
            "per_field_recenter": False,
            "trace_separated_status": "paused_internal_reproduction_only",
        }
    raise ValueError(f"Unknown seidel convention {convention!r}")


def parse_seidel_json(raw: str, *, convention: str) -> np.ndarray:
    """Parse a custom Seidel vector from JSON/list text."""

    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--gt-seidel-json is not valid JSON: {raw!r}") from exc
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("--gt-seidel-json contains non-finite values")
    if convention in CLASSICAL_CONVENTIONS:
        return coerce_classical_backend(arr, convention)
    expected = int(trace_model_dim(convention) or 6)
    if arr.size != expected:
        raise ValueError(
            f"--gt-seidel-json must contain exactly {expected} values for "
            f"--seidel-convention {convention}"
        )
    return arr


def preset_for_convention(preset: str, convention: str) -> np.ndarray:
    backend = SEIDEL_PRESETS[preset]
    if convention in CLASSICAL_CONVENTIONS:
        return coerce_classical_backend(backend, convention)
    dim = int(trace_model_dim(convention) or 4)
    return np.asarray(compress_trace_seidel(backend, model_dim=dim), dtype=np.float32)


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


BABOON = first_existing(
    PROJECT_ROOT / "hybrid_ring_cocoa" / "data" / "baboon.png",
    PROJECT_ROOT / "blur_fix_test" / "hybrid_ring_cocoa" / "data" / "baboon.png",
)
TEST_FIGURE_1 = first_existing(
    PROJECT_ROOT / "hybrid_ring_cocoa" / "data" / "Test_figure_1.png",
    PROJECT_ROOT
    / "hybrid_ring_cocoa"
    / "data"
    / "sharpe_simulation_figure_package"
    / "Test_figure_1.png",
    PROJECT_ROOT / "blur_fix_test" / "hybrid_ring_cocoa" / "data" / "Test_figure_1.png",
)
IKSUNG_BEADS = first_existing(
    PROJECT_ROOT
    / "hybrid_ring_cocoa"
    / "data"
    / "sharpe_simulation_figure_package"
    / "Iksung_beads.png",
    PROJECT_ROOT / "blur_fix_test" / "hybrid_ring_cocoa" / "data" / "Iksung_beads.png",
)
DENDRITES = first_existing(
    PROJECT_ROOT
    / "hybrid_ring_cocoa"
    / "data"
    / "sharpe_simulation_figure_package"
    / "dendrites.png",
    PROJECT_ROOT / "blur_fix_test" / "hybrid_ring_cocoa" / "data" / "dendrites.png",
)
DENDRITES_DENSE = first_existing(
    PROJECT_ROOT
    / "hybrid_ring_cocoa"
    / "data"
    / "sharpe_simulation_figure_package"
    / "dendrites_dense.png",
    PROJECT_ROOT
    / "blur_fix_test"
    / "hybrid_ring_cocoa"
    / "data"
    / "dendrites_dense.png",
)
IMAGE_PATHS = {
    "Test_figure_1": TEST_FIGURE_1,
    "fluorescence": TEST_FIGURE_1,
    "Iksung_beads": IKSUNG_BEADS,
    "dendrites": DENDRITES,
    "dendrites_dense": DENDRITES_DENSE,
    "baboon": BABOON,
}


def parse_nerf_skips(value: str | Sequence[int] | None) -> tuple[int, ...]:
    """Parse comma-separated skip-layer indices for NeuralObject2D."""
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null", "off", "[]"}:
            return ()
        parts = [part.strip() for part in text.split(",") if part.strip()]
    else:
        parts = list(value)

    try:
        skips = tuple(int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"nerf skips must be comma-separated integers or 'none', got {value!r}"
        ) from exc
    if any(skip <= 0 for skip in skips):
        raise argparse.ArgumentTypeError(
            f"nerf skips must be positive layer indices, got {value!r}"
        )
    return skips


def format_nerf_skips(skips: Sequence[int]) -> str:
    return "none" if not skips else ",".join(str(int(skip)) for skip in skips)


def normalize_nerf_capacity_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> argparse.Namespace:
    if int(args.nerf_depth) <= 0:
        parser.error(f"--nerf-depth must be positive, got {args.nerf_depth}")
    if int(args.nerf_width) < 2:
        parser.error(f"--nerf-width must be at least 2, got {args.nerf_width}")
    if hasattr(args, "fourier_num_angles") and int(args.fourier_num_angles) <= 0:
        parser.error(
            f"--fourier-num-angles must be positive, got {args.fourier_num_angles}"
        )
    if hasattr(args, "fourier_num_octaves") and int(args.fourier_num_octaves) <= 0:
        parser.error(
            f"--fourier-num-octaves must be positive, got {args.fourier_num_octaves}"
        )
    try:
        args.nerf_skips = parse_nerf_skips(args.nerf_skips)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


class CocoaLikeObject2D(NeuralObject2D):
    """NeuralObject2D with CoCoA-style positive, capped rendering."""

    def __init__(
        self,
        *,
        max_val: float = 40.0,
        beta: float = 1.0,
        output_mode: str = "softplus",
        depth: int = 6,
        width: int = 128,
        skips: Sequence[int] | None = (2, 4, 6),
        fourier_num_angles: int = 60,
        fourier_num_octaves: int = 7,
    ) -> None:
        fourier_num_angles = int(fourier_num_angles)
        fourier_num_octaves = int(fourier_num_octaves)
        super().__init__(
            in_features=2 * fourier_num_angles * fourier_num_octaves,
            depth=int(depth),
            width=int(width),
            skips=tuple(int(skip) for skip in (skips or ())),
        )
        self.fourier_num_angles = fourier_num_angles
        self.fourier_num_octaves = fourier_num_octaves
        self.max_val = float(max_val)
        self.beta = float(beta)
        self.output_mode = output_mode
        self._feature_cache: dict[
            tuple[int, int, int, int, str, str], torch.Tensor
        ] = {}

    def _features(self, width: int, height: int) -> torch.Tensor:
        reference = next(self.parameters())
        key = (
            width,
            height,
            self.fourier_num_angles,
            self.fourier_num_octaves,
            str(reference.device),
            str(reference.dtype),
        )
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached
        coords = make_coord_grid_2d(
            width, height, device=reference.device, dtype=reference.dtype
        )
        features = radial_fourier_encoding(
            coords,
            num_angles=self.fourier_num_angles,
            num_octaves=self.fourier_num_octaves,
        )
        self._feature_cache[key] = features
        return features

    def render_raw(self, width: int, height: int) -> torch.Tensor:
        raw = self.forward(self._features(width, height))
        return raw.view(width, height)

    def render(self, width: int, height: int) -> torch.Tensor:
        raw = self.render_raw(width, height)
        if self.output_mode == "sigmoid":
            out = self.max_val * torch.sigmoid(raw)
        elif self.output_mode == "softplus":
            out = F.softplus(raw, beta=self.beta)
        else:
            raise ValueError(f"Unknown output_mode={self.output_mode!r}")
        return torch.minimum(out, torch.full_like(out, self.max_val))


class CocoaLikeResult(NamedTuple):
    sharp_final: torch.Tensor
    seidel_final: torch.Tensor
    measurement_pred: torch.Tensor
    loss_history: list[float]
    ssim_history: list[float]
    rsd_history: list[float]
    tv_history: list[float]
    anchor_history: list[float]
    seidel_rms_floor_history: list[float]
    seidel_wavefront_rms_history: list[float]
    pretrain_history: list[float]
    elapsed_s: float


def reciprocal_std_contrast_loss(img: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """CoCoA's reciprocal contrast prior: reciprocal(std / mean)."""

    mean = torch.mean(img).clamp_min(eps)
    contrast = torch.std(img) / mean
    return torch.reciprocal(contrast.clamp_min(eps))


def _coerce_backend6_torch(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.reshape(-1)
    if theta.numel() >= 6:
        return theta[:6]
    return torch.cat([theta, torch.zeros(6 - theta.numel(), device=theta.device, dtype=theta.dtype)])


def torch_pupil_grid(
    samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if samples < 3:
        raise ValueError(f"pupil sample count must be >= 3, got {samples}")
    axis = torch.linspace(-1.0, 1.0, int(samples), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    rho2 = xx.square() + yy.square()
    mask = rho2 <= 1.0
    return xx[mask], rho2[mask]


def torch_field_weighted_wavefront_rms(
    theta: torch.Tensor,
    *,
    pupil_x: torch.Tensor,
    pupil_rho2: torch.Tensor,
    field_h: torch.Tensor,
    eps: float = 1e-18,
) -> torch.Tensor:
    """Differentiable field-weighted Seidel wavefront RMS in waves."""

    coeffs = _coerce_backend6_torch(theta)
    h = field_h.reshape(-1, 1)
    x = pupil_x.reshape(1, -1)
    rho2 = pupil_rho2.reshape(1, -1)
    h2 = h.square()
    wavefront = (
        coeffs[0] * rho2.square()
        + coeffs[1] * h * rho2 * x
        + coeffs[2] * h2 * x.square()
        + coeffs[3] * h2 * rho2
        + coeffs[4] * h2 * h * x
        + coeffs[5] * rho2
    )
    centered = wavefront - wavefront.mean(dim=1, keepdim=True)
    per_field_rms = torch.sqrt(centered.square().mean(dim=1).clamp_min(eps))
    weights = field_h.clone()
    weights[0] = 0.0
    return torch.sum(per_field_rms * weights) / weights.sum().clamp_min(eps)


def seidel_wavefront_np(coeffs: np.ndarray, x: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    if coeffs.size < 6:
        coeffs = np.pad(coeffs, (0, 6 - coeffs.size))
    coeffs = coeffs[:6]
    rho2 = x * x + y * y
    return (
        coeffs[0] * rho2**2
        + coeffs[1] * h * rho2 * x
        + coeffs[2] * h**2 * x**2
        + coeffs[3] * h**2 * rho2
        + coeffs[4] * h**3 * x
        + coeffs[5] * rho2
    )


def field_weighted_wavefront_rms_np(
    coeffs: np.ndarray,
    *,
    field_samples: int = 51,
    pupil_samples: int = 201,
) -> float:
    axis = np.linspace(-1.0, 1.0, int(pupil_samples), dtype=np.float64)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    mask = (x * x + y * y) <= 1.0
    hs = np.linspace(0.0, 1.0, int(field_samples), dtype=np.float64)
    weights = hs.copy()
    weights[0] = 0.0
    values: list[float] = []
    for h in hs:
        w = seidel_wavefront_np(coeffs, x, y, float(h))[mask]
        w = w - float(np.mean(w))
        values.append(float(np.sqrt(np.mean(w * w))))
    denom = float(np.sum(weights))
    if denom <= 0:
        return values[-1]
    return float(np.sum(np.asarray(values, dtype=np.float64) * weights) / denom)


def pretrain_cocoa_like(
    net_obj: CocoaLikeObject2D,
    measurement_gt: torch.Tensor,
    *,
    num_iter: int,
    lr: float,
    measurement_scalar: float,
    verbose: bool,
) -> list[float]:
    H, W = measurement_gt.shape
    target = measurement_scalar * measurement_gt.detach()
    optimizer = torch.optim.Adam(net_obj.parameters(), lr=lr)
    history: list[float] = []
    log_every = max(1, num_iter // 10)

    for step in range(num_iter):
        sharp = net_obj.render(H, W)
        loss = ssim_loss(sharp, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

        if verbose and (step % log_every == 0 or step == num_iter - 1):
            print(f"[pretrain {step:04d}] ssim={loss.item():.6f}", flush=True)

    return history


def train_cocoa_like(
    net_obj: CocoaLikeObject2D,
    seidel_coeffs: nn.Parameter,
    measurement_gt: torch.Tensor,
    sys_params: dict,
    *,
    mode: str,
    num_iter: int,
    lr_obj: float,
    lr_seidel: float,
    rsd_weight: float,
    tv_weight: float,
    defocus_anchor_weight: float,
    defocus_index: int,
    seidel_model_dim: int | None,
    fixed_seidel_indices: Sequence[int] | None,
    scheduler: str | None,
    eta_min_ratio: float,
    seidel_rms_prior_mode: str,
    seidel_rms_floor_weight: float,
    seidel_rms_floor_alpha: float,
    seidel_rms_floor_target: float | None,
    seidel_rms_floor_field_samples: int,
    seidel_rms_floor_pupil_samples: int,
    pretrain_history: list[float],
    verbose: bool,
) -> CocoaLikeResult:
    H, W = measurement_gt.shape
    resolved_sys = build_sys_params(H, sys_params)
    fixed = tuple(sorted({int(idx) for idx in (fixed_seidel_indices or [])}))
    fixed_mask: torch.Tensor | None = None
    if fixed:
        bad = [idx for idx in fixed if idx < 0 or idx >= int(seidel_coeffs.numel())]
        if bad:
            raise ValueError(f"fixed_seidel_indices out of range for parameter length {seidel_coeffs.numel()}: {bad}")
        fixed_mask = torch.ones_like(seidel_coeffs)
        fixed_mask[list(fixed)] = 0.0
        with torch.no_grad():
            seidel_coeffs[list(fixed)] = 0.0

    param_groups: list[dict] = [{"params": net_obj.parameters(), "lr": lr_obj}]
    if seidel_coeffs.requires_grad:
        param_groups.append({"params": [seidel_coeffs], "lr": lr_seidel})

    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)
    lr_scheduler = (
        CosineAnnealingLR(
            optimizer, T_max=num_iter, eta_min=lr_seidel * eta_min_ratio
        )
        if scheduler == "cosine"
        else None
    )

    psfs_cached: torch.Tensor | None = None
    if not seidel_coeffs.requires_grad:
        if seidel_model_dim is None:
            psfs_cached = get_trainable_ring_psfs(
                seidel_coeffs, H, resolved_sys, device=measurement_gt.device
            ).detach()
        else:
            psfs_cached = get_trainable_trace_ring_psfs(
                seidel_coeffs,
                H,
                resolved_sys,
                model_dim=seidel_model_dim,
                device=measurement_gt.device,
            ).detach()

    loss_history: list[float] = []
    ssim_history: list[float] = []
    rsd_history: list[float] = []
    tv_history: list[float] = []
    anchor_history: list[float] = []
    seidel_rms_floor_history: list[float] = []
    seidel_wavefront_rms_history: list[float] = []
    log_every = max(1, num_iter // 10)

    rms_prior_enabled = (
        seidel_coeffs.requires_grad
        and seidel_model_dim is None
        and float(seidel_rms_floor_weight) > 0.0
        and float(seidel_rms_floor_alpha) > 0.0
        and seidel_rms_floor_target is not None
        and float(seidel_rms_floor_target) > 0.0
    )
    if float(seidel_rms_floor_weight) > 0.0 and not rms_prior_enabled and verbose:
        print(
            "[warn] Seidel RMS prior disabled because the current mode/model "
            "does not expose trainable backend-6 Seidel coefficients or target RMS.",
            flush=True,
        )
    if rms_prior_enabled:
        pupil_x, pupil_rho2 = torch_pupil_grid(
            int(seidel_rms_floor_pupil_samples),
            device=measurement_gt.device,
            dtype=measurement_gt.dtype,
        )
        field_h = torch.linspace(
            0.0,
            1.0,
            int(seidel_rms_floor_field_samples),
            device=measurement_gt.device,
            dtype=measurement_gt.dtype,
        )
        rms_target = torch.as_tensor(
            float(seidel_rms_floor_target),
            device=measurement_gt.device,
            dtype=measurement_gt.dtype,
        )
    else:
        pupil_x = pupil_rho2 = field_h = rms_target = None

    sharp = torch.zeros_like(measurement_gt)
    measurement_pred = torch.zeros_like(measurement_gt)
    t0 = time.time()

    for step in range(num_iter):
        sharp = net_obj.render(H, W)
        seidel_for_forward = seidel_coeffs * fixed_mask if fixed_mask is not None else seidel_coeffs
        if psfs_cached is not None:
            measurement_pred = blur_ring_with_psfs(sharp, psfs_cached)
        elif seidel_model_dim is not None:
            measurement_pred = blur_ring_trace_trainable(
                sharp,
                seidel_for_forward,
                resolved_sys,
                model_dim=seidel_model_dim,
            )
        else:
            measurement_pred = blur_ring_trainable(sharp, seidel_for_forward, resolved_sys)

        loss_ssim = ssim_loss(measurement_pred, measurement_gt)
        loss_rsd = reciprocal_std_contrast_loss(sharp)
        loss_tv = tv_2d(sharp) if tv_weight != 0.0 else torch.zeros_like(loss_ssim)
        if (
            seidel_coeffs.requires_grad
            and seidel_model_dim is None
            and int(defocus_index) not in fixed
        ):
            loss_anchor = single_mode_control(seidel_coeffs, defocus_index, 0.0, 0.0)
        else:
            loss_anchor = torch.zeros_like(loss_ssim)

        if rms_prior_enabled:
            assert pupil_x is not None
            assert pupil_rho2 is not None
            assert field_h is not None
            assert rms_target is not None
            seidel_wavefront_rms = torch_field_weighted_wavefront_rms(
                seidel_for_forward,
                pupil_x=pupil_x,
                pupil_rho2=pupil_rho2,
                field_h=field_h,
            )
            if seidel_rms_prior_mode == "floor":
                floor_target = float(seidel_rms_floor_alpha) * rms_target
                loss_rms_floor = torch.relu(floor_target - seidel_wavefront_rms).square()
            elif seidel_rms_prior_mode == "ratio_target":
                recovered_over_target = seidel_wavefront_rms / rms_target.clamp_min(1e-12)
                loss_rms_floor = (
                    recovered_over_target - float(seidel_rms_floor_alpha)
                ).square()
            else:
                raise ValueError(f"Unknown seidel_rms_prior_mode={seidel_rms_prior_mode!r}")
        else:
            seidel_wavefront_rms = torch.zeros_like(loss_ssim)
            loss_rms_floor = torch.zeros_like(loss_ssim)

        loss = (
            loss_ssim
            + rsd_weight * loss_rsd
            + tv_weight * loss_tv
            + defocus_anchor_weight * loss_anchor
            + seidel_rms_floor_weight * loss_rms_floor
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if fixed and seidel_coeffs.requires_grad:
            with torch.no_grad():
                seidel_coeffs[list(fixed)] = 0.0
        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_history.append(float(loss.item()))
        ssim_history.append(float(loss_ssim.item()))
        rsd_history.append(float(loss_rsd.item()))
        tv_history.append(float(loss_tv.item()))
        anchor_history.append(float(loss_anchor.item()))
        seidel_rms_floor_history.append(float(loss_rms_floor.item()))
        seidel_wavefront_rms_history.append(float(seidel_wavefront_rms.item()))

        if verbose and (step % log_every == 0 or step == num_iter - 1):
            coeffs = ", ".join(f"{x:.4f}" for x in seidel_coeffs.detach().cpu())
            print(
                f"[{mode} train {step:04d}] total={loss.item():.6f} "
                f"ssim={loss_ssim.item():.6f} rsd={loss_rsd.item():.6f} "
                f"tv={loss_tv.item():.6f} anchor={loss_anchor.item():.6f} "
                f"rms_floor={loss_rms_floor.item():.6f} "
                f"wf_rms={seidel_wavefront_rms.item():.6f} "
                f"seidel=[{coeffs}]",
                flush=True,
            )

    elapsed = time.time() - t0
    return CocoaLikeResult(
        sharp_final=sharp.detach(),
        seidel_final=seidel_coeffs.detach().clone(),
        measurement_pred=measurement_pred.detach(),
        loss_history=loss_history,
        ssim_history=ssim_history,
        rsd_history=rsd_history,
        tv_history=tv_history,
        anchor_history=anchor_history,
        seidel_rms_floor_history=seidel_rms_floor_history,
        seidel_wavefront_rms_history=seidel_wavefront_rms_history,
        pretrain_history=pretrain_history,
        elapsed_s=elapsed,
    )


def high_frequency_ratio(arr: np.ndarray, cutoff: float = 0.25) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    centered = arr - float(np.mean(arr))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    total = float(np.sum(spectrum))
    if total <= 1e-20:
        return 0.0

    h, w = arr.shape
    yy, xx = np.mgrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = np.sqrt(cy**2 + cx**2)
    mask = radius >= cutoff * max_radius
    return float(np.sum(spectrum[mask]) / total)


def optimal_gain(target: np.ndarray, recon: np.ndarray) -> float:
    denom = float(np.sum(recon * recon))
    if denom <= 1e-20:
        return 0.0
    return float(np.sum(target * recon) / denom)


def tensor_metric(fn, *args, default: float | None = None, **kwargs) -> float | None:
    try:
        value = fn(*args, **kwargs)
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def summarize_array(prefix: str, arr: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_std_over_mean": float(std / max(abs(mean), 1e-12)),
        f"{prefix}_hf_ratio": high_frequency_ratio(arr),
    }


def compute_metrics(
    sharp_gt: torch.Tensor,
    meas_gt: torch.Tensor,
    result: CocoaLikeResult,
    gt_seidel: np.ndarray,
    *,
    mode: str,
    args: argparse.Namespace,
) -> dict:
    sgt = sharp_gt.detach().cpu().numpy()
    mgt = meas_gt.detach().cpu().numpy()
    sr = result.sharp_final.detach().cpu().numpy()
    mp = result.measurement_pred.detach().cpu().numpy()
    gain = optimal_gain(sgt, sr)
    sr_gain = gain * sr
    sr_gain_t = torch.as_tensor(sr_gain, dtype=sharp_gt.dtype, device=sharp_gt.device)

    seidel_final = result.seidel_final.detach().cpu().numpy()
    metrics = {
        "mode": mode,
        "image": args.image,
        "size": args.size,
        "gt_preset": args.gt_preset,
        "gt_label": args.gt_label,
        "gt_source": args.gt_source,
        "seidel_convention": args.seidel_convention,
        **convention_metadata(args.seidel_convention),
        "elapsed_s": result.elapsed_s,
        "final_loss": float(result.loss_history[-1]),
        "final_ssim_loss": float(result.ssim_history[-1]),
        "final_rsd_loss": float(result.rsd_history[-1]),
        "final_tv_loss": float(result.tv_history[-1]),
        "final_anchor_loss": float(result.anchor_history[-1]),
        "final_seidel_rms_floor_loss": float(result.seidel_rms_floor_history[-1]),
        "final_seidel_wavefront_rms_floor_estimate": float(
            result.seidel_wavefront_rms_history[-1]
        ),
        "seidel_rms_prior_mode": str(args.seidel_rms_prior_mode),
        "seidel_rms_floor_weight": float(args.seidel_rms_floor_weight),
        "seidel_rms_floor_alpha": float(args.seidel_rms_floor_alpha),
        "seidel_rms_floor_target": (
            None if args.seidel_rms_floor_target is None else float(args.seidel_rms_floor_target)
        ),
        "seidel_rms_floor_field_samples": int(args.seidel_rms_floor_field_samples),
        "seidel_rms_floor_pupil_samples": int(args.seidel_rms_floor_pupil_samples),
        "best_gain_recon_to_gt": gain,
        "nrmse_recon_raw_vs_gt": compute_nrmse(sgt, sr),
        "nrmse_recon_gain_vs_gt": compute_nrmse(sgt, sr_gain),
        "nrmse_meas_pred_vs_meas": compute_nrmse(mgt, mp),
        "ssim_recon_raw_vs_gt": tensor_metric(
            ssim_window, result.sharp_final, sharp_gt, val_range=1.0
        ),
        "ssim_recon_gain_vs_gt": tensor_metric(
            ssim_window, sr_gain_t, sharp_gt, val_range=1.0
        ),
        "msssim_recon_gain_vs_gt": tensor_metric(
            msssim, sr_gain_t, sharp_gt, val_range=1.0, normalize="relu"
        ),
        "ssim_meas_pred_vs_meas": tensor_metric(
            ssim_window, result.measurement_pred, meas_gt, val_range=1.0
        ),
        "l2_seidel_vs_gt": float(np.linalg.norm(seidel_final - gt_seidel)),
        "seidel_final": seidel_final.tolist(),
        "seidel_gt": gt_seidel.tolist(),
        "config": vars(args),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }
    metrics.update(summarize_array("gt", sgt))
    metrics.update(summarize_array("measurement", mgt))
    metrics.update(summarize_array("recon_raw", sr))
    metrics.update(summarize_array("recon_gain", sr_gain))
    metrics.update(summarize_array("pred_measurement", mp))
    return metrics


def as_percentile_image(arr: np.ndarray, lo: float = 1.0, hi: float = 99.7) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    p0, p1 = np.percentile(arr, [lo, hi])
    if p1 <= p0:
        return np.zeros_like(arr)
    return np.clip((arr - p0) / (p1 - p0), 0.0, 1.0)


def save_mode_figures(
    out_dir: Path,
    sharp_gt: torch.Tensor,
    meas_gt: torch.Tensor,
    result: CocoaLikeResult,
    metrics: dict,
    *,
    title: str,
) -> None:
    sgt = sharp_gt.detach().cpu().numpy()
    mgt = meas_gt.detach().cpu().numpy()
    sr = result.sharp_final.detach().cpu().numpy()
    mp = result.measurement_pred.detach().cpu().numpy()
    sr_gain = metrics["best_gain_recon_to_gt"] * sr
    err = np.abs(sgt - sr_gain)

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    panels = [
        (sgt, "Sharp GT", "gray", 0.0, 1.0),
        (mgt, "Measurement", "gray", None, None),
        (np.clip(sr, 0.0, 1.0), "Recon raw clipped", "gray", 0.0, 1.0),
        (as_percentile_image(sr), "Recon percentile", "gray", 0.0, 1.0),
        (mp, "Predicted measurement", "gray", None, None),
        (err, "Gain-aligned abs error", "magma", None, None),
    ]
    for a, (im, label, cmap, vmin, vmax) in zip(ax.flat, panels):
        a.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
        a.set_title(label)
        a.axis("off")
    fig.suptitle(
        f"{title}\n"
        f"SSIM_gain={metrics['ssim_recon_gain_vs_gt']:.4f}  "
        f"NRMSE_gain={metrics['nrmse_recon_gain_vs_gt']:.4f}  "
        f"HF raw={metrics['recon_raw_hf_ratio']:.4f}  "
        f"HF meas={metrics['measurement_hf_ratio']:.4f}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if result.pretrain_history:
        ax.plot(result.pretrain_history, label="pretrain ssim", alpha=0.75)
    offset = len(result.pretrain_history)
    x = np.arange(len(result.loss_history)) + offset
    ax.plot(x, result.loss_history, label="joint total")
    ax.plot(x, result.ssim_history, label="joint ssim", alpha=0.75)
    ax.plot(x, np.asarray(result.rsd_history) * metrics["config"]["rsd_weight"],
            label="weighted rsd", alpha=0.75)
    if metrics["config"].get("seidel_rms_floor_weight", 0.0) > 0.0:
        ax.plot(
            x,
            np.asarray(result.seidel_rms_floor_history)
            * metrics["config"]["seidel_rms_floor_weight"],
            label="weighted seidel rms floor",
            alpha=0.75,
        )
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=140)
    plt.close(fig)


def save_summary_figure(
    root_dir: Path,
    sharp_gt: torch.Tensor,
    meas_gt: torch.Tensor,
    rows: list[tuple[str, CocoaLikeResult, dict]],
) -> None:
    if not rows:
        return
    sgt = sharp_gt.detach().cpu().numpy()
    mgt = meas_gt.detach().cpu().numpy()
    nrows = len(rows)
    fig, ax = plt.subplots(nrows, 5, figsize=(16, 3.6 * nrows), squeeze=False)

    for row_idx, (mode, result, metrics) in enumerate(rows):
        sr = result.sharp_final.detach().cpu().numpy()
        mp = result.measurement_pred.detach().cpu().numpy()
        sr_gain = metrics["best_gain_recon_to_gt"] * sr
        err = np.abs(sgt - sr_gain)
        panels = [
            (sgt, "GT", "gray", 0.0, 1.0),
            (mgt, "Measurement", "gray", None, None),
            (as_percentile_image(sr), f"{mode} recon", "gray", 0.0, 1.0),
            (mp, "Pred meas", "gray", None, None),
            (err, "Gain abs err", "magma", None, None),
        ]
        for col_idx, (im, label, cmap, vmin, vmax) in enumerate(panels):
            a = ax[row_idx, col_idx]
            a.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
            a.set_title(label)
            a.axis("off")
        ax[row_idx, 0].set_ylabel(
            f"SSIM={metrics['ssim_recon_gain_vs_gt']:.3f}\n"
            f"HF={metrics['recon_raw_hf_ratio']:.3f}",
            rotation=0,
            labelpad=44,
            va="center",
        )
    fig.tight_layout()
    fig.savefig(root_dir / "summary_comparison.png", dpi=140)
    plt.close(fig)


def run_one_mode(
    args: argparse.Namespace,
    *,
    mode: str,
    sharp_gt: torch.Tensor,
    meas_gt: torch.Tensor,
    gt_vec: torch.Tensor,
    gt_np: np.ndarray,
    root_dir: Path,
    device: torch.device,
) -> tuple[CocoaLikeResult, dict]:
    out_dir = root_dir / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net_obj = CocoaLikeObject2D(
        max_val=args.max_val,
        beta=args.nerf_beta,
        output_mode=args.output_mode,
        depth=args.nerf_depth,
        width=args.nerf_width,
        skips=args.nerf_skips,
        fourier_num_angles=args.fourier_num_angles,
        fourier_num_octaves=args.fourier_num_octaves,
    ).to(device)

    if mode == "frozen":
        seidel = nn.Parameter(gt_vec.detach().clone())
        seidel.requires_grad_(False)
    elif mode == "joint":
        dim = int(trace_model_dim(args.seidel_convention) or 6)
        seidel = nn.Parameter(torch.zeros(dim, device=device, dtype=sharp_gt.dtype))
    else:
        raise ValueError(f"Unknown mode={mode!r}")

    print(
        f"[start] mode={mode} image={args.image} size={args.size} "
        f"device={device} pretrain={args.pretrain_iter} joint={args.num_iter} "
        f"mlp={args.nerf_depth}x{args.nerf_width} "
        f"skips={format_nerf_skips(args.nerf_skips)} "
        f"fourier={args.fourier_num_angles}x{args.fourier_num_octaves} "
        f"rms_prior_mode={args.seidel_rms_prior_mode} "
        f"rms_floor_weight={args.seidel_rms_floor_weight:g} "
        f"rms_floor_alpha={args.seidel_rms_floor_alpha:g} "
        f"rms_floor_target={args.seidel_rms_floor_target}",
        flush=True,
    )
    t0 = time.time()
    pretrain_history = pretrain_cocoa_like(
        net_obj,
        meas_gt,
        num_iter=args.pretrain_iter,
        lr=args.lr_obj,
        measurement_scalar=args.pretrain_scalar,
        verbose=args.verbose,
    )
    result = train_cocoa_like(
        net_obj,
        seidel,
        meas_gt,
        SYS_PARAMS,
        mode=mode,
        num_iter=args.num_iter,
        lr_obj=args.lr_obj,
        lr_seidel=args.lr_seidel,
        rsd_weight=args.rsd_weight,
        tv_weight=args.tv_weight,
        defocus_anchor_weight=args.defocus_anchor_weight,
        defocus_index=args.defocus_index,
        seidel_model_dim=trace_model_dim(args.seidel_convention),
        fixed_seidel_indices=(
            fixed_seidel_indices_for_convention(args.seidel_convention)
            if trace_model_dim(args.seidel_convention) is None
            else []
        ),
        scheduler=args.scheduler,
        eta_min_ratio=args.eta_min_ratio,
        seidel_rms_prior_mode=args.seidel_rms_prior_mode,
        seidel_rms_floor_weight=args.seidel_rms_floor_weight,
        seidel_rms_floor_alpha=args.seidel_rms_floor_alpha,
        seidel_rms_floor_target=args.seidel_rms_floor_target,
        seidel_rms_floor_field_samples=args.seidel_rms_floor_field_samples,
        seidel_rms_floor_pupil_samples=args.seidel_rms_floor_pupil_samples,
        pretrain_history=pretrain_history,
        verbose=args.verbose,
    )
    result = result._replace(elapsed_s=time.time() - t0)
    metrics = compute_metrics(sharp_gt, meas_gt, result, gt_np, mode=mode, args=args)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(
        {
            "sharp_gt": sharp_gt.detach().cpu(),
            "measurement_gt": meas_gt.detach().cpu(),
            "sharp_recon": result.sharp_final.detach().cpu(),
            "measurement_pred": result.measurement_pred.detach().cpu(),
            "seidel_final": result.seidel_final.detach().cpu(),
            "loss_history": result.loss_history,
            "ssim_history": result.ssim_history,
            "rsd_history": result.rsd_history,
            "tv_history": result.tv_history,
            "anchor_history": result.anchor_history,
            "seidel_rms_floor_history": result.seidel_rms_floor_history,
            "seidel_wavefront_rms_history": result.seidel_wavefront_rms_history,
            "pretrain_history": result.pretrain_history,
        },
        out_dir / "tensors.pt",
    )
    save_mode_figures(out_dir, sharp_gt, meas_gt, result, metrics, title=mode)

    print(
        f"[done] mode={mode} "
        f"SSIM_gain={metrics['ssim_recon_gain_vs_gt']:.4f} "
        f"NRMSE_gain={metrics['nrmse_recon_gain_vs_gt']:.4f} "
        f"HF_recon={metrics['recon_raw_hf_ratio']:.4f} "
        f"HF_meas={metrics['measurement_hf_ratio']:.4f} "
        f"L2_seidel={metrics['l2_seidel_vs_gt']:.4f} "
        f"elapsed={metrics['elapsed_s']:.1f}s",
        flush=True,
    )
    return result, metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", choices=sorted(IMAGE_PATHS), default="Test_figure_1")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--modes", nargs="+", choices=["joint", "frozen"], default=["joint"])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--num-iter", type=int, default=1000)
    ap.add_argument("--pretrain-iter", type=int, default=400)
    ap.add_argument("--lr-obj", type=float, default=5e-3)
    ap.add_argument("--lr-seidel", type=float, default=1e-2)
    ap.add_argument("--rsd-weight", type=float, default=5e-4)
    ap.add_argument("--tv-weight", type=float, default=0.0)
    ap.add_argument("--pretrain-scalar", type=float, default=5.0)
    ap.add_argument("--defocus-anchor-weight", type=float, default=1.0)
    ap.add_argument("--defocus-index", type=int, default=5)
    ap.add_argument(
        "--seidel-rms-prior-mode",
        choices=["floor", "ratio_target"],
        default="floor",
        help=(
            "Seidel RMS prior form. 'floor' uses max(0, alpha*target - recovered)^2; "
            "'ratio_target' uses (recovered/target - alpha)^2."
        ),
    )
    ap.add_argument(
        "--seidel-rms-floor-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a hinge prior that penalizes recovered Seidel wavefront "
            "RMS below alpha * target. Set to 0 to disable."
        ),
    )
    ap.add_argument(
        "--seidel-rms-floor-alpha",
        type=float,
        default=0.8,
        help="Floor fraction of the target wavefront RMS used by the Seidel RMS prior.",
    )
    ap.add_argument(
        "--seidel-rms-floor-target",
        type=float,
        default=None,
        help=(
            "Target wavefront RMS for the floor prior. If omitted and the prior "
            "is enabled, it is computed from the GT Seidel vector."
        ),
    )
    ap.add_argument("--seidel-rms-floor-field-samples", type=int, default=21)
    ap.add_argument("--seidel-rms-floor-pupil-samples", type=int, default=51)
    ap.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    ap.add_argument("--eta-min-ratio", type=float, default=1.0 / 25.0)
    ap.add_argument("--max-val", type=float, default=40.0)
    ap.add_argument("--nerf-beta", type=float, default=1.0)
    ap.add_argument("--nerf-depth", type=int, default=6)
    ap.add_argument("--nerf-width", type=int, default=128)
    ap.add_argument(
        "--nerf-skips",
        default="2,4,6",
        help="Comma-separated NeuralObject2D skip indices, or 'none'.",
    )
    ap.add_argument("--fourier-num-angles", type=int, default=60)
    ap.add_argument("--fourier-num-octaves", type=int, default=7)
    ap.add_argument("--output-mode", choices=["softplus", "sigmoid"], default="softplus")
    ap.add_argument("--gt-preset", choices=sorted(SEIDEL_PRESETS), default="ucla")
    ap.add_argument(
        "--gt-seidel-json",
        default=None,
        help="Custom Seidel vector JSON in the convention selected by --seidel-convention.",
    )
    ap.add_argument(
        "--seidel-convention",
        choices=SEIDEL_CONVENTIONS,
        default="classical6d",
        help=(
            "Seidel recovery convention. Defaults to classical6d/backend "
            "[W040,W131,W222,W220,W311,Wd]. Classical fixed-index variants are "
            "classical5d (Wd fixed) and classical4d (W311,Wd fixed). "
            "Trace-separated trace5/trace4/trace3 are paused internal helpers "
            "and are not exposed by this primary CLI."
        ),
    )
    ap.add_argument(
        "--gt-label",
        default=None,
        help="Human-readable label for a custom Seidel vector.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    return normalize_nerf_capacity_args(ap, ap.parse_args(argv))


def main() -> None:
    args = parse_args()
    args.scheduler = None if args.scheduler == "none" else args.scheduler
    if args.seidel_rms_floor_weight < 0.0:
        raise ValueError("--seidel-rms-floor-weight must be non-negative")
    if args.seidel_rms_floor_alpha < 0.0:
        raise ValueError("--seidel-rms-floor-alpha must be non-negative")
    if args.seidel_rms_floor_field_samples < 2:
        raise ValueError("--seidel-rms-floor-field-samples must be >= 2")
    if args.seidel_rms_floor_pupil_samples < 3:
        raise ValueError("--seidel-rms-floor-pupil-samples must be >= 3")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.gt_seidel_json is not None:
        gt_np = parse_seidel_json(args.gt_seidel_json, convention=args.seidel_convention)
        args.gt_source = "custom"
        args.gt_label = args.gt_label or "custom"
    else:
        gt_np = preset_for_convention(args.gt_preset, args.seidel_convention)
        args.gt_source = "preset"
        args.gt_label = args.gt_label or args.gt_preset
    if args.seidel_rms_floor_weight > 0.0 and args.seidel_rms_floor_target is None:
        args.seidel_rms_floor_target = field_weighted_wavefront_rms_np(gt_np)
    gt_vec = torch.tensor(gt_np, device=device, dtype=torch.float32)
    img_path = IMAGE_PATHS[args.image]

    run_name = args.run_name
    if run_name is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (
            f"cocoa_like2d_{args.image}{args.size}_{args.gt_label}_"
            f"pre{args.pretrain_iter}_joint{args.num_iter}_{stamp}"
        )

    root_dir = PROJECT_ROOT / "outputs" / "cocoa_like_2d_mechanism" / run_name
    root_dir.mkdir(parents=True, exist_ok=True)

    sharp_gt = load_baboon_gt(args.size, path=img_path, device=device)
    model_dim = trace_model_dim(args.seidel_convention)
    if model_dim is None:
        meas_gt = synthesize_measurement(sharp_gt, gt_vec, SYS_PARAMS)
    else:
        meas_gt = synthesize_trace_measurement(
            sharp_gt,
            gt_vec,
            SYS_PARAMS,
            model_dim=model_dim,
        )
    print(
        f"[measurement] gt_hf={high_frequency_ratio(sharp_gt.detach().cpu().numpy()):.6f} "
        f"meas_hf={high_frequency_ratio(meas_gt.detach().cpu().numpy()):.6f} "
        f"meas_min={float(meas_gt.min()):.6f} meas_max={float(meas_gt.max()):.6f}",
        flush=True,
    )

    rows: list[tuple[str, CocoaLikeResult, dict]] = []
    summary: dict[str, dict] = {
        "run_name": run_name,
        "root_dir": str(root_dir),
        "args": vars(args),
        "modes": {},
    }

    for mode in args.modes:
        result, metrics = run_one_mode(
            args,
            mode=mode,
            sharp_gt=sharp_gt,
            meas_gt=meas_gt,
            gt_vec=gt_vec,
            gt_np=gt_np,
            root_dir=root_dir,
            device=device,
        )
        summary["modes"][mode] = metrics
        rows.append((mode, result, metrics))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_summary_figure(root_dir, sharp_gt, meas_gt, rows)
    (root_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[summary] wrote {root_dir}", flush=True)


if __name__ == "__main__":
    main()
