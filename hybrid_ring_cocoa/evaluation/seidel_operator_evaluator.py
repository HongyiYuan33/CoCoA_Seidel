"""Physical-operator evaluation for recovered Seidel coefficients.

The primary metric in this module compares the exact image-space ring
convolution forward operator, ``A(theta)``, through deterministic probe images.
When ``OperatorProbeConfig.full_delta_basis`` is enabled, the probe set is the
full pixel delta basis and the metric becomes a deterministic Frobenius-norm
operator proxy. Otherwise it is a fixed, weighted probe approximation using
delta, radial/ring, Fourier, and fixed-seed random probe groups.

``operator_error_calibrated`` is strict only at the calibrated forward-operator
level. It is not a strict raw coefficient success criterion; raw coefficient
and wavefront residuals are reported separately.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from hybrid_ring_cocoa._rdm._src.psf_model import compute_rdm_psfs
from hybrid_ring_cocoa.optics.ring_forward import blur_ring_with_psfs
from hybrid_ring_cocoa.optics.seidel_psf import (
    build_sys_params,
    compress_trace_seidel,
    expand_trace_seidel,
    get_reference_ring_psfs,
    get_trainable_ring_psfs,
    normalize_seidel_coeffs,
)

SEIDEL_COEFF_NAMES: tuple[str, ...] = ("W040", "W131", "W222", "W220", "W311", "Wd")
NUM_SEIDEL = 6

SEIDEL_TRANSFORM_SIGNS: dict[str, tuple[float, ...]] = {
    "I": (+1.0, +1.0, +1.0, +1.0, +1.0, +1.0),
    "mirror_x": (+1.0, -1.0, +1.0, +1.0, -1.0, +1.0),
    "twin": (-1.0, +1.0, -1.0, -1.0, +1.0, -1.0),
    "twin_mirror": (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
}
OPERATOR_TRANSFORM_ORDER: tuple[str, ...] = ("I", "twin", "mirror_x", "twin_mirror")
COORDINATE_DIAGNOSTIC_TRANSFORMS: tuple[str, ...] = (
    "I",
    "mirror_x",
    "twin",
    "twin_mirror",
)
GAUGE_TRANSFORM_SET: tuple[str, ...] = COORDINATE_DIAGNOSTIC_TRANSFORMS
GAUGE_TRANSFORM_ALIASES: dict[str, tuple[str, ...]] = {
    "I": ("identity", "y_reflection"),
    "mirror_x": ("x_reflection", "rot180"),
    "twin": ("phase_conjugate_twin",),
    "twin_mirror": ("phase_conjugate_twin_mirror",),
}
# Trace-separated evaluator helpers are paused for default experiments and
# retained only for explicit reproduction/diagnostics of previous trace runs.
TRACE5_TRANSFORM_SIGNS: dict[str, tuple[float, ...]] = {
    "I": (+1.0, +1.0, +1.0, +1.0, +1.0),
    "mirror_x": (+1.0, -1.0, +1.0, +1.0, -1.0),
    "twin": (-1.0, +1.0, -1.0, -1.0, +1.0),
    "twin_mirror": (-1.0, -1.0, -1.0, -1.0, -1.0),
}
TRACE4_TRANSFORM_SIGNS: dict[str, tuple[float, ...]] = {
    "I": (+1.0, +1.0, +1.0, +1.0),
    "mirror_x": (+1.0, -1.0, +1.0, +1.0),
    "twin": (-1.0, +1.0, -1.0, -1.0),
    "twin_mirror": (-1.0, -1.0, -1.0, -1.0),
}
TRACE3_TRANSFORM_SIGNS: dict[str, tuple[float, ...]] = {
    "I": (+1.0, +1.0, +1.0),
    "mirror_x": (+1.0, -1.0, +1.0),
    "twin": (-1.0, +1.0, -1.0),
    "twin_mirror": (-1.0, -1.0, -1.0),
}


def _default_group_weights() -> dict[str, float]:
    return {
        "delta_grid": 1.0,
        "radial_basis": 1.0,
        "fourier": 1.0,
        "random": 1.0,
        "full_delta": 1.0,
    }


@dataclass(frozen=True)
class OperatorProbeConfig:
    """Configuration for deterministic operator-probe evaluation.

    All generated probes are L2-normalized before they are passed through the
    forward operator. Output images are not normalized independently; each group
    contributes its weighted mean squared output residual and reference energy.
    """

    delta_grid_size: int = 3
    delta_grid_margin_fraction: float = 0.125
    radial_basis_count: int = 4
    radial_basis_width_fraction: float = 0.085
    fourier_frequencies: tuple[tuple[int, int], ...] = (
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 0),
        (0, 2),
    )
    random_count: int = 4
    random_seed: int = 1729
    random_distribution: str = "rademacher"
    group_weights: Mapping[str, float] = field(default_factory=_default_group_weights)
    full_delta_basis: bool = False
    input_l2_normalize: bool = True
    patch_size: int = 0
    eps: float = 1e-12
    twin_invariance_tol: float = 1e-5
    tie_operator_tol: float = 1e-8
    wavefront_field_samples: int = 41
    wavefront_pupil_samples: int = 151
    diagnostic_psf_points: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.25, -0.25),
        (0.5, -0.5),
        (0.75, -0.75),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_grid_size": int(self.delta_grid_size),
            "delta_grid_margin_fraction": float(self.delta_grid_margin_fraction),
            "radial_basis_count": int(self.radial_basis_count),
            "radial_basis_width_fraction": float(self.radial_basis_width_fraction),
            "fourier_frequencies": [
                [int(kx), int(ky)] for kx, ky in self.fourier_frequencies
            ],
            "random_count": int(self.random_count),
            "random_seed": int(self.random_seed),
            "random_distribution": str(self.random_distribution),
            "group_weights": {
                key: float(self.group_weights[key]) for key in sorted(self.group_weights)
            },
            "full_delta_basis": bool(self.full_delta_basis),
            "input_l2_normalize": bool(self.input_l2_normalize),
            "patch_size": int(self.patch_size),
            "eps": float(self.eps),
            "twin_invariance_tol": float(self.twin_invariance_tol),
            "tie_operator_tol": float(self.tie_operator_tol),
            "wavefront_field_samples": int(self.wavefront_field_samples),
            "wavefront_pupil_samples": int(self.wavefront_pupil_samples),
            "diagnostic_psf_points": [
                [float(x), float(y)] for x, y in self.diagnostic_psf_points
            ],
        }

    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def resolved_group_weights(self) -> dict[str, float]:
        defaults = _default_group_weights()
        defaults.update({str(k): float(v) for k, v in self.group_weights.items()})
        return defaults


def coerce_seidel_vector(
    theta: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    fixed_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Return a 6-vector, accepting compact 5D/4D vectors when fixed indices are known."""

    arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    fixed = normalize_fixed_indices(fixed_indices)
    if arr.size == NUM_SEIDEL:
        out = arr.astype(np.float64, copy=True)
    elif fixed and arr.size + len(fixed) == NUM_SEIDEL:
        out = np.zeros(NUM_SEIDEL, dtype=np.float64)
        active = [idx for idx in range(NUM_SEIDEL) if idx not in fixed]
        out[active] = arr
    else:
        raise ValueError(
            "Expected a 6-element Seidel vector, or a compact vector whose "
            f"length plus fixed_indices equals 6. Got length {arr.size} with "
            f"fixed_indices={fixed}."
        )
    for idx in fixed:
        out[idx] = 0.0
    return out


def normalize_fixed_indices(fixed_indices: Sequence[int] | None) -> tuple[int, ...]:
    if fixed_indices is None:
        return ()
    fixed = sorted({int(idx) for idx in fixed_indices})
    invalid = [idx for idx in fixed if idx < 0 or idx >= NUM_SEIDEL]
    if invalid:
        raise ValueError(f"Invalid Seidel fixed indices: {invalid}")
    return tuple(fixed)


def apply_seidel_transform(
    theta: Sequence[float] | np.ndarray | torch.Tensor,
    transform: str,
    *,
    fixed_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Apply one hard-coded Seidel sign transform and re-enforce fixed terms."""

    if transform not in SEIDEL_TRANSFORM_SIGNS:
        raise ValueError(f"Unknown Seidel transform {transform!r}")
    full = coerce_seidel_vector(theta, fixed_indices=fixed_indices)
    out = full * np.asarray(SEIDEL_TRANSFORM_SIGNS[transform], dtype=np.float64)
    for idx in normalize_fixed_indices(fixed_indices):
        out[idx] = 0.0
    return out


def _normalize_trace_dim(model_dim: int | str | None, length: int | None = None) -> int:
    if isinstance(model_dim, str):
        aliases = {
            "trace5": 5,
            "5d": 5,
            "trace5_no_defocus": 5,
            "no_defocus_5d": 5,
            "trace4": 4,
            "4d": 4,
            "no_distortion_no_defocus_4d": 4,
            "trace3": 3,
            "3d": 3,
            "no_field_curvature_3d": 3,
            "strict_no_scalar_quadratic": 3,
        }
        key = model_dim.strip().lower()
        if key not in aliases:
            raise ValueError(f"Unknown trace Seidel model_dim={model_dim!r}")
        model_dim = aliases[key]
    if model_dim is None:
        if length not in (3, 4, 5):
            raise ValueError("Trace Seidel model_dim must be explicit for this input")
        return int(length)
    if int(model_dim) not in (3, 4, 5):
        raise ValueError(f"Trace Seidel model_dim must be 3, 4, or 5, got {model_dim}")
    return int(model_dim)


def coerce_trace_vector(
    theta_trace: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    model_dim: int | str | None = None,
) -> np.ndarray:
    arr = np.asarray(theta_trace, dtype=np.float64).reshape(-1)
    dim = _normalize_trace_dim(model_dim, int(arr.size))
    if arr.size != dim:
        raise ValueError(f"Expected theta_trace{dim} length {dim}, got {arr.size}")
    return arr.astype(np.float64, copy=True)


def apply_trace_transform(
    theta_trace: Sequence[float] | np.ndarray | torch.Tensor,
    transform: str,
    *,
    model_dim: int | str | None = None,
) -> np.ndarray:
    """Apply a public trace-space sign transform before backend expansion."""
    trace = coerce_trace_vector(theta_trace, model_dim=model_dim)
    dim = _normalize_trace_dim(model_dim, int(trace.size))
    signs = {
        5: TRACE5_TRANSFORM_SIGNS,
        4: TRACE4_TRANSFORM_SIGNS,
        3: TRACE3_TRANSFORM_SIGNS,
    }[dim]
    if transform not in signs:
        raise ValueError(f"Unknown trace Seidel transform {transform!r}")
    return trace * np.asarray(signs[transform], dtype=np.float64)


def _seidel_wavefront(theta: np.ndarray, x: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    rho2 = x * x + y * y
    return (
        theta[0] * rho2**2
        + theta[1] * h * rho2 * x
        + theta[2] * h**2 * x**2
        + theta[3] * h**2 * rho2
        + theta[4] * h**3 * x
        + theta[5] * rho2
    )


def field_weighted_wavefront_rms(
    theta: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    field_samples: int = 41,
    pupil_samples: int = 151,
) -> float:
    """Field-weighted scalar Seidel wavefront RMS used by the sweep diagnostics."""

    coeffs = coerce_seidel_vector(theta)
    x1 = np.linspace(-1.0, 1.0, int(pupil_samples), dtype=np.float64)
    x, y = np.meshgrid(x1, x1, indexing="xy")
    mask = (x * x + y * y) <= 1.0
    hs = np.linspace(0.0, 1.0, int(field_samples), dtype=np.float64)
    weights = hs.copy()
    weights[0] = 0.0
    rms_values: list[float] = []
    for h in hs:
        w = _seidel_wavefront(coeffs, x, y, float(h))[mask]
        w = w - float(np.mean(w))
        rms_values.append(math.sqrt(float(np.mean(w * w))))
    rms = np.asarray(rms_values, dtype=np.float64)
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return float(rms[-1])
    return float(np.sum(rms * weights) / denom)


def relative_wavefront_residual(
    theta_ref: Sequence[float] | np.ndarray | torch.Tensor,
    theta_candidate: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    field_samples: int,
    pupil_samples: int,
    eps: float,
) -> float:
    ref = coerce_seidel_vector(theta_ref)
    cand = coerce_seidel_vector(theta_candidate)
    num = field_weighted_wavefront_rms(
        cand - ref,
        field_samples=field_samples,
        pupil_samples=pupil_samples,
    )
    den = field_weighted_wavefront_rms(
        ref,
        field_samples=field_samples,
        pupil_samples=pupil_samples,
    )
    return float(num / max(den, eps))


def coefficient_residuals(
    theta_ref: Sequence[float] | np.ndarray | torch.Tensor,
    theta_candidate: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    eps: float,
) -> tuple[float, float]:
    ref = coerce_seidel_vector(theta_ref)
    cand = coerce_seidel_vector(theta_candidate)
    absolute = float(np.linalg.norm(cand - ref))
    relative = float(absolute / max(float(np.linalg.norm(ref)), eps))
    return absolute, relative


def validate_hardcoded_transform_wavefronts(
    *,
    theta: Sequence[float] | None = None,
    grid_size: int = 65,
    field_values: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Validate hard-coded sign transforms against direct wavefront definitions."""

    coeffs = coerce_seidel_vector(
        theta
        if theta is not None
        else np.asarray([0.17, -0.09, 0.13, 0.07, -0.05, 0.11], dtype=np.float64)
    )
    axis = np.linspace(-1.0, 1.0, int(grid_size), dtype=np.float64)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    mask = (x * x + y * y) <= 1.0

    max_errors: dict[str, float] = {}
    for transform in ("I", "mirror_x", "twin", "twin_mirror"):
        transformed = apply_seidel_transform(coeffs, transform)
        worst = 0.0
        for h in field_values:
            lhs = _seidel_wavefront(transformed, x, y, float(h))[mask]
            if transform == "I":
                rhs = _seidel_wavefront(coeffs, x, y, float(h))[mask]
            elif transform == "mirror_x":
                rhs = _seidel_wavefront(coeffs, -x, y, float(h))[mask]
            elif transform == "twin":
                rhs = -_seidel_wavefront(coeffs, -x, y, float(h))[mask]
            else:
                rhs = -_seidel_wavefront(coeffs, x, y, float(h))[mask]
            worst = max(worst, float(np.max(np.abs(lhs - rhs))))
        max_errors[transform] = worst
    return {
        "pass": bool(all(value <= atol for value in max_errors.values())),
        "atol": float(atol),
        "max_errors": max_errors,
    }


def _normalize_probe(probe: torch.Tensor, *, eps: float, enabled: bool) -> torch.Tensor:
    probe = probe.float()
    if not enabled:
        return probe
    norm = torch.linalg.vector_norm(probe)
    if float(norm.detach().cpu()) <= eps:
        return probe
    return probe / norm


def _linspace_indices(dim: int, count: int, margin_fraction: float) -> list[int]:
    if count <= 0:
        return []
    margin = int(round(max(0.0, min(0.45, margin_fraction)) * (dim - 1)))
    lo = margin
    hi = dim - 1 - margin
    if count == 1:
        return [dim // 2]
    return sorted({int(round(v)) for v in np.linspace(lo, hi, count)})


def build_operator_probe_groups(
    dim: int,
    config: OperatorProbeConfig,
    *,
    device: torch.device | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Build deterministic probe images grouped by probe family."""

    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"dim must be a positive even integer, got {dim}")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    groups: dict[str, list[torch.Tensor]] = {}
    eps = float(config.eps)

    if config.full_delta_basis:
        probes: list[torch.Tensor] = []
        for row in range(dim):
            for col in range(dim):
                probe = torch.zeros((dim, dim), dtype=torch.float32, device=device)
                probe[row, col] = 1.0
                probes.append(probe)
        groups["full_delta"] = probes
        return groups

    delta_indices = _linspace_indices(
        dim,
        int(config.delta_grid_size),
        float(config.delta_grid_margin_fraction),
    )
    delta_probes: list[torch.Tensor] = []
    for row in delta_indices:
        for col in delta_indices:
            probe = torch.zeros((dim, dim), dtype=torch.float32, device=device)
            probe[row, col] = 1.0
            delta_probes.append(probe)
    if delta_probes:
        groups["delta_grid"] = delta_probes

    coords = torch.linspace(-1.0, 1.0, dim, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    rr = torch.sqrt(xx * xx + yy * yy)
    radial_count = int(config.radial_basis_count)
    radial_probes: list[torch.Tensor] = []
    if radial_count > 0:
        centers = torch.linspace(0.0, math.sqrt(2.0), radial_count, device=device)
        width = max(float(config.radial_basis_width_fraction), 1.0 / max(dim, 1))
        for center in centers:
            probe = torch.exp(-0.5 * ((rr - center) / width) ** 2)
            radial_probes.append(
                _normalize_probe(
                    probe,
                    eps=eps,
                    enabled=bool(config.input_l2_normalize),
                )
            )
    if radial_probes:
        groups["radial_basis"] = radial_probes

    x_idx = torch.arange(dim, device=device, dtype=torch.float32) / float(dim)
    y_idx = torch.arange(dim, device=device, dtype=torch.float32) / float(dim)
    fy, fx = torch.meshgrid(y_idx, x_idx, indexing="ij")
    fourier_probes: list[torch.Tensor] = []
    for kx, ky in config.fourier_frequencies:
        phase = 2.0 * math.pi * (float(kx) * fx + float(ky) * fy)
        for probe in (torch.cos(phase), torch.sin(phase)):
            fourier_probes.append(
                _normalize_probe(
                    probe,
                    eps=eps,
                    enabled=bool(config.input_l2_normalize),
                )
            )
    if fourier_probes:
        groups["fourier"] = fourier_probes

    rng = np.random.default_rng(int(config.random_seed))
    random_probes: list[torch.Tensor] = []
    for _ in range(int(config.random_count)):
        if config.random_distribution == "normal":
            arr = rng.standard_normal((dim, dim)).astype(np.float32)
        elif config.random_distribution == "uniform":
            arr = rng.uniform(-1.0, 1.0, size=(dim, dim)).astype(np.float32)
        elif config.random_distribution == "rademacher":
            arr = rng.choice([-1.0, 1.0], size=(dim, dim)).astype(np.float32)
        else:
            raise ValueError(f"Unknown random_distribution {config.random_distribution!r}")
        probe = torch.as_tensor(arr, dtype=torch.float32, device=device)
        random_probes.append(
            _normalize_probe(
                probe,
                eps=eps,
                enabled=bool(config.input_l2_normalize),
            )
        )
    if random_probes:
        groups["random"] = random_probes

    for name, probes in list(groups.items()):
        groups[name] = [
            _normalize_probe(
                probe,
                eps=eps,
                enabled=bool(config.input_l2_normalize),
            )
            for probe in probes
        ]
    return groups


class RingOperatorProbeEvaluator:
    """Evaluate relative forward-operator distances with a fixed probe set."""

    def __init__(
        self,
        *,
        dim: int,
        sys_params: Mapping[str, Any] | None,
        probe_config: OperatorProbeConfig,
        device: torch.device | None = None,
    ) -> None:
        self.dim = int(dim)
        self.sys_params = build_sys_params(self.dim, dict(sys_params or {}))
        self.probe_config = probe_config
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.probe_groups = build_operator_probe_groups(
            self.dim,
            probe_config,
            device=self.device,
        )
        self.group_weights = probe_config.resolved_group_weights()
        self._psf_cache: dict[tuple[float, ...], torch.Tensor] = {}

    def probe_group_counts(self) -> dict[str, int]:
        return {name: len(probes) for name, probes in self.probe_groups.items()}

    def _cache_key(self, theta: np.ndarray) -> tuple[float, ...]:
        return tuple(float(f"{value:.12g}") for value in theta.reshape(-1))

    def psfs_for(self, theta: Sequence[float] | np.ndarray | torch.Tensor) -> torch.Tensor:
        coeffs_np = coerce_seidel_vector(theta)
        key = self._cache_key(coeffs_np)
        cached = self._psf_cache.get(key)
        if cached is not None:
            return cached
        coeffs = normalize_seidel_coeffs(coeffs_np, device=self.device)
        psf_backend = os.environ.get("COCOA_OPERATOR_EVAL_PSF_BACKEND", "trainable").strip().lower()
        if int(self.probe_config.patch_size) == 0 and psf_backend != "reference":
            with torch.no_grad():
                psfs = get_trainable_ring_psfs(
                    coeffs,
                    self.dim,
                    self.sys_params,
                    patch_size=0,
                    device=self.device,
                )
        else:
            psfs = get_reference_ring_psfs(
                coeffs,
                self.dim,
                self.sys_params,
                patch_size=int(self.probe_config.patch_size),
                device=self.device,
            )
        self._psf_cache[key] = psfs
        return psfs

    def distance(
        self,
        theta_ref: Sequence[float] | np.ndarray | torch.Tensor,
        theta_candidate: Sequence[float] | np.ndarray | torch.Tensor,
    ) -> float:
        """Return ||A(candidate)-A(ref)|| / ||A(ref)|| over configured probes."""

        if os.environ.get("COCOA_OPERATOR_EVAL_STREAM_DISTANCE", "0").strip() in {"1", "true", "yes"}:
            return self._distance_streamed(theta_ref, theta_candidate)

        psfs_ref = self.psfs_for(theta_ref)
        psfs_candidate = self.psfs_for(theta_candidate)
        numerator = torch.zeros((), dtype=torch.float64, device=self.device)
        denominator = torch.zeros((), dtype=torch.float64, device=self.device)
        for group_name, probes in self.probe_groups.items():
            if not probes:
                continue
            weight = float(self.group_weights.get(group_name, 0.0))
            if weight <= 0.0:
                continue
            group_num = torch.zeros((), dtype=torch.float64, device=self.device)
            group_den = torch.zeros((), dtype=torch.float64, device=self.device)
            for probe in probes:
                ref = blur_ring_with_psfs(
                    probe,
                    psfs_ref,
                    patch_size=int(self.probe_config.patch_size),
                ).double()
                cand = blur_ring_with_psfs(
                    probe,
                    psfs_candidate,
                    patch_size=int(self.probe_config.patch_size),
                ).double()
                diff = cand - ref
                group_num = group_num + torch.sum(diff * diff)
                group_den = group_den + torch.sum(ref * ref)
            scale = weight / float(len(probes))
            numerator = numerator + scale * group_num
            denominator = denominator + scale * group_den
        den_value = float(torch.sqrt(torch.clamp(denominator, min=self.probe_config.eps)).cpu())
        if den_value <= self.probe_config.eps:
            return float("inf")
        num_value = float(torch.sqrt(torch.clamp(numerator, min=0.0)).cpu())
        return float(num_value / den_value)

    def _distance_streamed(
        self,
        theta_ref: Sequence[float] | np.ndarray | torch.Tensor,
        theta_candidate: Sequence[float] | np.ndarray | torch.Tensor,
    ) -> float:
        ref_outputs: dict[str, list[torch.Tensor]] = {}
        denominator = torch.zeros((), dtype=torch.float64, device=self.device)

        psfs_ref = self.psfs_for(theta_ref)
        for group_name, probes in self.probe_groups.items():
            weight = float(self.group_weights.get(group_name, 0.0))
            if not probes or weight <= 0.0:
                continue
            group_den = torch.zeros((), dtype=torch.float64, device=self.device)
            refs: list[torch.Tensor] = []
            for probe in probes:
                ref = blur_ring_with_psfs(
                    probe,
                    psfs_ref,
                    patch_size=int(self.probe_config.patch_size),
                ).double()
                group_den = group_den + torch.sum(ref * ref)
                refs.append(ref.detach().cpu())
                del ref
            denominator = denominator + (weight / float(len(probes))) * group_den
            ref_outputs[group_name] = refs
        del psfs_ref
        self._psf_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        psfs_candidate = self.psfs_for(theta_candidate)
        numerator = torch.zeros((), dtype=torch.float64, device=self.device)
        for group_name, probes in self.probe_groups.items():
            weight = float(self.group_weights.get(group_name, 0.0))
            if not probes or weight <= 0.0:
                continue
            refs = ref_outputs.get(group_name, [])
            group_num = torch.zeros((), dtype=torch.float64, device=self.device)
            for probe, ref_cpu in zip(probes, refs):
                ref = ref_cpu.to(device=self.device)
                cand = blur_ring_with_psfs(
                    probe,
                    psfs_candidate,
                    patch_size=int(self.probe_config.patch_size),
                ).double()
                diff = cand - ref
                group_num = group_num + torch.sum(diff * diff)
                del ref, cand, diff
            numerator = numerator + (weight / float(len(probes))) * group_num
        del psfs_candidate
        self._psf_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        den_value = float(torch.sqrt(torch.clamp(denominator, min=self.probe_config.eps)).cpu())
        if den_value <= self.probe_config.eps:
            return float("inf")
        num_value = float(torch.sqrt(torch.clamp(numerator, min=0.0)).cpu())
        return float(num_value / den_value)


def _transform_errors(
    evaluator: RingOperatorProbeEvaluator,
    theta_gt: np.ndarray,
    theta_hat: np.ndarray,
    transforms: Iterable[str],
    *,
    fixed_indices: Sequence[int],
    config: OperatorProbeConfig,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for transform in transforms:
        aligned = apply_seidel_transform(
            theta_hat,
            transform,
            fixed_indices=fixed_indices,
        )
        coeff_abs, coeff_rel = coefficient_residuals(theta_gt, aligned, eps=config.eps)
        wavefront = relative_wavefront_residual(
            theta_gt,
            aligned,
            field_samples=config.wavefront_field_samples,
            pupil_samples=config.wavefront_pupil_samples,
            eps=config.eps,
        )
        out[transform] = {
            "operator": evaluator.distance(theta_gt, aligned),
            "coeff_abs": coeff_abs,
            "coeff_rel": coeff_rel,
            "wavefront": wavefront,
            "aligned": aligned,
        }
    return out


def _trace_coefficient_residuals(
    theta_ref: np.ndarray,
    theta_candidate: np.ndarray,
    *,
    eps: float,
) -> tuple[float, float]:
    absolute = float(np.linalg.norm(theta_candidate - theta_ref))
    relative = float(absolute / max(float(np.linalg.norm(theta_ref)), eps))
    return absolute, relative


def _trace_transform_errors(
    evaluator: RingOperatorProbeEvaluator,
    theta_trace_gt: np.ndarray,
    theta_trace_hat: np.ndarray,
    transforms: Iterable[str],
    *,
    model_dim: int,
    config: OperatorProbeConfig,
) -> dict[str, dict[str, Any]]:
    gt_backend = np.asarray(
        expand_trace_seidel(theta_trace_gt, model_dim=model_dim),
        dtype=np.float64,
    )
    out: dict[str, dict[str, Any]] = {}
    for transform in transforms:
        aligned_trace = apply_trace_transform(
            theta_trace_hat,
            transform,
            model_dim=model_dim,
        )
        aligned_backend = np.asarray(
            expand_trace_seidel(aligned_trace, model_dim=model_dim),
            dtype=np.float64,
        )
        coeff_abs, coeff_rel = _trace_coefficient_residuals(
            theta_trace_gt,
            aligned_trace,
            eps=config.eps,
        )
        wavefront = relative_wavefront_residual(
            gt_backend,
            aligned_backend,
            field_samples=config.wavefront_field_samples,
            pupil_samples=config.wavefront_pupil_samples,
            eps=config.eps,
        )
        out[transform] = {
            "operator": evaluator.distance(gt_backend, aligned_backend),
            "coeff_abs": coeff_abs,
            "coeff_rel": coeff_rel,
            "wavefront": wavefront,
            "aligned": aligned_trace,
            "aligned_backend": aligned_backend,
        }
    return out


def _choose_best_transform(
    errors: Mapping[str, Mapping[str, Any]],
    *,
    tie_operator_tol: float,
) -> tuple[str, Mapping[str, Any]]:
    min_operator = min(float(item["operator"]) for item in errors.values())
    tied = [
        name
        for name, item in errors.items()
        if float(item["operator"]) <= min_operator + tie_operator_tol * max(1.0, abs(min_operator))
    ]
    order_rank = {name: idx for idx, name in enumerate(OPERATOR_TRANSFORM_ORDER)}
    tied.sort(
        key=lambda name: (
            float(errors[name]["wavefront"]),
            float(errors[name]["coeff_abs"]),
            order_rank.get(name, 10_000),
        )
    )
    best = tied[0]
    return best, errors[best]


def _sign_vector(theta: np.ndarray, *, tol: float = 1e-8) -> list[int]:
    signs: list[int] = []
    for value in np.asarray(theta, dtype=np.float64).reshape(-1):
        if abs(float(value)) <= tol:
            signs.append(0)
        elif float(value) > 0.0:
            signs.append(1)
        else:
            signs.append(-1)
    return signs


def _sign_agreement_report(
    theta_gt: np.ndarray,
    theta_candidate: np.ndarray,
    *,
    coeff_names: Sequence[str],
    fixed_indices: Sequence[int],
    tol: float = 1e-8,
) -> dict[str, Any]:
    gt = np.asarray(theta_gt, dtype=np.float64).reshape(-1)
    cand = np.asarray(theta_candidate, dtype=np.float64).reshape(-1)
    gt_sign = _sign_vector(gt, tol=tol)
    cand_sign = _sign_vector(cand, tol=tol)
    fixed = set(int(idx) for idx in fixed_indices)
    matches: list[bool | None] = []
    mismatch_coeffs: list[str] = []
    valid = 0
    matched = 0
    for idx, (sgt, scand) in enumerate(zip(gt_sign, cand_sign)):
        name = str(coeff_names[idx]) if idx < len(coeff_names) else f"theta{idx}"
        if idx in fixed or sgt == 0:
            matches.append(None)
            continue
        valid += 1
        is_match = bool(scand == sgt)
        matches.append(is_match)
        if is_match:
            matched += 1
        else:
            mismatch_coeffs.append(name)
    return {
        "sign_gt": gt_sign,
        "sign_hat": cand_sign,
        "sign_match": matches,
        "sign_match_rate": float(matched / valid) if valid else float("nan"),
        "sign_mismatch_coeffs": mismatch_coeffs,
        "sign_valid_coeff_count": int(valid),
    }


def _canonical_summary(
    theta_gt: np.ndarray,
    theta_hat: np.ndarray,
    *,
    raw_errors: Mapping[str, Any],
    physical_transform: str,
    physical: Mapping[str, Any],
    gauge_transform: str,
    gauge: Mapping[str, Any],
    fixed_indices: Sequence[int],
    coeff_names: Sequence[str],
    config: OperatorProbeConfig,
) -> dict[str, Any]:
    physical_theta = np.asarray(physical["aligned"], dtype=np.float64).reshape(-1)
    gauge_theta = np.asarray(gauge["aligned"], dtype=np.float64).reshape(-1)
    raw_report = _sign_agreement_report(
        theta_gt,
        theta_hat,
        coeff_names=coeff_names,
        fixed_indices=fixed_indices,
    )
    physical_report = _sign_agreement_report(
        theta_gt,
        physical_theta,
        coeff_names=coeff_names,
        fixed_indices=fixed_indices,
    )
    gauge_report = _sign_agreement_report(
        theta_gt,
        gauge_theta,
        coeff_names=coeff_names,
        fixed_indices=fixed_indices,
    )

    gt_rms = field_weighted_wavefront_rms(
        theta_gt,
        field_samples=config.wavefront_field_samples,
        pupil_samples=config.wavefront_pupil_samples,
    )
    raw_rms = field_weighted_wavefront_rms(
        theta_hat,
        field_samples=config.wavefront_field_samples,
        pupil_samples=config.wavefront_pupil_samples,
    )
    physical_rms = field_weighted_wavefront_rms(
        physical_theta,
        field_samples=config.wavefront_field_samples,
        pupil_samples=config.wavefront_pupil_samples,
    )
    gauge_rms = field_weighted_wavefront_rms(
        gauge_theta,
        field_samples=config.wavefront_field_samples,
        pupil_samples=config.wavefront_pupil_samples,
    )
    denom = max(float(gt_rms), float(config.eps))
    return {
        "canonical_sign_source": "gauge",
        "canonical_gauge_v1_note": (
            "Gauge v1 uses hard-coded discrete transforms only; continuous "
            "recenter/refocus is not enabled in the current forward model."
        ),
        "canonical_gauge_transform_aliases": {
            key: list(value) for key, value in GAUGE_TRANSFORM_ALIASES.items()
        },
        "theta_hat_canonical_physical": [float(value) for value in physical_theta],
        "canonical_transform_physical": physical_transform,
        "canonical_transform_physical_aliases": list(
            GAUGE_TRANSFORM_ALIASES.get(physical_transform, (physical_transform,))
        ),
        "canonical_operator_error_physical": float(physical["operator"]),
        "canonical_wavefront_error_physical": float(physical["wavefront"]),
        "canonical_coeff_absolute_error_physical": float(physical["coeff_abs"]),
        "canonical_coeff_relative_error_physical": float(physical["coeff_rel"]),
        "theta_hat_canonical_gauge": [float(value) for value in gauge_theta],
        "canonical_transform_gauge": gauge_transform,
        "canonical_transform_gauge_aliases": list(
            GAUGE_TRANSFORM_ALIASES.get(gauge_transform, (gauge_transform,))
        ),
        "canonical_operator_error_gauge": float(gauge["operator"]),
        "canonical_wavefront_error_gauge": float(gauge["wavefront"]),
        "canonical_coeff_absolute_error_gauge": float(gauge["coeff_abs"]),
        "canonical_coeff_relative_error_gauge": float(gauge["coeff_rel"]),
        "raw_wavefront_rms_eval": float(raw_rms),
        "canonical_wavefront_rms_physical": float(physical_rms),
        "canonical_wavefront_rms_gauge": float(gauge_rms),
        "raw_recovered_over_gt_wavefront_rms_eval": float(raw_rms / denom),
        "canonical_recovered_over_gt_wavefront_rms_physical": float(physical_rms / denom),
        "canonical_recovered_over_gt_wavefront_rms_gauge": float(gauge_rms / denom),
        "canonical_sign_match_raw": raw_report["sign_match"],
        "canonical_sign_match_rate_raw": raw_report["sign_match_rate"],
        "canonical_sign_mismatch_coeffs_raw": raw_report["sign_mismatch_coeffs"],
        "canonical_sign_valid_coeff_count_raw": raw_report["sign_valid_coeff_count"],
        "canonical_sign_gt": raw_report["sign_gt"],
        "canonical_sign_hat_raw": raw_report["sign_hat"],
        "canonical_sign_match_physical": physical_report["sign_match"],
        "canonical_sign_match_rate_physical": physical_report["sign_match_rate"],
        "canonical_sign_mismatch_coeffs_physical": physical_report["sign_mismatch_coeffs"],
        "canonical_sign_valid_coeff_count_physical": physical_report["sign_valid_coeff_count"],
        "canonical_sign_hat_physical": physical_report["sign_hat"],
        "canonical_sign_match_gauge": gauge_report["sign_match"],
        "canonical_sign_match_rate_gauge": gauge_report["sign_match_rate"],
        "canonical_sign_mismatch_coeffs_gauge": gauge_report["sign_mismatch_coeffs"],
        "canonical_sign_valid_coeff_count_gauge": gauge_report["sign_valid_coeff_count"],
        "canonical_sign_hat_gauge": gauge_report["sign_hat"],
    }


def _ambiguity_gap(errors: Mapping[str, Mapping[str, Any]], calibrated_error: float) -> float:
    if len(errors) <= 1:
        return 0.0
    best = min(float(item["operator"]) for item in errors.values())
    return float(calibrated_error - best)


def _diagnostic_point_list(dim: int, config: OperatorProbeConfig) -> list[tuple[float, float]]:
    half = dim / 2.0
    points = []
    for x_frac, y_frac in config.diagnostic_psf_points:
        x = max(-0.95, min(0.95, float(x_frac))) * half
        y = max(-0.95, min(0.95, float(y_frac))) * half
        points.append((x, y))
    return points


def _flux_normalized_psf_stack(
    theta: np.ndarray,
    *,
    dim: int,
    sys_params: Mapping[str, Any],
    config: OperatorProbeConfig,
    device: torch.device,
) -> torch.Tensor:
    coeffs = normalize_seidel_coeffs(theta, device=device)
    psfs = compute_rdm_psfs(
        coeffs,
        _diagnostic_point_list(dim, config),
        dim=dim,
        sys_params=dict(sys_params),
        polar=False,
        stack=False,
        buffer=0,
        shift=True,
        verbose=False,
        device=device,
    )
    stack = torch.stack([psf.float() for psf in psfs], dim=0)
    flux = stack.sum(dim=(-2, -1), keepdim=True)
    return stack / torch.clamp(flux, min=float(config.eps))


def _psf_stack_error(
    theta_gt: np.ndarray,
    theta_candidate: np.ndarray,
    *,
    dim: int,
    sys_params: Mapping[str, Any],
    config: OperatorProbeConfig,
    device: torch.device,
) -> float:
    gt = _flux_normalized_psf_stack(
        theta_gt,
        dim=dim,
        sys_params=sys_params,
        config=config,
        device=device,
    ).double()
    cand = _flux_normalized_psf_stack(
        theta_candidate,
        dim=dim,
        sys_params=sys_params,
        config=config,
        device=device,
    ).double()
    num = torch.linalg.vector_norm(cand - gt)
    den = torch.linalg.vector_norm(gt)
    return float((num / torch.clamp(den, min=float(config.eps))).cpu())


def _otf_complex_error(
    theta_gt: np.ndarray,
    theta_candidate: np.ndarray,
    *,
    dim: int,
    sys_params: Mapping[str, Any],
    config: OperatorProbeConfig,
    device: torch.device,
) -> float:
    gt_psf = _flux_normalized_psf_stack(
        theta_gt,
        dim=dim,
        sys_params=sys_params,
        config=config,
        device=device,
    ).double()
    cand_psf = _flux_normalized_psf_stack(
        theta_candidate,
        dim=dim,
        sys_params=sys_params,
        config=config,
        device=device,
    ).double()
    gt_otf = torch.fft.fft2(gt_psf)
    cand_otf = torch.fft.fft2(cand_psf)
    gt_dc = gt_otf[..., :1, :1]
    cand_dc = cand_otf[..., :1, :1]
    gt_otf = gt_otf / torch.where(
        torch.abs(gt_dc) > config.eps,
        gt_dc,
        torch.ones_like(gt_dc),
    )
    cand_otf = cand_otf / torch.where(
        torch.abs(cand_dc) > config.eps,
        cand_dc,
        torch.ones_like(cand_dc),
    )
    num = torch.linalg.vector_norm(cand_otf - gt_otf)
    den = torch.linalg.vector_norm(gt_otf)
    return float((num / torch.clamp(den.real, min=float(config.eps))).cpu())


def _min_diagnostic_error(
    metric_fn: Any,
    theta_gt: np.ndarray,
    theta_hat: np.ndarray,
    transforms: Sequence[str],
    *,
    fixed_indices: Sequence[int],
) -> float:
    values = []
    for transform in transforms:
        aligned = apply_seidel_transform(
            theta_hat,
            transform,
            fixed_indices=fixed_indices,
        )
        values.append(float(metric_fn(theta_gt, aligned)))
    return float(min(values)) if values else float("nan")


def _coerce_eval_device(device: torch.device | str | None) -> torch.device | None:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def evaluate_seidel_recovery(
    theta_gt: Sequence[float] | np.ndarray | torch.Tensor,
    theta_hat: Sequence[float] | np.ndarray | torch.Tensor,
    dim: int,
    sys_params: Mapping[str, Any] | None,
    fixed_indices: Sequence[int] | None,
    probe_config: OperatorProbeConfig | None,
    dataset_twin_invariance_pass: bool,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Evaluate a recovered Seidel vector against the exact ring operator.

    Parameters
    ----------
    theta_gt, theta_hat
        Ground-truth and recovered Seidel coefficients. Either full 6-vectors
        or compact active vectors compatible with ``fixed_indices``.
    dim
        Image side length used by the ring-convolution operator.
    sys_params
        Optical system parameters passed to the ring PSF generator.
    fixed_indices
        Fixed Seidel indices for 5D/4D sweeps. These are zeroed after every
        transform.
    probe_config
        Deterministic probe configuration. Defaults to
        :class:`OperatorProbeConfig`.
    dataset_twin_invariance_pass
        Dataset-level random-theta twin invariance gate. The sample-level twin
        gate must also pass for both GT and theta_hat before ``twin`` enters
        physical equivalence.
    """

    config = probe_config or OperatorProbeConfig()
    fixed = normalize_fixed_indices(fixed_indices)
    gt = coerce_seidel_vector(theta_gt, fixed_indices=fixed)
    hat = coerce_seidel_vector(theta_hat, fixed_indices=fixed)
    evaluator = RingOperatorProbeEvaluator(
        dim=int(dim),
        sys_params=sys_params,
        probe_config=config,
        device=_coerce_eval_device(device),
    )

    twin_gt = apply_seidel_transform(gt, "twin", fixed_indices=fixed)
    twin_hat = apply_seidel_transform(hat, "twin", fixed_indices=fixed)
    twin_invariance_error_gt = evaluator.distance(gt, twin_gt)
    twin_invariance_error_hat = evaluator.distance(hat, twin_hat)
    twin_invariance_pass_gt = bool(twin_invariance_error_gt <= config.twin_invariance_tol)
    twin_invariance_pass_hat = bool(twin_invariance_error_hat <= config.twin_invariance_tol)
    twin_allowed_for_sample = bool(
        dataset_twin_invariance_pass
        and twin_invariance_pass_gt
        and twin_invariance_pass_hat
    )

    calibrated_errors = _transform_errors(
        evaluator,
        gt,
        hat,
        ("I",),
        fixed_indices=fixed,
        config=config,
    )
    physical_transforms = ("I", "twin") if twin_allowed_for_sample else ("I",)
    physical_errors = _transform_errors(
        evaluator,
        gt,
        hat,
        physical_transforms,
        fixed_indices=fixed,
        config=config,
    )
    coordinate_errors = _transform_errors(
        evaluator,
        gt,
        hat,
        COORDINATE_DIAGNOSTIC_TRANSFORMS,
        fixed_indices=fixed,
        config=config,
    )

    calibrated = calibrated_errors["I"]
    best_physical_transform, best_physical = _choose_best_transform(
        physical_errors,
        tie_operator_tol=config.tie_operator_tol,
    )
    best_coord_transform, best_coord = _choose_best_transform(
        coordinate_errors,
        tie_operator_tol=config.tie_operator_tol,
    )

    raw_coeff_abs, raw_coeff_rel = coefficient_residuals(gt, hat, eps=config.eps)

    sys_params_resolved = evaluator.sys_params
    device = evaluator.device

    def psf_metric(a: np.ndarray, b: np.ndarray) -> float:
        return _psf_stack_error(
            a,
            b,
            dim=int(dim),
            sys_params=sys_params_resolved,
            config=config,
            device=device,
        )

    def otf_metric(a: np.ndarray, b: np.ndarray) -> float:
        return _otf_complex_error(
            a,
            b,
            dim=int(dim),
            sys_params=sys_params_resolved,
            config=config,
            device=device,
        )

    psf_error_calibrated = psf_metric(gt, hat)
    psf_error_phys_equiv = _min_diagnostic_error(
        psf_metric,
        gt,
        hat,
        physical_transforms,
        fixed_indices=fixed,
    )
    otf_complex_error_calibrated = otf_metric(gt, hat)
    otf_complex_error_phys_equiv = _min_diagnostic_error(
        otf_metric,
        gt,
        hat,
        physical_transforms,
        fixed_indices=fixed,
    )
    canonical = _canonical_summary(
        gt,
        hat,
        raw_errors=calibrated,
        physical_transform=best_physical_transform,
        physical=best_physical,
        gauge_transform=best_coord_transform,
        gauge=best_coord,
        fixed_indices=fixed,
        coeff_names=SEIDEL_COEFF_NAMES,
        config=config,
    )

    output: dict[str, Any] = {
        "operator_error_calibrated": float(calibrated["operator"]),
        "operator_error_phys_equiv": float(best_physical["operator"]),
        "operator_error_coord_diagnostic": float(best_coord["operator"]),
        "best_physical_transform": best_physical_transform,
        "best_coordinate_diagnostic_transform": best_coord_transform,
        "physical_ambiguity_gap": _ambiguity_gap(
            physical_errors,
            float(calibrated["operator"]),
        ),
        "coordinate_ambiguity_gap": _ambiguity_gap(
            coordinate_errors,
            float(calibrated["operator"]),
        ),
        "twin_invariance_error_gt": float(twin_invariance_error_gt),
        "twin_invariance_error_hat": float(twin_invariance_error_hat),
        "twin_invariance_pass_gt": twin_invariance_pass_gt,
        "twin_invariance_pass_hat": twin_invariance_pass_hat,
        "twin_allowed_for_sample": twin_allowed_for_sample,
        "raw_coeff_absolute_error": float(raw_coeff_abs),
        "raw_coeff_relative_error": float(raw_coeff_rel),
        "wavefront_error_calibrated": float(calibrated["wavefront"]),
        "aligned_coeff_absolute_error_physical": float(best_physical["coeff_abs"]),
        "aligned_coeff_relative_error_physical": float(best_physical["coeff_rel"]),
        "aligned_wavefront_error_physical": float(best_physical["wavefront"]),
        "aligned_coeff_absolute_error_coord_diagnostic": float(best_coord["coeff_abs"]),
        "aligned_coeff_relative_error_coord_diagnostic": float(best_coord["coeff_rel"]),
        "aligned_wavefront_error_coord_diagnostic": float(best_coord["wavefront"]),
        "aligned_seidel_physical": [
            float(value) for value in np.asarray(best_physical["aligned"]).reshape(-1)
        ],
        "aligned_seidel_coord_diagnostic": [
            float(value) for value in np.asarray(best_coord["aligned"]).reshape(-1)
        ],
        "psf_error_calibrated": float(psf_error_calibrated),
        "psf_error_phys_equiv": float(psf_error_phys_equiv),
        "otf_complex_error_calibrated": float(otf_complex_error_calibrated),
        "otf_complex_error_phys_equiv": float(otf_complex_error_phys_equiv),
        "probe_config_hash": config.stable_hash(),
        "probe_config": config.to_dict(),
        "probe_group_weights": config.resolved_group_weights(),
        "probe_group_counts": evaluator.probe_group_counts(),
        "physical_transform_set": list(physical_transforms),
        "gauge_transform_set": list(GAUGE_TRANSFORM_SET),
        "coordinate_diagnostic_transform_set": list(COORDINATE_DIAGNOSTIC_TRANSFORMS),
        "fixed_seidel_indices": list(fixed),
    }
    output.update(canonical)
    return output


def evaluate_trace_seidel_recovery(
    theta_trace_gt: Sequence[float] | np.ndarray | torch.Tensor,
    theta_trace_hat: Sequence[float] | np.ndarray | torch.Tensor,
    dim: int,
    sys_params: Mapping[str, Any] | None,
    *,
    model_dim: int | str | None = None,
    probe_config: OperatorProbeConfig | None = None,
    dataset_twin_invariance_pass: bool = False,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Evaluate public trace-separated Seidel recovery with the exact RDM operator.

    This is the optimizer/evaluator-facing reduced API. Public trace-space
    transforms are applied first, and each candidate is expanded to backend 6D
    before the unchanged RDM ring-convolution operator is evaluated.
    """
    config = probe_config or OperatorProbeConfig()
    gt_trace = coerce_trace_vector(theta_trace_gt, model_dim=model_dim)
    dim_model = _normalize_trace_dim(model_dim, int(gt_trace.size))
    hat_trace = coerce_trace_vector(theta_trace_hat, model_dim=dim_model)
    gt_backend = np.asarray(expand_trace_seidel(gt_trace, model_dim=dim_model), dtype=np.float64)
    hat_backend = np.asarray(expand_trace_seidel(hat_trace, model_dim=dim_model), dtype=np.float64)

    evaluator = RingOperatorProbeEvaluator(
        dim=int(dim),
        sys_params=sys_params,
        probe_config=config,
        device=_coerce_eval_device(device),
    )

    twin_gt_trace = apply_trace_transform(gt_trace, "twin", model_dim=dim_model)
    twin_hat_trace = apply_trace_transform(hat_trace, "twin", model_dim=dim_model)
    twin_gt_backend = np.asarray(
        expand_trace_seidel(twin_gt_trace, model_dim=dim_model),
        dtype=np.float64,
    )
    twin_hat_backend = np.asarray(
        expand_trace_seidel(twin_hat_trace, model_dim=dim_model),
        dtype=np.float64,
    )
    twin_invariance_error_gt = evaluator.distance(gt_backend, twin_gt_backend)
    twin_invariance_error_hat = evaluator.distance(hat_backend, twin_hat_backend)
    twin_invariance_pass_gt = bool(twin_invariance_error_gt <= config.twin_invariance_tol)
    twin_invariance_pass_hat = bool(twin_invariance_error_hat <= config.twin_invariance_tol)
    twin_allowed_for_sample = bool(
        dataset_twin_invariance_pass
        and twin_invariance_pass_gt
        and twin_invariance_pass_hat
    )

    calibrated_errors = _trace_transform_errors(
        evaluator,
        gt_trace,
        hat_trace,
        ("I",),
        model_dim=dim_model,
        config=config,
    )
    physical_transforms = ("I", "twin") if twin_allowed_for_sample else ("I",)
    physical_errors = _trace_transform_errors(
        evaluator,
        gt_trace,
        hat_trace,
        physical_transforms,
        model_dim=dim_model,
        config=config,
    )
    coordinate_errors = _trace_transform_errors(
        evaluator,
        gt_trace,
        hat_trace,
        COORDINATE_DIAGNOSTIC_TRANSFORMS,
        model_dim=dim_model,
        config=config,
    )

    calibrated = calibrated_errors["I"]
    best_physical_transform, best_physical = _choose_best_transform(
        physical_errors,
        tie_operator_tol=config.tie_operator_tol,
    )
    best_coord_transform, best_coord = _choose_best_transform(
        coordinate_errors,
        tie_operator_tol=config.tie_operator_tol,
    )

    raw_coeff_abs, raw_coeff_rel = _trace_coefficient_residuals(
        gt_trace,
        hat_trace,
        eps=config.eps,
    )

    sys_params_resolved = evaluator.sys_params
    device = evaluator.device

    def psf_metric(a: np.ndarray, b: np.ndarray) -> float:
        return _psf_stack_error(
            a,
            b,
            dim=int(dim),
            sys_params=sys_params_resolved,
            config=config,
            device=device,
        )

    def otf_metric(a: np.ndarray, b: np.ndarray) -> float:
        return _otf_complex_error(
            a,
            b,
            dim=int(dim),
            sys_params=sys_params_resolved,
            config=config,
            device=device,
        )

    psf_error_strict = psf_metric(gt_backend, hat_backend)
    otf_complex_error_strict = otf_metric(gt_backend, hat_backend)
    psf_error_phys_equiv = min(
        psf_metric(
            gt_backend,
            np.asarray(
                expand_trace_seidel(
                    apply_trace_transform(hat_trace, transform, model_dim=dim_model),
                    model_dim=dim_model,
                ),
                dtype=np.float64,
            ),
        )
        for transform in physical_transforms
    )
    otf_complex_error_phys_equiv = min(
        otf_metric(
            gt_backend,
            np.asarray(
                expand_trace_seidel(
                    apply_trace_transform(hat_trace, transform, model_dim=dim_model),
                    model_dim=dim_model,
                ),
                dtype=np.float64,
            ),
        )
        for transform in physical_transforms
    )
    fixed_trace_indices = [5] if dim_model == 5 else [4, 5]
    physical_backend_coeff_abs, physical_backend_coeff_rel = coefficient_residuals(
        gt_backend,
        np.asarray(best_physical["aligned_backend"], dtype=np.float64),
        eps=config.eps,
    )
    gauge_backend_coeff_abs, gauge_backend_coeff_rel = coefficient_residuals(
        gt_backend,
        np.asarray(best_coord["aligned_backend"], dtype=np.float64),
        eps=config.eps,
    )
    physical_backend = dict(best_physical)
    physical_backend.update(
        {
            "aligned": np.asarray(best_physical["aligned_backend"], dtype=np.float64),
            "coeff_abs": physical_backend_coeff_abs,
            "coeff_rel": physical_backend_coeff_rel,
        }
    )
    gauge_backend = dict(best_coord)
    gauge_backend.update(
        {
            "aligned": np.asarray(best_coord["aligned_backend"], dtype=np.float64),
            "coeff_abs": gauge_backend_coeff_abs,
            "coeff_rel": gauge_backend_coeff_rel,
        }
    )
    canonical = _canonical_summary(
        gt_backend,
        hat_backend,
        raw_errors=calibrated,
        physical_transform=best_physical_transform,
        physical=physical_backend,
        gauge_transform=best_coord_transform,
        gauge=gauge_backend,
        fixed_indices=fixed_trace_indices,
        coeff_names=SEIDEL_COEFF_NAMES,
        config=config,
    )

    symmetry_status = {
        "eta_1": "not_applicable",
        "eta_2": "not_applicable",
        "parity_residual": "not_applicable",
        "field_scaling_residual": "not_applicable",
        "symmetry_diagnostic_source": "not_applicable_no_local_proxy",
    }

    output: dict[str, Any] = {
        "theta_convention": f"trace{dim_model}",
        "no_defocus": True,
        "no_w311_no_defocus": bool(dim_model != 5),
        "theta_trace_gt": [float(value) for value in gt_trace],
        "theta_trace_hat": [float(value) for value in hat_trace],
        "theta_backend6_gt": [float(value) for value in gt_backend],
        "theta_backend6_hat": [float(value) for value in hat_backend],
        "operator_error_strict": float(calibrated["operator"]),
        "operator_error_phys_equiv": float(best_physical["operator"]),
        "operator_error_coord_diagnostic": float(best_coord["operator"]),
        "operator_error_calibrated": float(calibrated["operator"]),
        "best_physical_transform": best_physical_transform,
        "best_coordinate_diagnostic_transform": best_coord_transform,
        "symmetry_ambiguity_gap": _ambiguity_gap(
            physical_errors,
            float(calibrated["operator"]),
        ),
        "physical_ambiguity_gap": _ambiguity_gap(
            physical_errors,
            float(calibrated["operator"]),
        ),
        "coordinate_ambiguity_gap": _ambiguity_gap(
            coordinate_errors,
            float(calibrated["operator"]),
        ),
        "twin_invariance_error_gt": float(twin_invariance_error_gt),
        "twin_invariance_error_hat": float(twin_invariance_error_hat),
        "twin_invariance_pass_gt": twin_invariance_pass_gt,
        "twin_invariance_pass_hat": twin_invariance_pass_hat,
        "twin_allowed_for_sample": twin_allowed_for_sample,
        "raw_coeff_absolute_error": float(raw_coeff_abs),
        "raw_coeff_relative_error": float(raw_coeff_rel),
        "wavefront_error_strict": float(calibrated["wavefront"]),
        "wavefront_error_calibrated": float(calibrated["wavefront"]),
        "aligned_coeff_absolute_error_physical": float(best_physical["coeff_abs"]),
        "aligned_coeff_relative_error_physical": float(best_physical["coeff_rel"]),
        "aligned_wavefront_error_physical": float(best_physical["wavefront"]),
        "aligned_coeff_absolute_error_coord_diagnostic": float(best_coord["coeff_abs"]),
        "aligned_coeff_relative_error_coord_diagnostic": float(best_coord["coeff_rel"]),
        "aligned_wavefront_error_coord_diagnostic": float(best_coord["wavefront"]),
        "aligned_trace_physical": [
            float(value) for value in np.asarray(best_physical["aligned"]).reshape(-1)
        ],
        "aligned_trace_coord_diagnostic": [
            float(value) for value in np.asarray(best_coord["aligned"]).reshape(-1)
        ],
        "aligned_backend6_physical": [
            float(value) for value in np.asarray(best_physical["aligned_backend"]).reshape(-1)
        ],
        "aligned_backend6_coord_diagnostic": [
            float(value) for value in np.asarray(best_coord["aligned_backend"]).reshape(-1)
        ],
        "psf_error_strict": float(psf_error_strict),
        "psf_error_phys_equiv": float(psf_error_phys_equiv),
        "psf_error_calibrated": float(psf_error_strict),
        "otf_complex_error_strict": float(otf_complex_error_strict),
        "otf_complex_error_phys_equiv": float(otf_complex_error_phys_equiv),
        "otf_complex_error_calibrated": float(otf_complex_error_strict),
        "probe_config_hash": config.stable_hash(),
        "probe_config": config.to_dict(),
        "probe_group_weights": config.resolved_group_weights(),
        "probe_group_counts": evaluator.probe_group_counts(),
        "calibrated_transform_set": ["I"],
        "physical_transform_set": list(physical_transforms),
        "gauge_transform_set": list(GAUGE_TRANSFORM_SET),
        "coordinate_diagnostic_transform_set": list(COORDINATE_DIAGNOSTIC_TRANSFORMS),
        "fixed_seidel_indices": fixed_trace_indices,
    }
    output.update(canonical)
    output.update(symmetry_status)
    return output


def check_dataset_twin_invariance(
    *,
    dim: int,
    sys_params: Mapping[str, Any] | None,
    fixed_indices: Sequence[int] | None,
    probe_config: OperatorProbeConfig | None = None,
    num_samples: int = 8,
    random_seed: int = 314159,
    theta_scale: float = 0.15,
) -> dict[str, Any]:
    """Run the dataset-level random-theta twin invariance gate."""

    config = probe_config or OperatorProbeConfig()
    fixed = normalize_fixed_indices(fixed_indices)
    evaluator = RingOperatorProbeEvaluator(
        dim=int(dim),
        sys_params=sys_params,
        probe_config=config,
    )
    rng = np.random.default_rng(int(random_seed))
    errors: list[float] = []
    thetas: list[list[float]] = []
    for _ in range(int(num_samples)):
        theta = rng.normal(0.0, float(theta_scale), size=NUM_SEIDEL)
        for idx in fixed:
            theta[idx] = 0.0
        twin = apply_seidel_transform(theta, "twin", fixed_indices=fixed)
        errors.append(float(evaluator.distance(theta, twin)))
        thetas.append([float(value) for value in theta])
    max_error = float(max(errors)) if errors else float("nan")
    return {
        "dataset_twin_invariance_pass": bool(
            errors and max_error <= config.twin_invariance_tol
        ),
        "dataset_twin_invariance_tol": float(config.twin_invariance_tol),
        "dataset_twin_invariance_errors": errors,
        "dataset_twin_invariance_max_error": max_error,
        "dataset_twin_invariance_mean_error": float(np.mean(errors)) if errors else float("nan"),
        "dataset_twin_invariance_num_samples": int(num_samples),
        "dataset_twin_invariance_seed": int(random_seed),
        "dataset_twin_invariance_theta_scale": float(theta_scale),
        "sampled_thetas": thetas,
        "probe_config_hash": config.stable_hash(),
        "probe_config": config.to_dict(),
        "probe_group_weights": config.resolved_group_weights(),
    }


def check_trace_dataset_twin_invariance(
    *,
    dim: int,
    sys_params: Mapping[str, Any] | None,
    model_dim: int | str = 4,
    probe_config: OperatorProbeConfig | None = None,
    num_samples: int = 8,
    random_seed: int = 314159,
    theta_scale: float = 0.15,
) -> dict[str, Any]:
    """Run a dataset-level twin gate using public trace-space transforms."""
    config = probe_config or OperatorProbeConfig()
    dim_model = _normalize_trace_dim(model_dim)
    evaluator = RingOperatorProbeEvaluator(
        dim=int(dim),
        sys_params=sys_params,
        probe_config=config,
    )
    rng = np.random.default_rng(int(random_seed))
    errors: list[float] = []
    thetas: list[list[float]] = []
    for _ in range(int(num_samples)):
        theta_trace = rng.normal(0.0, float(theta_scale), size=dim_model)
        theta_backend = np.asarray(
            expand_trace_seidel(theta_trace, model_dim=dim_model),
            dtype=np.float64,
        )
        twin_trace = apply_trace_transform(theta_trace, "twin", model_dim=dim_model)
        twin_backend = np.asarray(
            expand_trace_seidel(twin_trace, model_dim=dim_model),
            dtype=np.float64,
        )
        errors.append(float(evaluator.distance(theta_backend, twin_backend)))
        thetas.append([float(value) for value in theta_trace])
    max_error = float(max(errors)) if errors else float("nan")
    return {
        "dataset_twin_invariance_pass": bool(
            errors and max_error <= config.twin_invariance_tol
        ),
        "dataset_twin_invariance_tol": float(config.twin_invariance_tol),
        "dataset_twin_invariance_errors": errors,
        "dataset_twin_invariance_max_error": max_error,
        "dataset_twin_invariance_mean_error": float(np.mean(errors)) if errors else float("nan"),
        "dataset_twin_invariance_num_samples": int(num_samples),
        "dataset_twin_invariance_seed": int(random_seed),
        "dataset_twin_invariance_theta_scale": float(theta_scale),
        "theta_convention": f"trace{dim_model}",
        "sampled_theta_trace": thetas,
        "probe_config_hash": config.stable_hash(),
        "probe_config": config.to_dict(),
        "probe_group_weights": config.resolved_group_weights(),
    }
