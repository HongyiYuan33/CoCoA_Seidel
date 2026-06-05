"""Seidel symmetry ablation sweep.

This script compares classical backend Seidel parameterizations by default
using exact image-space RDM operator metrics. It never modifies the frozen RDM
forward model.

Current defaults use the classical backend family. Trace-separated
``trace5``/``trace4``/``trace3`` and trace5 proxy helpers remain only for
internal reproduction of paused trace-separated ablations; the primary CLI
does not expose them.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_ring_cocoa.evaluation import (  # noqa: E402
    OperatorProbeConfig,
    evaluate_seidel_recovery,
)
from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import (  # noqa: E402
    evaluate_trace_seidel_recovery,
    relative_wavefront_residual,
)
from hybrid_ring_cocoa.optics.seidel_psf import expand_trace_seidel  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "seidel_symmetry_ablation"
MODEL_NAMES = (
    "classical4d",
    "classical5d",
    "classical6d",
    "backend6",
)
DEFAULT_MODELS = ("classical4d", "classical5d", "classical6d")
IMAGE_CHOICES = ("baboon", "Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense")
DIRECTIONS: dict[str, np.ndarray] = {
    "balanced": np.asarray([0.30, -0.10, 0.10, 0.03, 0.00, 0.00], dtype=np.float64),
    "coma_dominant": np.asarray([0.05, 0.20, 0.04, 0.02, 0.00, 0.00], dtype=np.float64),
    "astig_field": np.asarray([0.08, 0.04, 0.32, -0.06, 0.00, 0.00], dtype=np.float64),
    "spherical_field": np.asarray([0.22, 0.02, 0.00, 0.08, 0.00, 0.00], dtype=np.float64),
    "field_curvature_only": np.asarray([0.00, 0.00, 0.00, 0.10, 0.00, 0.00], dtype=np.float64),
    "pure_astig": np.asarray([0.00, 0.00, 0.10, 0.00, 0.00, 0.00], dtype=np.float64),
    "pure_distortion": np.asarray([0.00, 0.00, 0.00, 0.00, 0.04, 0.00], dtype=np.float64),
    "coma_distortion_mixed": np.asarray([0.00, -0.10, 0.00, 0.00, 0.04, 0.00], dtype=np.float64),
    "balanced_with_D": np.asarray([0.30, -0.10, 0.10, 0.03, 0.04, 0.00], dtype=np.float64),
}
TRACE_DIRECTIONS: dict[str, np.ndarray] = {
    "balanced": np.asarray([0.30, -0.10, 0.05, 0.08, 0.00], dtype=np.float64),
    "coma_dominant": np.asarray([0.05, 0.20, 0.02, 0.04, 0.00], dtype=np.float64),
    "astig_field": np.asarray([0.08, 0.04, 0.16, 0.10, 0.00], dtype=np.float64),
    "spherical_field": np.asarray([0.22, 0.02, 0.00, 0.08, 0.00], dtype=np.float64),
    "field_curvature_only": np.asarray([0.00, 0.00, 0.00, 0.10, 0.00], dtype=np.float64),
    "pure_astig": np.asarray([0.00, 0.00, 0.10, 0.00, 0.00], dtype=np.float64),
    "pure_distortion": np.asarray([0.00, 0.00, 0.00, 0.00, 0.04], dtype=np.float64),
    "coma_distortion_mixed": np.asarray([0.00, -0.10, 0.00, 0.00, 0.04], dtype=np.float64),
    "balanced_with_D": np.asarray([0.30, -0.10, 0.05, 0.08, 0.04], dtype=np.float64),
}
TRACE_MODELS = {"trace5", "trace4", "trace3", "field_scaling_only", "spin_tying", "full_seidel"}
CLASSICAL_MODELS = {"classical4d", "classical5d", "classical6d", "backend6"}
TRAIN_FIELDS = tuple(
    (h, psi)
    for h in (0.35, 0.70)
    for psi in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
)
HELDOUT_FIELDS = (
    (0.125, -0.125),
    (0.375, -0.375),
    (0.625, -0.625),
    (0.875, -0.875),
)


@dataclass(frozen=True)
class AblationCase:
    case_id: str
    model_name: str
    image: str
    direction: str
    strength: float
    seed: int
    dim: int
    num_iter: int
    pretrain_iter: int
    sys_na: float
    lamb: float


def tag_float(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def parse_float_list(values: Iterable[str] | None, defaults: Iterable[float]) -> list[float]:
    if values is None:
        return [float(v) for v in defaults]
    return [float(v) for v in values]


def field_weighted_rms_backend(theta_backend6: np.ndarray) -> float:
    # Lightweight local RMS helper, kept independent from production forward.
    xs = np.linspace(-1.0, 1.0, 101, dtype=np.float64)
    x, y = np.meshgrid(xs, xs, indexing="xy")
    mask = (x * x + y * y) <= 1.0
    rho2 = x * x + y * y
    hs = np.linspace(0.0, 1.0, 31, dtype=np.float64)
    weights = hs.copy()
    weights[0] = 0.0
    rms = []
    for h in hs:
        w = (
            theta_backend6[0] * rho2**2
            + theta_backend6[1] * h * rho2 * x
            + theta_backend6[2] * h**2 * x**2
            + theta_backend6[3] * h**2 * rho2
            + theta_backend6[4] * h**3 * x
            + theta_backend6[5] * rho2
        )[mask]
        w = w - float(np.mean(w))
        rms.append(math.sqrt(float(np.mean(w * w))))
    denom = float(np.sum(weights))
    return float(np.sum(np.asarray(rms) * weights) / max(denom, 1e-12))


def scaled_backend_gt(direction: str, strength: float) -> np.ndarray:
    base = DIRECTIONS[direction].astype(np.float64)
    base_rms = field_weighted_rms_backend(base)
    if base_rms <= 1e-12:
        return base.copy()
    return base * (float(strength) / base_rms)


def scaled_trace_gt(direction: str, strength: float) -> np.ndarray:
    base = TRACE_DIRECTIONS[direction].astype(np.float64)
    base_backend = np.asarray(expand_trace_seidel(base, model_dim=5), dtype=np.float64)
    base_rms = field_weighted_rms_backend(base_backend)
    if base_rms <= 1e-12:
        return base.copy()
    return base * (float(strength) / base_rms)


def fixed_indices_for_model(model_name: str) -> list[int]:
    if model_name == "classical4d":
        return [4, 5]
    if model_name == "classical5d":
        return [5]
    if model_name in {"classical6d", "backend6"}:
        return []
    if model_name in {"trace5", "trace5_projected"}:
        return [5]
    if model_name in {"trace4", "trace3"}:
        return [4, 5]
    return []


def active_backend_vector(theta_backend6: np.ndarray, fixed_indices: Sequence[int]) -> np.ndarray:
    fixed = set(int(idx) for idx in fixed_indices)
    return np.asarray([theta_backend6[idx] for idx in range(6) if idx not in fixed], dtype=np.float64)


def local_coefficients(theta_trace5: np.ndarray, fields: Iterable[tuple[float, float]]) -> dict[str, np.ndarray]:
    s, c, a, f, d = [float(v) for v in theta_trace5]
    hs = []
    psis = []
    z_s = []
    z_c = []
    z_a = []
    z_f = []
    z_d = []
    for h, psi in fields:
        hs.append(float(h))
        psis.append(float(psi))
        z_s.append(s)
        z_c.append(c * h * np.exp(1j * psi))
        z_a.append(a * h * h * np.exp(2j * psi))
        z_f.append(f * h * h)
        z_d.append(d * h * h * h * np.exp(1j * psi))
    return {
        "H": np.asarray(hs, dtype=np.float64),
        "psi": np.asarray(psis, dtype=np.float64),
        "zS": np.asarray(z_s, dtype=np.complex128),
        "zC": np.asarray(z_c, dtype=np.complex128),
        "zA": np.asarray(z_a, dtype=np.complex128),
        "zF": np.asarray(z_f, dtype=np.complex128),
        "zD": np.asarray(z_d, dtype=np.complex128),
    }


def noisy_proxy(theta_trace5: np.ndarray, seed: int, strength: float) -> dict[str, np.ndarray]:
    proxy = local_coefficients(theta_trace5, TRAIN_FIELDS)
    rng = np.random.default_rng(1009 + int(seed))
    scale = max(float(strength), 1e-6) * 0.025
    for key in ("zS", "zF"):
        proxy[key] = proxy[key] + rng.normal(0.0, scale, size=proxy[key].shape)
    for key in ("zC", "zA", "zD"):
        proxy[key] = proxy[key] + (
            rng.normal(0.0, scale, size=proxy[key].shape)
            + 1j * rng.normal(0.0, scale, size=proxy[key].shape)
        )
    return proxy


def noisy_backend(theta_backend6: np.ndarray, seed: int, strength: float) -> np.ndarray:
    theta = np.asarray(theta_backend6, dtype=np.float64).reshape(6).copy()
    rng = np.random.default_rng(1009 + int(seed))
    scale = max(float(strength), 1e-6) * 0.025
    return theta + rng.normal(0.0, scale, size=theta.shape)


def fit_full_seidel(proxy: dict[str, np.ndarray]) -> np.ndarray:
    h = proxy["H"]
    psi = proxy["psi"]
    s = float(np.real(np.mean(proxy["zS"])))
    c = float(np.sum(h * np.real(proxy["zC"] * np.exp(-1j * psi))) / max(np.sum(h * h), 1e-12))
    a = float(np.sum((h**2) * np.real(proxy["zA"] * np.exp(-2j * psi))) / max(np.sum(h**4), 1e-12))
    f = float(np.sum((h**2) * np.real(proxy["zF"])) / max(np.sum(h**4), 1e-12))
    d = float(np.sum((h**3) * np.real(proxy["zD"] * np.exp(-1j * psi))) / max(np.sum(h**6), 1e-12))
    return np.asarray([s, c, a, f, d], dtype=np.float64)


def fit_trace4(proxy: dict[str, np.ndarray]) -> np.ndarray:
    full = fit_full_seidel(proxy)
    return full[:4].copy()


def fit_trace3(proxy: dict[str, np.ndarray]) -> np.ndarray:
    full = fit_full_seidel(proxy)
    return full[:3].copy()


def fit_field_scaling_only(proxy: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    h = proxy["H"]
    s = float(np.real(np.mean(proxy["zS"])))
    c = float(np.real(np.mean(proxy["zC"] / np.maximum(h, 1e-12))))
    a = float(np.real(np.mean(proxy["zA"] / np.maximum(h**2, 1e-12))))
    f = float(np.sum((h**2) * np.real(proxy["zF"])) / max(np.sum(h**4), 1e-12))
    d = float(np.real(np.mean(proxy["zD"] / np.maximum(h**3, 1e-12))))
    theta = np.asarray([s, c, a, f, d], dtype=np.float64)
    param_count = 2 + 5 * len(h)
    return theta, {"parameter_count": float(param_count)}


def fit_spin_tying(proxy: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    h = proxy["H"]
    psi = proxy["psi"]
    s = float(np.real(np.mean(proxy["zS"])))
    c_complex = np.sum(h * proxy["zC"] * np.exp(-1j * psi)) / max(np.sum(h * h), 1e-12)
    a_complex = np.sum((h**2) * proxy["zA"] * np.exp(-2j * psi)) / max(np.sum(h**4), 1e-12)
    f = float(np.sum((h**2) * np.real(proxy["zF"])) / max(np.sum(h**4), 1e-12))
    d_complex = np.sum((h**3) * proxy["zD"] * np.exp(-1j * psi)) / max(np.sum(h**6), 1e-12)
    theta = np.asarray(
        [s, float(np.real(c_complex)), float(np.real(a_complex)), f, float(np.real(d_complex))],
        dtype=np.float64,
    )
    return theta, {
        "parameter_count": 8.0,
        "spin_c_imag": float(np.imag(c_complex)),
        "spin_a_imag": float(np.imag(a_complex)),
        "spin_d_imag": float(np.imag(d_complex)),
    }


def diagnostics_from_proxy(proxy: dict[str, np.ndarray]) -> dict[str, Any]:
    h = proxy["H"]
    psi = proxy["psi"]
    zc = proxy["zC"]
    za = proxy["zA"]
    zd = proxy["zD"]
    eta_1 = float(
        np.sum(np.imag(zc * np.exp(-1j * psi)) ** 2)
        / max(float(np.sum(np.abs(zc) ** 2)), 1e-12)
    )
    eta_2 = float(
        np.sum(np.imag(za * np.exp(-2j * psi)) ** 2)
        / max(float(np.sum(np.abs(za) ** 2)), 1e-12)
    )
    eta_d = float(
        np.sum(np.imag(zd * np.exp(-1j * psi)) ** 2)
        / max(float(np.sum(np.abs(zd) ** 2)), 1e-12)
    )
    field_scaling_residual = float(
        np.mean(np.abs(proxy["zF"] - fit_full_seidel(proxy)[3] * h**2))
    )
    return {
        "eta_1": eta_1,
        "eta_2": eta_2,
        "eta_d": eta_d,
        "parity_residual": "not_applicable",
        "field_scaling_residual": field_scaling_residual,
    }


def unavailable_symmetry_diagnostics() -> dict[str, str]:
    return {
        "eta_1": "not_applicable",
        "eta_2": "not_applicable",
        "eta_d": "not_applicable",
        "parity_residual": "not_applicable",
        "field_scaling_residual": "not_applicable",
    }


def fast_probe_config(seed: int, *, heldout: bool = False) -> OperatorProbeConfig:
    offset = 10000 if heldout else 0
    points = HELDOUT_FIELDS if heldout else tuple((h, -h) for h, _ in TRAIN_FIELDS[:4])
    return OperatorProbeConfig(
        delta_grid_size=2,
        radial_basis_count=2,
        fourier_frequencies=((1, 0), (0, 1), (1, 1)),
        random_count=2,
        random_seed=1729 + int(seed) + offset,
        diagnostic_psf_points=points,
        wavefront_field_samples=11,
        wavefront_pupil_samples=51,
        twin_invariance_tol=1e-7,
    )


def evaluate_exact_metrics(
    *,
    model_name: str,
    theta_gt_backend6: np.ndarray,
    theta_final_public: np.ndarray,
    theta_final_backend6: np.ndarray,
    dim: int,
    sys_params: dict[str, float],
    seed: int,
    device: torch.device | str | None = None,
    theta_gt_trace5: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_config = fast_probe_config(seed, heldout=False)
    heldout_config = fast_probe_config(seed, heldout=True)
    theta_gt_backend6 = np.asarray(theta_gt_backend6, dtype=np.float64).reshape(6)

    if model_name in {"trace5", "full_seidel", "field_scaling_only", "spin_tying"}:
        if theta_gt_trace5 is None:
            raise ValueError(f"{model_name} requires trace5 ground truth for trace evaluator")
        train = evaluate_trace_seidel_recovery(
            theta_gt_trace5,
            theta_final_public,
            dim=dim,
            sys_params=sys_params,
            model_dim=5,
            probe_config=train_config,
            dataset_twin_invariance_pass=False,
            device=device,
        )
        heldout = evaluate_trace_seidel_recovery(
            theta_gt_trace5,
            theta_final_public,
            dim=dim,
            sys_params=sys_params,
            model_dim=5,
            probe_config=heldout_config,
            dataset_twin_invariance_pass=False,
            device=device,
        )
    else:
        train = evaluate_seidel_recovery(
            theta_gt_backend6,
            theta_final_backend6,
            dim=dim,
            sys_params=sys_params,
            fixed_indices=[],
            probe_config=train_config,
            dataset_twin_invariance_pass=False,
            device=device,
        )
        heldout = evaluate_seidel_recovery(
            theta_gt_backend6,
            theta_final_backend6,
            dim=dim,
            sys_params=sys_params,
            fixed_indices=[],
            probe_config=heldout_config,
            dataset_twin_invariance_pass=False,
            device=device,
        )
    return train, heldout


def model_parameter_count(model_name: str, proxy: dict[str, np.ndarray] | None = None) -> int:
    if model_name == "classical4d":
        return 4
    if model_name == "classical5d":
        return 5
    if model_name in {"classical6d", "backend6"}:
        return 6
    if model_name == "trace5":
        return 5
    if model_name == "trace3":
        return 3
    if model_name in {"trace4", "full_seidel"}:
        return 5 if model_name == "full_seidel" else 4
    if model_name == "spin_tying":
        return 6
    if model_name == "field_scaling_only":
        n = len(proxy["H"]) if proxy is not None else len(TRAIN_FIELDS)
        return 2 + 4 * n
    raise ValueError(model_name)


def recover_case(case: AblationCase, *, device: torch.device | str | None = None) -> dict[str, Any]:
    sys_params = {"NA": float(case.sys_na), "lamb": float(case.lamb)}
    if case.model_name in TRACE_MODELS:
        theta_gt_trace5: np.ndarray | None = scaled_trace_gt(case.direction, case.strength)
        theta_gt_backend6 = np.asarray(expand_trace_seidel(theta_gt_trace5, model_dim=5), dtype=np.float64)
        proxy: dict[str, np.ndarray] | None = noisy_proxy(theta_gt_trace5, case.seed, case.strength)
        gt_convention = "trace5"
    else:
        theta_gt_trace5 = None
        theta_gt_backend6 = scaled_backend_gt(case.direction, case.strength)
        proxy = None
        gt_convention = "classical_backend6"
    result_type = "native_exact_rdm"
    extra: dict[str, Any] = {}

    if case.model_name in {"classical4d", "classical5d", "classical6d"}:
        theta_backend6 = noisy_backend(theta_gt_backend6, case.seed, case.strength)
        fixed = fixed_indices_for_model(case.model_name)
        if fixed:
            theta_backend6[fixed] = 0.0
        theta_public = active_backend_vector(theta_backend6, fixed)
        model_dim = 6 - len(fixed)
        convention = case.model_name
    elif case.model_name in {"trace5", "full_seidel"}:
        assert proxy is not None
        theta_public = fit_full_seidel(proxy)
        theta_backend6 = np.asarray(expand_trace_seidel(theta_public, model_dim=5), dtype=np.float64)
        model_dim = 5
        convention = "trace5"
    elif case.model_name == "trace4":
        assert proxy is not None
        theta_public = fit_trace4(proxy)
        theta_backend6 = np.asarray(expand_trace_seidel(theta_public, model_dim=4), dtype=np.float64)
        model_dim = 4
        convention = "trace4"
    elif case.model_name == "trace3":
        assert proxy is not None
        theta_public = fit_trace3(proxy)
        theta_backend6 = np.asarray(expand_trace_seidel(theta_public, model_dim=3), dtype=np.float64)
        model_dim = 3
        convention = "trace3"
    elif case.model_name == "field_scaling_only":
        assert proxy is not None
        theta_public, extra = fit_field_scaling_only(proxy)
        theta_backend6 = np.asarray(expand_trace_seidel(theta_public, model_dim=5), dtype=np.float64)
        model_dim = 5
        convention = "trace5_projected"
        result_type = "projected_operator_ablation"
    elif case.model_name == "spin_tying":
        assert proxy is not None
        theta_public, extra = fit_spin_tying(proxy)
        theta_backend6 = np.asarray(expand_trace_seidel(theta_public, model_dim=5), dtype=np.float64)
        model_dim = 5
        convention = "trace5_projected"
        result_type = "projected_operator_ablation"
    elif case.model_name == "backend6":
        theta_backend6 = noisy_backend(theta_gt_backend6, case.seed, case.strength)
        rng = np.random.default_rng(7127 + int(case.seed))
        leakage = max(case.strength, 1e-6) * 0.015
        theta_backend6[5] = rng.normal(0.0, leakage)
        theta_public = theta_backend6.copy()
        model_dim = 6
        convention = "backend6"
    else:
        raise ValueError(case.model_name)

    train_metrics, heldout_metrics = evaluate_exact_metrics(
        model_name=case.model_name,
        theta_gt_backend6=theta_gt_backend6,
        theta_final_public=theta_public,
        theta_final_backend6=theta_backend6,
        dim=case.dim,
        sys_params=sys_params,
        seed=case.seed,
        device=device,
        theta_gt_trace5=theta_gt_trace5,
    )
    diagnostics = diagnostics_from_proxy(proxy) if proxy is not None else unavailable_symmetry_diagnostics()
    misspecified = bool(
        (theta_gt_trace5 is not None and case.model_name == "trace4" and abs(float(theta_gt_trace5[4])) > 1e-10)
        or (
            theta_gt_trace5 is not None
            and
            case.model_name == "trace3"
            and (abs(float(theta_gt_trace5[3])) > 1e-10 or abs(float(theta_gt_trace5[4])) > 1e-10)
        )
        or (case.model_name == "classical4d" and (
            abs(float(theta_gt_backend6[4])) > 1e-10 or abs(float(theta_gt_backend6[5])) > 1e-10
        ))
        or (case.model_name == "classical5d" and abs(float(theta_gt_backend6[5])) > 1e-10)
    )
    parameter_count = int(extra.get("parameter_count", model_parameter_count(case.model_name, proxy)))
    heldout_wavefront = relative_wavefront_residual(
        theta_gt_backend6,
        theta_backend6,
        field_samples=13,
        pupil_samples=51,
        eps=1e-12,
    )

    row: dict[str, Any] = {
        **asdict(case),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "result_type": result_type,
        "model_dim": model_dim,
        "parameter_count": parameter_count,
        "gt_convention": gt_convention,
        "theta_convention": convention,
        "theta_gt": json.dumps(
            theta_gt_trace5.tolist() if theta_gt_trace5 is not None else theta_gt_backend6.tolist()
        ),
        "theta_gt_trace5": json.dumps(
            theta_gt_trace5.tolist() if theta_gt_trace5 is not None else "not_applicable"
        ),
        "theta_final": json.dumps(np.asarray(theta_public).reshape(-1).tolist()),
        "theta_backend6_gt": json.dumps(theta_gt_backend6.tolist()),
        "theta_backend6_final": json.dumps(theta_backend6.tolist()),
        "fixed_seidel_indices": fixed_indices_for_model(convention),
        "no_defocus": 5 in fixed_indices_for_model(convention),
        "no_w311_no_defocus": 4 in fixed_indices_for_model(convention) and 5 in fixed_indices_for_model(convention),
        "distortion_forward_model": (
            "frozen_backend_W311"
            if convention.startswith("trace5")
            else ("disabled_W311_zero" if 4 in fixed_indices_for_model(convention) else "backend_W311")
        ),
        "distortion_warp": False,
        "per_field_recenter": False,
        "trace_separated_status": (
            "paused_internal_reproduction_only"
            if convention in {"trace5", "trace4", "trace3", "trace5_projected"}
            else "not_used_by_default"
        ),
        "misspecified_gt": misspecified,
        "final_loss": "not_applicable",
        "final_ssim_loss": "not_applicable",
        "measurement_nrmse": "not_applicable",
        "reconstruction_ssim": "not_applicable",
        "reconstruction_msssim": "not_applicable",
        "object_tv": "not_applicable",
        "rsd_metric": "not_applicable",
        "operator_error_strict": float(train_metrics.get("operator_error_strict", train_metrics.get("operator_error_calibrated"))),
        "operator_error_phys_equiv": float(train_metrics["operator_error_phys_equiv"]),
        "operator_error_coord_diagnostic": float(train_metrics["operator_error_coord_diagnostic"]),
        "symmetry_ambiguity_gap": float(train_metrics.get("symmetry_ambiguity_gap", train_metrics.get("physical_ambiguity_gap", 0.0))),
        "best_physical_transform": train_metrics["best_physical_transform"],
        "best_coordinate_diagnostic_transform": train_metrics["best_coordinate_diagnostic_transform"],
        "heldout_operator_error_strict": float(heldout_metrics.get("operator_error_strict", heldout_metrics.get("operator_error_calibrated"))),
        "heldout_operator_error_phys_equiv": float(heldout_metrics["operator_error_phys_equiv"]),
        "heldout_wavefront_error": float(heldout_wavefront),
        "heldout_field_set_description": json.dumps([list(p) for p in HELDOUT_FIELDS]),
        "psf_error_strict": float(train_metrics.get("psf_error_strict", train_metrics.get("psf_error_calibrated", np.nan))),
        "otf_complex_error_strict": float(train_metrics.get("otf_complex_error_strict", train_metrics.get("otf_complex_error_calibrated", np.nan))),
        "wavefront_error_strict": float(train_metrics.get("wavefront_error_strict", train_metrics.get("wavefront_error_calibrated", np.nan))),
        "raw_coeff_relative_error": float(train_metrics.get("raw_coeff_relative_error", np.nan)),
        **diagnostics,
    }
    row["train_operator_error_strict"] = row["operator_error_strict"]
    row["generalization_gap"] = row["heldout_operator_error_strict"] - row["train_operator_error_strict"]
    row.update({key: value for key, value in extra.items() if key not in row})
    return row


def build_cases(args: argparse.Namespace) -> list[AblationCase]:
    num_shards = int(args.num_shards)
    shard_index = int(args.shard_index)
    if num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {num_shards - 1}], got {shard_index}"
        )

    strengths = parse_float_list(args.strengths, [0.06])
    models = list(args.models or list(DEFAULT_MODELS))
    cases: list[AblationCase] = []
    for image in args.image:
        for model in models:
            for direction in args.directions:
                for strength in strengths:
                    for seed in range(int(args.num_seeds)):
                        case_id = "__".join(
                            [
                                image,
                                model,
                                direction,
                                f"rms{tag_float(strength)}",
                                f"seed{seed}",
                                f"dim{args.dim}",
                            ]
                        )
                        cases.append(
                            AblationCase(
                                case_id=case_id,
                                model_name=model,
                                image=image,
                                direction=direction,
                                strength=float(strength),
                                seed=int(seed),
                                dim=int(args.dim),
                                num_iter=int(args.num_iter),
                                pretrain_iter=int(args.pretrain_iter),
                                sys_na=float(args.sys_na),
                                lamb=float(args.lamb),
                            )
                        )
    if num_shards > 1:
        cases = [
            case
            for index, case in enumerate(cases)
            if (index % num_shards) == shard_index
        ]
    if args.limit is not None:
        cases = cases[: int(args.limit)]
    return cases


def metrics_path(output_root: Path, case: AblationCase) -> Path:
    return output_root / "cases" / case.case_id / "metrics.json"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def write_csv_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        tmp.write_text("")
        tmp.replace(path)
        return
    preferred = [
        "case_id",
        "model_name",
        "result_type",
        "image",
        "direction",
        "strength",
        "seed",
        "dim",
        "parameter_count",
        "misspecified_gt",
        "operator_error_strict",
        "heldout_operator_error_strict",
        "generalization_gap",
        "eta_1",
        "eta_2",
        "eta_d",
        "fixed_seidel_indices",
        "no_defocus",
        "no_w311_no_defocus",
        "parity_residual",
        "field_scaling_residual",
    ]
    extras = sorted({k for row in rows for k in row if k not in preferred})
    fieldnames = [k for k in preferred if any(k in row for row in rows)] + extras
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def collect_completed(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output_root / "cases").glob("*/metrics.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def grouped_summary(rows: list[dict[str, Any]], failure_threshold: float) -> list[dict[str, Any]]:
    out = []
    for model in sorted({row["model_name"] for row in rows}):
        group = [row for row in rows if row["model_name"] == model]
        op = np.asarray([float(row["operator_error_strict"]) for row in group], dtype=np.float64)
        held = np.asarray([float(row["heldout_operator_error_strict"]) for row in group], dtype=np.float64)
        gap = np.asarray([float(row["generalization_gap"]) for row in group], dtype=np.float64)
        pc = np.asarray([float(row["parameter_count"]) for row in group], dtype=np.float64)
        out.append(
            {
                "model_name": model,
                "num_cases": len(group),
                "mean_operator_error_strict": float(np.mean(op)),
                "median_operator_error_strict": float(np.median(op)),
                "mean_heldout_operator_error_strict": float(np.mean(held)),
                "median_heldout_operator_error_strict": float(np.median(held)),
                "mean_generalization_gap": float(np.mean(gap)),
                "mean_parameter_count": float(np.mean(pc)),
                "failure_rate": float(np.mean(op > float(failure_threshold))),
            }
        )
    out.sort(key=lambda row: row["mean_heldout_operator_error_strict"])
    return out


def make_plots(rows: list[dict[str, Any]], output_root: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({row["model_name"] for row in rows})

    def values(key: str, model: str) -> list[float]:
        return [float(row[key]) for row in rows if row["model_name"] == model]

    plot_specs = [
        ("operator_error_by_model.png", "operator_error_strict", "Train exact operator error"),
        ("heldout_operator_error_by_model.png", "heldout_operator_error_strict", "Held-out exact operator error"),
        ("generalization_gap_by_model.png", "generalization_gap", "Generalization gap"),
    ]
    for filename, key, title in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        boxplot_values = [values(key, model) for model in models]
        try:
            ax.boxplot(boxplot_values, tick_labels=models, showmeans=True)
        except TypeError:
            ax.boxplot(boxplot_values, labels=models, showmeans=True)
        ax.set_title(title)
        ax.set_ylabel(key)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in models:
        xs = values("parameter_count", model)
        ys = values("heldout_operator_error_strict", model)
        ax.scatter(xs, ys, label=model, alpha=0.75)
    ax.set_xlabel("parameter_count")
    ax.set_ylabel("heldout_operator_error_strict")
    ax.set_title("Search space vs held-out operator error")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "parameter_count_vs_heldout_operator_error.png", dpi=140)
    plt.close(fig)


def write_summary(output_root: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    summary = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit_hash(),
        "args": vars(args),
        "probe_config_hash_train": fast_probe_config(0, heldout=False).stable_hash(),
        "probe_config_hash_heldout": fast_probe_config(0, heldout=True).stable_hash(),
        "dataset_twin_invariance_report": "not_run_default_identity_only",
        "model_definitions": {
            "classical4d": "default classical backend active [W040,W131,W222,W220], fixed W311=Wd=0",
            "classical5d": "default classical backend active [W040,W131,W222,W220,W311], fixed Wd=0",
            "classical6d": "default classical backend [W040,W131,W222,W220,W311,Wd]",
            "backend6": "legacy alias for classical6d; backend [W040,W131,W222,W220,W311,Wd]",
            "trace5": "paused internal reproduction helper; not exposed by the primary CLI",
            "trace4": "paused internal reproduction helper; not exposed by the primary CLI",
            "trace3": "paused internal reproduction helper; not exposed by the primary CLI",
            "field_scaling_only": "paused internal trace proxy helper; not exposed by the primary CLI",
            "spin_tying": "paused internal trace proxy helper; not exposed by the primary CLI",
            "full_seidel": "paused internal trace helper; not exposed by the primary CLI",
        },
        "tolerances": {"failure_threshold": float(args.failure_threshold)},
        "grouped_summary": grouped_summary(rows, float(args.failure_threshold)),
        "top_failures": sorted(
            rows,
            key=lambda row: float(row["heldout_operator_error_strict"]),
            reverse=True,
        )[:10],
    }
    write_json_atomic(output_root / "ablation_summary.json", summary)


def resolve_eval_device(device_arg: str) -> torch.device | None:
    if device_arg == "auto":
        return None
    return torch.device(device_arg)


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    cases = build_cases(args)
    if args.aggregate_only:
        rows = collect_completed(output_root)
        write_csv_atomic(rows, output_root / "ablation_results.csv")
        write_summary(output_root, args, rows)
        if not args.no_plots:
            make_plots(rows, output_root)
        print(json.dumps({"aggregate_only": True, "num_completed_cases": len(rows)}, indent=2))
        return rows

    if args.dry_run:
        print(json.dumps({"num_cases": len(cases), "cases": [asdict(c) for c in cases]}, indent=2))
        return []

    rows: list[dict[str, Any]] = []
    eval_device = resolve_eval_device(str(args.device))
    for idx, case in enumerate(cases, start=1):
        path = metrics_path(output_root, case)
        if args.resume and path.is_file():
            row = json.loads(path.read_text())
            print(f"[{idx}/{len(cases)}] skip {case.case_id}", flush=True)
        else:
            print(f"[{idx}/{len(cases)}] run {case.case_id}", flush=True)
            row = recover_case(case, device=eval_device)
            write_json_atomic(path, row)
        rows.append(row)
        if not args.skip_aggregate:
            completed = collect_completed(output_root)
            write_csv_atomic(completed, output_root / "ablation_results.csv")
            write_summary(output_root, args, completed)
            if not args.no_plots:
                make_plots(completed, output_root)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-iter", type=int, default=50)
    parser.add_argument("--pretrain-iter", type=int, default=20)
    parser.add_argument("--sys-na", type=float, default=0.45)
    parser.add_argument("--lamb", type=float, default=0.55e-6)
    parser.add_argument("--image", nargs="+", choices=IMAGE_CHOICES, default=["baboon"])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(DEFAULT_MODELS),
        help=(
            "Models to compare. Defaults to classical4d/classical5d/classical6d. "
            "Trace-separated and proxy modes are paused and not exposed by this "
            "primary CLI."
        ),
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        default=["0.06"],
        help="Target field-weighted RMS strengths.",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=sorted(DIRECTIONS),
        default=["balanced", "coma_dominant", "astig_field"],
        help=(
            "Ground-truth direction names. Defaults are backend-classical directions."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch evaluation device: auto, cpu, cuda, cuda:0, cuda:1, ...",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the deterministic case matrix into this many modulo shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Run only cases whose deterministic matrix index belongs to this shard.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Only write per-case metrics; useful for parallel shards sharing an output root.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild CSV/summary/plots from completed case markers and run no cases.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--failure-threshold", type=float, default=0.1)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but torch.cuda.is_available() is false")
    run_sweep(args)


if __name__ == "__main__":
    main()
