"""Blind Seidel recovery sweep with classical backend measurements by default.

This runner tests blind object+aberration recovery without altering the frozen
RDM forward model.  Current defaults use the classical backend coefficient
family and synthesize measurements directly from backend-6D Seidel vectors.

This runner reports strict/calibrated operator diagnostics directly from
``RingOperatorProbeEvaluator`` for sweep triage. Use
``evaluate_seidel_physical_operator_sweep.py`` for the post-hoc
physical-equivalence metrics, coordinate diagnostics, and twin gating columns;
strict-only metrics are not the final physical-equivalence score.

``trace5``/``trace4``/``trace3`` helpers remain in this module only for
internal reproduction of paused trace-separated experiments; the primary CLI
does not expose them.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_cocoa_like_2d_mechanism as cocoa  # noqa: E402
from hybrid_ring_cocoa.evaluation import OperatorProbeConfig  # noqa: E402
from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import (  # noqa: E402
    RingOperatorProbeEvaluator,
    relative_wavefront_residual,
)
from hybrid_ring_cocoa.optics.seidel_psf import expand_trace_seidel  # noqa: E402


MODEL_NAMES = (
    "classical4d",
    "classical5d",
    "classical6d",
    "backend6",
)
DEFAULT_IMAGES = ("Test_figure_1", "dendrites_dense")
DEFAULT_MODELS = ("classical4d", "classical5d", "classical6d")
DEFAULT_DIRECTIONS = (
    "balanced",
    "coma_dominant",
    "astig_field",
)
SANITY_STRENGTHS = (0.06,)
MEDIUM_STRENGTHS = (0.04, 0.06, 0.08, 0.10)
SANITY_SEEDS = (0,)
MEDIUM_SEEDS = (0, 1, 2)
DIRECTIONS: dict[str, np.ndarray] = {
    "balanced": np.asarray([0.30, -0.10, 0.10, 0.03, 0.00, 0.00], dtype=np.float64),
    "coma_dominant": np.asarray([0.05, 0.20, 0.04, 0.02, 0.00, 0.00], dtype=np.float64),
    "astig_field": np.asarray([0.08, 0.04, 0.32, -0.06, 0.00, 0.00], dtype=np.float64),
    "pure_distortion": np.asarray([0.0, 0.0, 0.0, 0.0, 0.04, 0.00], dtype=np.float64),
    "coma_distortion_mixed": np.asarray([0.0, -0.10, 0.0, 0.0, 0.04, 0.00], dtype=np.float64),
    "balanced_with_D": np.asarray([0.30, -0.10, 0.10, 0.03, 0.04, 0.00], dtype=np.float64),
}
TRACE_DIRECTIONS: dict[str, np.ndarray] = {
    "balanced": np.asarray([0.30, -0.10, 0.05, 0.08, 0.0], dtype=np.float64),
    "coma_dominant": np.asarray([0.05, 0.20, 0.02, 0.04, 0.0], dtype=np.float64),
    "astig_field": np.asarray([0.08, 0.04, 0.16, 0.10, 0.0], dtype=np.float64),
    "pure_distortion": np.asarray([0.0, 0.0, 0.0, 0.0, 0.04], dtype=np.float64),
    "coma_distortion_mixed": np.asarray([0.0, -0.10, 0.0, 0.0, 0.04], dtype=np.float64),
    "balanced_with_D": np.asarray([0.30, -0.10, 0.05, 0.08, 0.04], dtype=np.float64),
}
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "seidel_blind_recovery"
TRACE_MODELS = {"trace5", "trace4", "trace3"}
CLASSICAL_MODELS = {"backend6", "classical4d", "classical5d", "classical6d"}


@dataclass(frozen=True)
class BlindCase:
    case_id: str
    stage: str
    image: str
    model_name: str
    direction: str
    strength: float
    seed: int
    dim: int
    pretrain_iter: int
    num_iter: int
    sys_na: float
    lamb: float


def tag_float(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")


def parse_float_list(values: Iterable[str] | None, defaults: Iterable[float]) -> list[float]:
    if values is None:
        return [float(v) for v in defaults]
    return [float(v) for v in values]


def parse_int_list(values: Iterable[str] | None, defaults: Iterable[int]) -> list[int]:
    if values is None:
        return [int(v) for v in defaults]
    return [int(v) for v in values]


def trace_model_dim(model_name: str) -> int | None:
    if model_name == "trace5":
        return 5
    if model_name == "trace4":
        return 4
    if model_name == "trace3":
        return 3
    if model_name in {"backend6", "classical4d", "classical5d", "classical6d"}:
        return None
    raise ValueError(f"Unknown model_name={model_name!r}")


def fixed_indices_for_model(model_name: str) -> list[int]:
    if model_name == "classical4d":
        return [4, 5]
    if model_name == "classical5d":
        return [5]
    if model_name in {"classical6d", "backend6"}:
        return []
    if model_name == "trace5":
        return [5]
    if model_name in {"trace4", "trace3"}:
        return [4, 5]
    raise ValueError(model_name)

def field_weighted_rms_backend(theta_backend6: np.ndarray) -> float:
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
    backend = np.asarray(expand_trace_seidel(base, model_dim=5), dtype=np.float64)
    base_rms = field_weighted_rms_backend(backend)
    if base_rms <= 1e-12:
        return base.copy()
    return base * (float(strength) / base_rms)


def theta_for_model_target(
    theta_gt_backend6: np.ndarray,
    model_name: str,
    *,
    theta_trace5_gt: np.ndarray | None = None,
) -> np.ndarray:
    theta_gt_backend6 = np.asarray(theta_gt_backend6, dtype=np.float64).reshape(-1)
    if model_name in CLASSICAL_MODELS:
        backend = theta_gt_backend6.astype(np.float32)
        fixed = fixed_indices_for_model(model_name)
        if fixed:
            backend[fixed] = 0.0
        return backend
    if theta_trace5_gt is None:
        raise ValueError(f"{model_name} requires an explicit trace5 ground-truth vector")
    theta_trace5_gt = np.asarray(theta_trace5_gt, dtype=np.float64).reshape(-1)
    if model_name == "trace4":
        return theta_trace5_gt[:4].astype(np.float32)
    if model_name == "trace3":
        return theta_trace5_gt[:3].astype(np.float32)
    if model_name == "trace5":
        return theta_trace5_gt.astype(np.float32)
    raise ValueError(model_name)


def theta_to_backend6(theta: np.ndarray, model_name: str) -> np.ndarray:
    arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    if model_name in {"backend6", "classical4d", "classical5d", "classical6d"}:
        if arr.size != 6:
            raise ValueError(f"{model_name} theta must have 6 backend entries")
        fixed = fixed_indices_for_model(model_name)
        if fixed:
            arr = arr.copy()
            arr[fixed] = 0.0
        return arr
    return np.asarray(expand_trace_seidel(arr, model_dim=trace_model_dim(model_name)), dtype=np.float64)


def fast_probe_config(seed: int) -> OperatorProbeConfig:
    return OperatorProbeConfig(
        delta_grid_size=2,
        radial_basis_count=2,
        fourier_frequencies=((1, 0), (0, 1), (1, 1)),
        random_count=2,
        random_seed=1729 + int(seed),
        diagnostic_psf_points=((0.0, 0.0), (0.25, -0.25), (0.5, -0.5), (0.75, -0.75)),
        wavefront_field_samples=11,
        wavefront_pupil_samples=51,
        twin_invariance_tol=1e-7,
    )


def exact_operator_metrics(
    *,
    theta_gt_backend6: np.ndarray,
    theta_hat: np.ndarray,
    model_name: str,
    dim: int,
    sys_params: dict[str, float],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    theta_gt_backend6 = np.asarray(theta_gt_backend6, dtype=np.float64).reshape(-1)
    theta_hat_backend6 = theta_to_backend6(theta_hat, model_name)
    zero_public = np.zeros_like(np.asarray(theta_hat, dtype=np.float64).reshape(-1))
    zero_backend6 = theta_to_backend6(zero_public, model_name)
    evaluator = RingOperatorProbeEvaluator(
        dim=int(dim),
        sys_params=sys_params,
        probe_config=fast_probe_config(seed),
        device=device,
    )
    initial_error = float(evaluator.distance(theta_gt_backend6, zero_backend6))
    final_error = float(evaluator.distance(theta_gt_backend6, theta_hat_backend6))
    ratio = float(final_error / max(initial_error, 1e-12))
    wavefront_error = relative_wavefront_residual(
        theta_gt_backend6,
        theta_hat_backend6,
        field_samples=13,
        pupil_samples=51,
        eps=1e-12,
    )
    return {
        "theta_gt_backend6": theta_gt_backend6.tolist(),
        "theta_hat_backend6": theta_hat_backend6.tolist(),
        "operator_error_initial": initial_error,
        "operator_error_strict": final_error,
        "operator_error_improvement_ratio": ratio,
        "operator_error_reduction": float(1.0 - ratio),
        "operator_error_improved": bool(final_error < initial_error),
        "wavefront_error_strict": float(wavefront_error),
        "probe_config_hash": fast_probe_config(seed).stable_hash(),
    }


def build_cases(args: argparse.Namespace) -> list[BlindCase]:
    stage = str(args.stage)
    default_strengths = SANITY_STRENGTHS if stage == "sanity" else MEDIUM_STRENGTHS
    default_seeds = SANITY_SEEDS if stage == "sanity" else MEDIUM_SEEDS
    strengths = parse_float_list(args.strengths, default_strengths)
    seeds = parse_int_list(args.seeds, default_seeds)
    models = list(args.models or DEFAULT_MODELS)
    images = list(args.images or DEFAULT_IMAGES)
    directions = list(args.directions or DEFAULT_DIRECTIONS)
    cases: list[BlindCase] = []
    for image in images:
        for model in models:
            for direction in directions:
                for strength in strengths:
                    for seed in seeds:
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
                            BlindCase(
                                case_id=case_id,
                                stage=stage,
                                image=image,
                                model_name=model,
                                direction=direction,
                                strength=float(strength),
                                seed=int(seed),
                                dim=int(args.dim),
                                pretrain_iter=int(args.pretrain_iter),
                                num_iter=int(args.num_iter),
                                sys_na=float(args.sys_na),
                                lamb=float(args.lamb),
                            )
                        )
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if int(args.num_shards) > 1:
        cases = [
            case
            for index, case in enumerate(cases)
            if index % int(args.num_shards) == int(args.shard_index)
        ]
    if args.limit is not None:
        cases = cases[: int(args.limit)]
    return cases


def metrics_path(output_root: Path, case: BlindCase) -> Path:
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
        "stage",
        "status",
        "image",
        "model_name",
        "direction",
        "strength",
        "seed",
        "dim",
        "misspecified_gt",
        "loss_decreased",
        "loss_first",
        "loss_final",
        "loss_descent_ratio",
        "operator_error_initial",
        "operator_error_strict",
        "operator_error_reduction",
        "operator_error_improved",
        "wavefront_error_strict",
        "W311_hat",
        "Wd_hat",
        "gauge_leakage_l2",
        "nrmse_meas_pred_vs_meas",
        "ssim_recon_gain_vs_gt",
        "elapsed_s",
    ]
    extras = sorted({key for row in rows for key in row if key not in preferred})
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + extras
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, dict, tuple)):
                    value = json.dumps(value)
                out[key] = value
            writer.writerow(out)
    tmp.replace(path)


def collect_completed(output_root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text())
        for path in sorted((output_root / "cases").glob("*/metrics.json"))
    ]


def run_args_for_case(
    case: BlindCase,
    model_target: np.ndarray,
    args: argparse.Namespace,
    *,
    gt_source: str,
) -> SimpleNamespace:
    fixed = fixed_indices_for_model(case.model_name)
    if 5 in fixed:
        anchor_weight = 0.0
    elif case.model_name in {"backend6", "classical6d"}:
        anchor_weight = float(args.backend6_defocus_anchor_weight)
    else:
        anchor_weight = float(args.defocus_anchor_weight)
    return SimpleNamespace(
        image=case.image,
        size=case.dim,
        modes=["joint"],
        run_name=None,
        num_iter=case.num_iter,
        pretrain_iter=case.pretrain_iter,
        lr_obj=float(args.lr_obj),
        lr_seidel=float(args.lr_seidel),
        rsd_weight=float(args.rsd_weight),
        tv_weight=float(args.tv_weight),
        pretrain_scalar=float(args.pretrain_scalar),
        defocus_anchor_weight=anchor_weight,
        defocus_index=int(args.defocus_index),
        scheduler=None if args.scheduler == "none" else args.scheduler,
        eta_min_ratio=float(args.eta_min_ratio),
        max_val=float(args.max_val),
        nerf_beta=float(args.nerf_beta),
        nerf_depth=int(args.nerf_depth),
        nerf_width=int(args.nerf_width),
        nerf_skips=tuple(args.nerf_skips),
        output_mode=args.output_mode,
        seidel_convention=case.model_name,
        gt_preset="custom",
        gt_seidel_json=json.dumps(model_target.astype(float).tolist()),
        gt_label=f"{case.direction}__rms{tag_float(case.strength)}",
        gt_source=gt_source,
        seed=case.seed,
        verbose=bool(args.train_verbose),
    )


def run_case(case: BlindCase, args: argparse.Namespace, *, device: torch.device) -> dict[str, Any]:
    t0 = time.time()
    sys_params = {"NA": float(case.sys_na), "lamb": float(case.lamb)}
    model_dim = trace_model_dim(case.model_name)
    if model_dim is None:
        theta_trace5_gt: np.ndarray | None = None
        theta_gt_backend6 = scaled_backend_gt(case.direction, case.strength)
        gt_convention = "classical_backend6"
        measurement_generated_by = "classical_backend6"
        gt_source = "classical_backend_measurement_custom"
    else:
        theta_trace5_gt = scaled_trace_gt(case.direction, case.strength)
        theta_gt_backend6 = np.asarray(expand_trace_seidel(theta_trace5_gt, model_dim=5), dtype=np.float64)
        gt_convention = "trace5"
        measurement_generated_by = "trace5_no_defocus"
        gt_source = "trace5_no_defocus_measurement_custom"
    model_target = theta_for_model_target(
        theta_gt_backend6,
        case.model_name,
        theta_trace5_gt=theta_trace5_gt,
    )
    run_args = run_args_for_case(case, model_target, args, gt_source=gt_source)
    fixed = fixed_indices_for_model(case.model_name)
    d_nonzero = abs(float(theta_gt_backend6[4])) > 1e-10
    wd_nonzero = abs(float(theta_gt_backend6[5])) > 1e-10
    if theta_trace5_gt is not None:
        f_nonzero = abs(float(theta_trace5_gt[3])) > 1e-10
    else:
        f_nonzero = False
    row_base: dict[str, Any] = {
        **asdict(case),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "gt_convention": gt_convention,
        "theta_gt_trace5": theta_trace5_gt.tolist() if theta_trace5_gt is not None else "not_applicable",
        "theta_gt_backend6": theta_gt_backend6.tolist(),
        "theta_model_target": model_target.astype(float).tolist(),
        "measurement_generated_by": measurement_generated_by,
        "measurement_model_dim": model_dim if model_dim is not None else 6,
        "theta_convention": case.model_name,
        "fixed_seidel_indices": fixed,
        "no_defocus": 5 in fixed,
        "no_w311_no_defocus": 4 in fixed and 5 in fixed,
        "distortion_forward_model": (
            "frozen_backend_W311"
            if case.model_name == "trace5"
            else ("disabled_W311_zero" if 4 in fixed else "backend_W311")
        ),
        "distortion_warp": False,
        "per_field_recenter": False,
        "trace_separated_status": (
            "paused_internal_reproduction_only"
            if case.model_name in {"trace5", "trace4", "trace3"}
            else "not_used_by_default"
        ),
        "misspecified_gt": bool(
            (case.model_name == "trace4" and d_nonzero)
            or (case.model_name == "trace3" and (f_nonzero or d_nonzero))
            or (case.model_name == "classical4d" and (d_nonzero or wd_nonzero))
            or (case.model_name == "classical5d" and wd_nonzero)
        ),
        "backend6_defocus_anchor_weight": float(run_args.defocus_anchor_weight),
        "nerf_depth": int(args.nerf_depth),
        "nerf_width": int(args.nerf_width),
        "nerf_skips": tuple(args.nerf_skips),
    }
    try:
        torch.manual_seed(case.seed)
        np.random.seed(case.seed)
        sharp_gt = cocoa.load_baboon_gt(
            case.dim,
            path=cocoa.IMAGE_PATHS[case.image],
            device=device,
        )
        if model_dim is None:
            theta_gt_t = torch.as_tensor(theta_gt_backend6, dtype=sharp_gt.dtype, device=device)
            meas_gt = cocoa.synthesize_measurement(sharp_gt, theta_gt_t, sys_params)
        else:
            theta_trace5_t = torch.as_tensor(theta_trace5_gt, dtype=sharp_gt.dtype, device=device)
            meas_gt = cocoa.synthesize_trace_measurement(
                sharp_gt,
                theta_trace5_t,
                sys_params,
                model_dim=5,
            )

        net_obj = cocoa.CocoaLikeObject2D(
            max_val=float(args.max_val),
            beta=float(args.nerf_beta),
            output_mode=args.output_mode,
            depth=int(args.nerf_depth),
            width=int(args.nerf_width),
            skips=tuple(args.nerf_skips),
        ).to(device)
        seidel = nn.Parameter(
            torch.zeros(
                6 if model_dim is None else int(model_dim),
                dtype=sharp_gt.dtype,
                device=device,
            )
        )
        pretrain_history = cocoa.pretrain_cocoa_like(
            net_obj,
            meas_gt,
            num_iter=case.pretrain_iter,
            lr=float(args.lr_obj),
            measurement_scalar=float(args.pretrain_scalar),
            verbose=bool(args.train_verbose),
        )
        result = cocoa.train_cocoa_like(
            net_obj,
            seidel,
            meas_gt,
            sys_params,
            mode="joint",
            num_iter=case.num_iter,
            lr_obj=float(args.lr_obj),
            lr_seidel=float(args.lr_seidel),
            rsd_weight=float(args.rsd_weight),
            tv_weight=float(args.tv_weight),
            defocus_anchor_weight=float(run_args.defocus_anchor_weight),
            defocus_index=int(args.defocus_index),
            seidel_model_dim=model_dim,
            fixed_seidel_indices=fixed if model_dim is None else [],
            scheduler=None if args.scheduler == "none" else args.scheduler,
            eta_min_ratio=float(args.eta_min_ratio),
            pretrain_history=pretrain_history,
            verbose=bool(args.train_verbose),
        )
        result = result._replace(elapsed_s=time.time() - t0)
        base_metrics = cocoa.compute_metrics(
            sharp_gt,
            meas_gt,
            result,
            model_target,
            mode="joint",
            args=run_args,
        )
        theta_hat_public = np.asarray(base_metrics["seidel_final"], dtype=np.float64)
        exact = exact_operator_metrics(
            theta_gt_backend6=theta_gt_backend6,
            theta_hat=theta_hat_public,
            model_name=case.model_name,
            dim=case.dim,
            sys_params=sys_params,
            seed=case.seed,
            device=device,
        )
        loss = np.asarray(result.loss_history, dtype=np.float64)
        ssim_loss = np.asarray(result.ssim_history, dtype=np.float64)
        pre = np.asarray(pretrain_history, dtype=np.float64) if pretrain_history else np.asarray([], dtype=np.float64)
        theta_hat_backend6 = np.asarray(exact["theta_hat_backend6"], dtype=np.float64)
        gauge_l2 = float(abs(theta_hat_backend6[5]))
        row = {
            **row_base,
            **base_metrics,
            **exact,
            "status": "success",
            "theta_hat_public": theta_hat_public.tolist(),
            "loss_first": float(loss[0]) if loss.size else float("nan"),
            "loss_final": float(loss[-1]) if loss.size else float("nan"),
            "loss_min": float(np.min(loss)) if loss.size else float("nan"),
            "loss_descent_ratio": float(loss[-1] / max(loss[0], 1e-12)) if loss.size else float("nan"),
            "loss_decreased": bool(loss.size and loss[-1] < loss[0]),
            "ssim_loss_first": float(ssim_loss[0]) if ssim_loss.size else float("nan"),
            "ssim_loss_final": float(ssim_loss[-1]) if ssim_loss.size else float("nan"),
            "pretrain_loss_first": float(pre[0]) if pre.size else float("nan"),
            "pretrain_loss_final": float(pre[-1]) if pre.size else float("nan"),
            "pretrain_loss_decreased": bool(pre.size and pre[-1] < pre[0]),
            "W311_hat": float(theta_hat_backend6[4]),
            "Wd_hat": float(theta_hat_backend6[5]),
            "gauge_leakage_l2": gauge_l2,
            "gauge_leakage_abs_sum": gauge_l2,
        }
    except Exception as exc:
        row = {
            **row_base,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_s": float(time.time() - t0),
        }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def sanity_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if row.get("status") == "success"]
    backend = [row for row in success if row.get("model_name") == "backend6"]
    classical4 = [row for row in success if row.get("model_name") == "classical4d"]
    classical5 = [row for row in success if row.get("model_name") == "classical5d"]
    classical6 = [row for row in success if row.get("model_name") == "classical6d"]
    classical4_errors = [
        float(row["operator_error_strict"]) for row in classical4 if finite_number(row.get("operator_error_strict"))
    ]
    classical5_errors = [
        float(row["operator_error_strict"]) for row in classical5 if finite_number(row.get("operator_error_strict"))
    ]
    classical6_errors = [
        float(row["operator_error_strict"]) for row in classical6 if finite_number(row.get("operator_error_strict"))
    ]
    checks = {
        "all_cases_success": len(success) == len(rows) and len(rows) > 0,
        "all_finite_operator_errors": all(finite_number(row.get("operator_error_strict")) for row in success),
        "all_loss_decreased": all(bool(row.get("loss_decreased")) for row in success),
    }
    if classical4:
        checks["classical4d_operator_improved_all"] = all(
            bool(row.get("operator_error_improved")) for row in classical4
        )
    if classical5:
        checks["classical5d_operator_improved_all"] = all(
            bool(row.get("operator_error_improved")) for row in classical5
        )
    if classical6:
        checks["classical6d_operator_improved_all"] = all(
            bool(row.get("operator_error_improved")) for row in classical6
        )
    if backend:
        checks["backend6_gauge_diagnostics_present"] = all(
            finite_number(row.get("W311_hat")) and finite_number(row.get("Wd_hat"))
            for row in backend
        )
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "num_rows": len(rows),
        "num_success": len(success),
        "classical4d_mean_operator_error": float(np.mean(classical4_errors)) if classical4_errors else None,
        "classical5d_mean_operator_error": float(np.mean(classical5_errors)) if classical5_errors else None,
        "classical6d_mean_operator_error": float(np.mean(classical6_errors)) if classical6_errors else None,
    }


def grouped_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({(row.get("model_name"), row.get("image")) for row in rows})
    out = []
    for model, image in groups:
        group = [row for row in rows if row.get("model_name") == model and row.get("image") == image]
        success = [row for row in group if row.get("status") == "success"]
        op = [float(row["operator_error_strict"]) for row in success if finite_number(row.get("operator_error_strict"))]
        red = [float(row["operator_error_reduction"]) for row in success if finite_number(row.get("operator_error_reduction"))]
        wf = [float(row["wavefront_error_strict"]) for row in success if finite_number(row.get("wavefront_error_strict"))]
        gauge = [float(row.get("gauge_leakage_l2", 0.0)) for row in success if finite_number(row.get("gauge_leakage_l2"))]
        out.append(
            {
                "model_name": model,
                "image": image,
                "num_cases": len(group),
                "num_success": len(success),
                "failure_rate": float(1.0 - len(success) / max(len(group), 1)),
                "mean_operator_error_strict": float(np.mean(op)) if op else None,
                "median_operator_error_strict": float(np.median(op)) if op else None,
                "mean_operator_error_reduction": float(np.mean(red)) if red else None,
                "mean_wavefront_error_strict": float(np.mean(wf)) if wf else None,
                "mean_gauge_leakage_l2": float(np.mean(gauge)) if gauge else None,
            }
        )
    out.sort(key=lambda row: (str(row["image"]), row["mean_operator_error_strict"] if row["mean_operator_error_strict"] is not None else float("inf")))
    return out


def make_plots(rows: list[dict[str, Any]], output_root: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    success = [row for row in rows if row.get("status") == "success"]
    models = [model for model in MODEL_NAMES if any(row.get("model_name") == model for row in success)]

    def vals(key: str, model: str) -> list[float]:
        return [
            float(row[key])
            for row in success
            if row.get("model_name") == model and finite_number(row.get(key))
        ]

    for filename, key, title in (
        ("operator_error_by_model.png", "operator_error_strict", "Exact operator error"),
        ("operator_error_reduction_by_model.png", "operator_error_reduction", "Exact operator error reduction"),
        ("wavefront_error_by_model.png", "wavefront_error_strict", "Wavefront error"),
        ("gauge_leakage_by_model.png", "gauge_leakage_l2", "Wd gauge leakage"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        data = [vals(key, model) for model in models]
        try:
            ax.boxplot(data, tick_labels=models, showmeans=True)
        except TypeError:
            ax.boxplot(data, labels=models, showmeans=True)
        ax.set_title(title)
        ax.set_ylabel(key)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in models:
        group = [row for row in success if row.get("model_name") == model]
        ax.scatter(
            [float(row["strength"]) for row in group],
            [float(row["operator_error_strict"]) for row in group],
            label=model,
            alpha=0.7,
        )
    ax.set_xlabel("strength")
    ax.set_ylabel("operator_error_strict")
    ax.set_title("Operator error vs strength")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "operator_error_vs_strength.png", dpi=140)
    plt.close(fig)


def write_summary(output_root: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    summary = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "model_definitions": {
            "classical4d": "default classical backend active [W040,W131,W222,W220], fixed W311=Wd=0",
            "classical5d": "default classical backend active [W040,W131,W222,W220,W311], fixed Wd=0",
            "classical6d": "default classical backend [W040,W131,W222,W220,W311,Wd]",
            "backend6": "legacy alias for classical6d; retained for old runs",
            "trace5": "paused internal reproduction helper; not exposed by the primary CLI",
            "trace4": "paused internal reproduction helper; not exposed by the primary CLI",
            "trace3": "paused internal reproduction helper; not exposed by the primary CLI",
        },
        "trace_separated_status": "paused_internal_reproduction_only",
        "num_rows": len(rows),
        "num_success": sum(1 for row in rows if row.get("status") == "success"),
        "sanity_gate": sanity_gate(rows),
        "grouped_summary": grouped_summary(rows),
    }
    write_json_atomic(output_root / "blind_recovery_summary.json", summary)


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    cases = build_cases(args)
    if args.aggregate_only:
        rows = collect_completed(output_root)
        write_csv_atomic(rows, output_root / "blind_recovery_results.csv")
        write_summary(output_root, args, rows)
        if not args.no_plots:
            make_plots(rows, output_root)
        print(json.dumps({"aggregate_only": True, "num_completed_cases": len(rows)}, indent=2))
        return rows
    if args.dry_run:
        print(json.dumps({"num_cases": len(cases), "cases": [asdict(case) for case in cases]}, indent=2))
        return []

    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        path = metrics_path(output_root, case)
        if args.resume and path.is_file() and not args.force:
            row = json.loads(path.read_text())
            print(f"[{idx}/{len(cases)}] skip {case.case_id}", flush=True)
        else:
            print(f"[{idx}/{len(cases)}] run {case.case_id}", flush=True)
            row = run_case(case, args, device=device)
            write_json_atomic(path, row)
        rows.append(row)
        if not args.skip_aggregate:
            completed = collect_completed(output_root)
            write_csv_atomic(completed, output_root / "blind_recovery_results.csv")
            write_summary(output_root, args, completed)
            if not args.no_plots:
                make_plots(completed, output_root)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sanity", "medium"], default="sanity")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--images", nargs="+", choices=sorted(cocoa.IMAGE_PATHS), default=list(DEFAULT_IMAGES))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(DEFAULT_MODELS),
        help=(
            "Models to run. Defaults to classical4d/classical5d/classical6d. "
            "Trace-separated modes are paused and not exposed by this primary CLI."
        ),
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=sorted(DIRECTIONS),
        default=list(DEFAULT_DIRECTIONS),
        help=(
            "Ground-truth direction names. Defaults synthesize classical backend measurements."
        ),
    )
    parser.add_argument("--strengths", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", default=None)
    parser.add_argument("--pretrain-iter", type=int, default=400)
    parser.add_argument("--num-iter", type=int, default=1000)
    parser.add_argument("--sys-na", type=float, default=0.45)
    parser.add_argument("--lamb", type=float, default=0.55e-6)
    parser.add_argument("--lr-obj", type=float, default=5e-3)
    parser.add_argument("--lr-seidel", type=float, default=1e-2)
    parser.add_argument("--rsd-weight", type=float, default=5e-4)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--pretrain-scalar", type=float, default=5.0)
    parser.add_argument("--defocus-anchor-weight", type=float, default=1.0)
    parser.add_argument("--backend6-defocus-anchor-weight", type=float, default=0.0)
    parser.add_argument("--defocus-index", type=int, default=5)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--eta-min-ratio", type=float, default=1.0 / 25.0)
    parser.add_argument("--max-val", type=float, default=40.0)
    parser.add_argument("--nerf-beta", type=float, default=1.0)
    parser.add_argument("--nerf-depth", type=int, default=6)
    parser.add_argument("--nerf-width", type=int, default=128)
    parser.add_argument(
        "--nerf-skips",
        default="2,4,6",
        help="Comma-separated NeuralObject2D skip indices, or 'none'.",
    )
    parser.add_argument("--output-mode", choices=["softplus", "sigmoid"], default="softplus")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--train-verbose", action="store_true")
    return cocoa.normalize_nerf_capacity_args(parser, parser.parse_args(argv))


def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is false")
    run_sweep(args)


if __name__ == "__main__":
    main()
