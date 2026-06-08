#!/usr/bin/env python3
"""Run a small seed-sensitivity test for Seidel recovery parameterizations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_cocoa_like_2d_mechanism as cocoa  # noqa: E402
from run_cocoa_like_seidel_accuracy_sweep import (  # noqa: E402
    Candidate,
    DIRECTIONS,
    field_weighted_wavefront_rms as candidate_wavefront_rms,
    make_candidates,
    run_case,
    tag_float,
)

from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import (  # noqa: E402
    field_weighted_wavefront_rms,
)


COEFF_LABELS = ["W040", "W131", "W222", "W220", "W311", "Wd"]


def zero_fixed_gt_without_rescale(candidate: Candidate, fixed_indices: list[int]) -> Candidate:
    """Zero recovery-fixed GT coefficients after canonical scaling, without renormalizing."""

    if not fixed_indices:
        return candidate
    seidel = np.asarray(candidate.seidel, dtype=np.float32).copy()
    seidel[fixed_indices] = 0.0
    return Candidate(
        candidate_id=candidate.candidate_id,
        direction=candidate.direction,
        target_rms=candidate.target_rms,
        seidel=seidel,
        actual_rms=field_weighted_wavefront_rms(seidel),
        coeff_rms=cocoa.seidel_coefficient_rms_np(seidel),
    )


def make_base_override_candidates(
    direction: str,
    strengths: list[float],
    *,
    seidel_convention: str,
    w311_base_value: float | None,
) -> list[Candidate]:
    if w311_base_value is None:
        return make_candidates([direction], strengths, seidel_convention=seidel_convention)

    fixed = cocoa.fixed_seidel_indices_for_convention(seidel_convention)
    base = np.asarray(DIRECTIONS[direction], dtype=np.float64).copy()
    base[4] = float(w311_base_value)
    if fixed:
        base[fixed] = 0.0
    base_rms = candidate_wavefront_rms(base)
    if base_rms <= 1e-12:
        raise ValueError(f"Direction {direction} with W311 override has near-zero wavefront RMS")
    direction_label = f"{direction}_W311base{tag_float(float(w311_base_value))}"
    candidates: list[Candidate] = []
    for target in strengths:
        seidel = base * (target / base_rms)
        actual = candidate_wavefront_rms(seidel)
        coeff_rms = cocoa.seidel_coefficient_rms_np(seidel)
        candidates.append(
            Candidate(
                candidate_id=f"{direction_label}__rms{tag_float(target)}",
                direction=direction_label,
                target_rms=float(target),
                seidel=seidel.astype(np.float32),
                actual_rms=actual,
                coeff_rms=coeff_rms,
            )
        )
    return candidates


def dump_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return value


def parse_vector(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=np.float64).reshape(6)


def make_sweep_args(
    args: argparse.Namespace,
    parameterization: str,
    rms_prior_measure: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        lr_obj=args.lr_obj,
        lr_seidel=args.lr_seidel,
        rsd_weight=args.rsd_weight,
        tv_weight=args.tv_weight,
        pretrain_scalar=args.pretrain_scalar,
        defocus_anchor_weight=args.defocus_anchor_weight,
        defocus_index=args.defocus_index,
        seidel_parameterization=parameterization,
        seidel_rms_prior_mode="ratio_target",
        seidel_rms_prior_measure=rms_prior_measure,
        seidel_rms_floor_weight=args.lambda_a,
        seidel_rms_floor_alpha=args.alpha,
        seidel_rms_floor_field_samples=args.seidel_rms_floor_field_samples,
        seidel_rms_floor_pupil_samples=args.seidel_rms_floor_pupil_samples,
        gt_fixed_seidel_indices=[],
        seidel_lr_multipliers=None,
        scheduler=None if args.scheduler == "none" else args.scheduler,
        eta_min_ratio=args.eta_min_ratio,
        max_val=args.max_val,
        nerf_beta=args.nerf_beta,
        output_mode=args.output_mode,
        nerf_depth=args.nerf_depth,
        nerf_width=args.nerf_width,
        nerf_skips=cocoa.parse_nerf_skips(args.nerf_skips),
        fourier_num_angles=args.fourier_num_angles,
        fourier_num_octaves=args.fourier_num_octaves,
    )


def build_eval_input(output_root: Path, run_name: str, out_csv: Path) -> list[dict[str, Any]]:
    run_root = output_root / run_name
    paths = sorted((run_root / "stage3").glob("seed*/*/joint/metrics.json"))
    rows: list[dict[str, Any]] = []
    for metrics_path in paths:
        metrics = json.loads(metrics_path.read_text())
        config = metrics.get("config", {})
        row = {
            "profile": metrics.get("candidate_id", metrics_path.parents[1].name),
            "candidate_id": metrics.get("candidate_id", metrics_path.parents[1].name),
            "seed": metrics.get("seed"),
            "lambda": metrics.get("seidel_rms_floor_weight", config.get("seidel_rms_floor_weight")),
            "alpha": metrics.get("seidel_rms_floor_alpha", config.get("seidel_rms_floor_alpha")),
            "target_wavefront_rms": metrics.get("target_wavefront_rms"),
            "target_coeff_rms": metrics.get("target_coeff_rms"),
            "direction": metrics.get("direction"),
            "pretrain_iter": config.get("pretrain_iter"),
            "pretrain_scalar": config.get("pretrain_scalar"),
            "image": metrics.get("image"),
            "size": metrics.get("size"),
            "seidel_convention": metrics.get("seidel_convention", config.get("seidel_convention")),
            "seidel_rms_prior_mode": metrics.get(
                "seidel_rms_prior_mode",
                config.get("seidel_rms_prior_mode"),
            ),
            "seidel_rms_prior_measure": metrics.get(
                "seidel_rms_prior_measure",
                config.get("seidel_rms_prior_measure", "wavefront"),
            ),
            "seidel_parameterization": metrics.get(
                "seidel_parameterization",
                config.get("seidel_parameterization"),
            ),
            "seidel_amplitude_final": metrics.get("seidel_amplitude_final"),
            "seidel_direction_rms_final": metrics.get("seidel_direction_rms_final"),
            "fixed_seidel_indices": metrics.get("fixed_seidel_indices", config.get("fixed_seidel_indices", [])),
            "seidel_gt": metrics.get("seidel_gt"),
            "seidel_final": metrics.get("seidel_final"),
            "ssim_recon_gain_vs_gt": metrics.get("ssim_recon_gain_vs_gt"),
            "nrmse_recon_gain_vs_gt": metrics.get("nrmse_recon_gain_vs_gt"),
            "nrmse_meas_pred_vs_meas": metrics.get("nrmse_meas_pred_vs_meas"),
            "recon_raw_hf_ratio": metrics.get("recon_raw_hf_ratio"),
            "l2_seidel_vs_gt": metrics.get("l2_seidel_vs_gt"),
            "wavefront_gt_rms": metrics.get("wavefront_gt_rms"),
            "wavefront_recovered_rms": metrics.get("wavefront_recovered_rms"),
            "wavefront_recovered_over_gt_rms": metrics.get("wavefront_recovered_over_gt_rms"),
            "coeff_gt_rms": metrics.get("coeff_gt_rms"),
            "coeff_recovered_rms": metrics.get("coeff_recovered_rms"),
            "coeff_recovered_over_gt_rms": metrics.get("coeff_recovered_over_gt_rms"),
            "relative_wavefront_error": metrics.get("relative_wavefront_error"),
            "final_seidel_rms_floor_loss": metrics.get("final_seidel_rms_floor_loss"),
            "final_seidel_wavefront_rms_floor_estimate": metrics.get("final_seidel_wavefront_rms_floor_estimate"),
            "final_seidel_wavefront_rms_estimate": metrics.get("final_seidel_wavefront_rms_estimate"),
            "final_seidel_coeff_rms_estimate": metrics.get("final_seidel_coeff_rms_estimate"),
            "nerf_depth": config.get("nerf_depth"),
            "nerf_width": config.get("nerf_width"),
            "nerf_skips": config.get("nerf_skips"),
            "fourier_num_angles": config.get("fourier_num_angles"),
            "fourier_num_octaves": config.get("fourier_num_octaves"),
            "output_mode": config.get("output_mode"),
            "max_val": config.get("max_val"),
            "rsd_weight": config.get("rsd_weight"),
            "nerf_beta": config.get("nerf_beta"),
            "metrics_path": str(metrics_path.relative_to(output_root)),
            "run_root": run_name,
        }
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No metrics found under {run_root / 'stage3'}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: dump_value(row.get(key, "")) for key in fieldnames})
    return rows


def write_raw_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        theta = parse_vector(row["seidel_final"])
        abs_theta = np.abs(theta)
        denom = float(abs_theta.sum())
        gt = parse_vector(row["seidel_gt"])
        summary = {
            "method": row["seidel_parameterization"],
            "seidel_rms_prior_measure": row.get("seidel_rms_prior_measure", "wavefront"),
            "seed": row["seed"],
            "image": row["image"],
            "target_wavefront_rms": row["target_wavefront_rms"],
            "target_coeff_rms": row.get("target_coeff_rms"),
            "lambda": row["lambda"],
            "operator_error_calibrated": "",
            "raw_recovered_over_gt_rms": field_weighted_wavefront_rms(theta) / max(field_weighted_wavefront_rms(gt), 1e-12),
            "raw_coeff_recovered_over_gt_rms": cocoa.seidel_coefficient_rms_np(theta) / max(cocoa.seidel_coefficient_rms_np(gt), 1e-12),
            "raw_w222_abs_share": float(abs_theta[2] / denom) if denom > 0 else math.nan,
            "raw_abs_coeff_cv": float(abs_theta.std(ddof=0) / abs_theta.mean()) if abs_theta.mean() > 0 else math.nan,
        }
        for idx, label in enumerate(COEFF_LABELS):
            summary[f"raw_{label}"] = float(theta[idx])
        summary_rows.append(summary)

    fieldnames: list[str] = []
    for row in summary_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cocoa_like_2d_mechanism"))
    parser.add_argument("--image", default="dendrites", choices=sorted(cocoa.IMAGE_PATHS))
    parser.add_argument("--direction", default="signed_balanced")
    parser.add_argument(
        "--seidel-convention",
        choices=list(cocoa.CLASSICAL_CONVENTIONS),
        default="classical6d",
        help=(
            "Seidel convention for direct recovery. Use classical5d for no defocus "
            "and classical4d for no W311/no defocus. amp_direction variants require 6D."
        ),
    )
    parser.add_argument(
        "--gt-seidel-convention",
        choices=list(cocoa.CLASSICAL_CONVENTIONS),
        default="classical6d",
        help=(
            "Convention used only to generate the GT Seidel vector and measurement. "
            "Keep this at classical6d for 4D/5D/6D recovery ablations with identical GT coefficients."
        ),
    )
    parser.add_argument(
        "--zero-fixed-gt-coefficients-without-rescale",
        action="store_true",
        help=(
            "After building the canonical GT coefficients from --gt-seidel-convention, "
            "zero coefficients fixed by --seidel-convention without renormalizing. "
            "This keeps active coefficients identical across 4D/5D/6D while removing "
            "Wd and/or W311 from lower-dimensional measurements."
        ),
    )
    parser.add_argument(
        "--gt-w311-base-value",
        type=float,
        default=None,
        help=(
            "Optional override for the W311 entry of the canonical direction before "
            "target-RMS scaling. For example, 0.08 makes signed_balanced include "
            "nonzero W311 in 5D/6D while 4D can zero it."
        ),
    )
    parser.add_argument("--target-rms", type=float, default=0.20)
    parser.add_argument(
        "--target-rms-values",
        nargs="+",
        type=float,
        default=None,
        help="Optional list of target wavefront RMS levels. Defaults to --target-rms.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--parameterizations", nargs="+", default=["amp_direction", "amp_direction_detach_norm"])
    parser.add_argument(
        "--seidel-rms-prior-measures",
        nargs="+",
        choices=cocoa.SEIDEL_RMS_PRIOR_MEASURES,
        default=["wavefront"],
        help="RMS measure(s) used by the strength prior and amp-direction normalization.",
    )
    parser.add_argument("--lambda-a", type=float, default=1000.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--pretrain-iter", type=int, default=400)
    parser.add_argument("--num-iter", type=int, default=1000)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--lr-obj", type=float, default=5e-3)
    parser.add_argument("--lr-seidel", type=float, default=1e-2)
    parser.add_argument("--rsd-weight", type=float, default=5e-4)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--pretrain-scalar", type=float, default=5.0)
    parser.add_argument("--defocus-anchor-weight", type=float, default=1.0)
    parser.add_argument("--defocus-index", type=int, default=5)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--eta-min-ratio", type=float, default=1.0 / 25.0)
    parser.add_argument("--max-val", type=float, default=40.0)
    parser.add_argument("--nerf-beta", type=float, default=1.0)
    parser.add_argument("--output-mode", choices=["softplus", "sigmoid"], default="softplus")
    parser.add_argument("--nerf-depth", type=int, default=6)
    parser.add_argument("--nerf-width", type=int, default=128)
    parser.add_argument("--nerf-skips", default="2,4,6")
    parser.add_argument("--fourier-num-angles", type=int, default=60)
    parser.add_argument("--fourier-num-octaves", type=int, default=7)
    parser.add_argument("--seidel-rms-floor-field-samples", type=int, default=21)
    parser.add_argument("--seidel-rms-floor-pupil-samples", type=int, default=51)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train-verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    supported_parameterizations = {"direct", "amp_direction", "amp_direction_detach_norm"}
    for parameterization in args.parameterizations:
        if parameterization not in supported_parameterizations:
            raise ValueError(f"Unsupported parameterization for this test: {parameterization}")
    target_rms_values = list(args.target_rms_values) if args.target_rms_values is not None else [float(args.target_rms)]

    if any(parameterization != "direct" for parameterization in args.parameterizations):
        if args.seidel_convention not in {"classical6d", "backend6"}:
            raise ValueError("--parameterizations amp_direction* require --seidel-convention classical6d/backend6")

    candidates = make_base_override_candidates(
        args.direction,
        target_rms_values,
        seidel_convention=args.gt_seidel_convention,
        w311_base_value=args.gt_w311_base_value,
    )
    if args.zero_fixed_gt_coefficients_without_rescale:
        fixed_for_recovery = cocoa.fixed_seidel_indices_for_convention(args.seidel_convention)
        candidates = [
            zero_fixed_gt_without_rescale(candidate, fixed_for_recovery)
            for candidate in candidates
        ]

    all_cases: list[tuple[int, str, str, Candidate]] = []
    for seed in args.seeds:
        for candidate in candidates:
            for rms_prior_measure in args.seidel_rms_prior_measures:
                for parameterization in args.parameterizations:
                    all_cases.append((int(seed), parameterization, rms_prior_measure, candidate))
    selected_cases = [case for idx, case in enumerate(all_cases) if idx % args.num_shards == args.shard_index]
    print(
        f"[seed-test] run={args.run_name} shard={args.shard_index}/{args.num_shards} "
        f"cases={len(selected_cases)}/{len(all_cases)} image={args.image} "
        f"seidel_convention={args.seidel_convention} gt_seidel_convention={args.gt_seidel_convention} "
        f"zero_fixed_gt_without_rescale={args.zero_fixed_gt_coefficients_without_rescale} "
        f"gt_w311_base_value={args.gt_w311_base_value} "
        f"direction={args.direction} rms={','.join(f'{v:g}' for v in target_rms_values)} "
        f"prior_measures={','.join(args.seidel_rms_prior_measures)}",
        flush=True,
    )

    for seed, parameterization, rms_prior_measure, base_candidate in selected_cases:
        if float(args.lambda_a) == 0.0:
            suffix = f"no_RMS__{parameterization}"
        else:
            suffix = f"{rms_prior_measure}_RMS_prior__{parameterization}"
        candidate = replace(
            base_candidate,
            candidate_id=f"{base_candidate.candidate_id}__{suffix}",
        )
        run_case(
            output_root=args.output_root / args.run_name,
            stage="stage3",
            image=args.image,
            candidate=candidate,
            size=args.size,
            pretrain_iter=args.pretrain_iter,
            num_iter=args.num_iter,
            seed=seed,
            force=args.force,
            train_verbose=args.train_verbose,
            seidel_convention=args.seidel_convention,
            sweep_args=make_sweep_args(args, parameterization, rms_prior_measure),
        )

    rows = build_eval_input(
        args.output_root,
        args.run_name,
        args.output_root / f"{args.run_name}_operator_input.csv",
    )
    write_raw_summary(rows, args.output_root / args.run_name / "raw_seed_summary.csv")
    print(f"[seed-test] wrote input rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
