#!/usr/bin/env python3
"""Run 4D/6D oracle controls for direct no-RMS Seidel recovery."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_cocoa_like_2d_mechanism as cocoa  # noqa: E402
import run_cocoa_like_seidel_accuracy_sweep as accuracy  # noqa: E402


IMAGES = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]
DIRECTIONS = ["cocoa_signed", "signed_balanced"]
STRENGTHS = [0.06, 0.20, 0.40]
DIMENSIONS = ["classical4d", "classical6d"]
ORACLE_MODES = ["joint_no_RMS", "seidel_gt_fixed", "object_gt_fixed"]
MODE_TO_RUNNER_MODE = {
    "joint_no_RMS": "joint",
    "seidel_gt_fixed": "seidel_gt_fixed",
    "object_gt_fixed": "object_gt_fixed",
}


def tag_float(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def rms_label(value: float) -> str:
    return f"rms{float(value):.2f}".replace(".", "p")


def safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def parse_float_list(values: list[str] | None, default: list[float]) -> list[float]:
    if values is None:
        return list(default)
    return [float(value) for value in values]


def shard_items(items: list[Any], *, shard_index: int, num_shards: int) -> list[Any]:
    if num_shards <= 1:
        return list(items)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_index]


def rel_to_output_base(path: Path) -> str:
    base = cocoa.PROJECT_ROOT / "outputs" / "cocoa_like_2d_mechanism"
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def case_parent_dir(
    output_root: Path,
    *,
    seidel_convention: str,
    seed: int,
    image: str,
    candidate_id: str,
    oracle_mode: str,
) -> Path:
    return (
        output_root
        / seidel_convention
        / f"seed{int(seed)}"
        / f"{safe_name(image)}__{safe_name(candidate_id)}"
        / oracle_mode
    )


def case_metrics_path(
    output_root: Path,
    *,
    seidel_convention: str,
    seed: int,
    image: str,
    candidate_id: str,
    oracle_mode: str,
) -> Path:
    runner_mode = MODE_TO_RUNNER_MODE[oracle_mode]
    return (
        case_parent_dir(
            output_root,
            seidel_convention=seidel_convention,
            seed=seed,
            image=image,
            candidate_id=candidate_id,
            oracle_mode=oracle_mode,
        )
        / runner_mode
        / "metrics.json"
    )


def run_args_for_case(
    *,
    args: argparse.Namespace,
    image: str,
    candidate: accuracy.Candidate,
    seidel_convention: str,
    seed: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        image=image,
        size=args.size,
        modes=[],
        run_name=None,
        num_iter=args.num_iter,
        pretrain_iter=args.pretrain_iter,
        lr_obj=args.lr_obj,
        lr_seidel=args.lr_seidel,
        rsd_weight=args.rsd_weight,
        tv_weight=args.tv_weight,
        pretrain_scalar=args.pretrain_scalar,
        defocus_anchor_weight=args.defocus_anchor_weight,
        defocus_index=args.defocus_index,
        seidel_parameterization="direct",
        seidel_rms_prior_mode="floor",
        seidel_rms_prior_measure="wavefront",
        seidel_rms_floor_weight=0.0,
        seidel_rms_floor_alpha=0.8,
        seidel_rms_floor_target=None,
        target_wavefront_rms=candidate.actual_rms,
        target_coeff_rms=candidate.coeff_rms,
        seidel_rms_floor_field_samples=args.seidel_rms_floor_field_samples,
        seidel_rms_floor_pupil_samples=args.seidel_rms_floor_pupil_samples,
        gt_fixed_seidel_indices=[],
        seidel_lr_multipliers=None,
        scheduler=args.scheduler,
        eta_min_ratio=args.eta_min_ratio,
        max_val=args.max_val,
        nerf_beta=args.nerf_beta,
        output_mode=args.output_mode,
        nerf_depth=args.nerf_depth,
        nerf_width=args.nerf_width,
        nerf_skips=args.nerf_skips,
        fourier_num_angles=args.fourier_num_angles,
        fourier_num_octaves=args.fourier_num_octaves,
        seidel_convention=seidel_convention,
        gt_preset="custom",
        gt_seidel_json=json.dumps(candidate.seidel.astype(float).tolist()),
        gt_label=candidate.candidate_id,
        gt_source="custom",
        seed=seed,
        verbose=args.train_verbose,
    )


def augment_metrics(
    metrics: dict[str, Any],
    *,
    output_root: Path,
    metrics_path: Path,
    seidel_convention: str,
    oracle_mode: str,
    runner_mode: str,
    image: str,
    candidate: accuracy.Candidate,
    seed: int,
) -> dict[str, Any]:
    gt = np.asarray(metrics["seidel_gt"], dtype=np.float64)
    rec = np.asarray(metrics["seidel_final"], dtype=np.float64)
    gt_rms = accuracy.field_weighted_wavefront_rms(gt)
    rec_rms = accuracy.field_weighted_wavefront_rms(rec)
    err_rms = accuracy.field_weighted_wavefront_rms(rec - gt)
    gt_coeff_rms = cocoa.seidel_coefficient_rms_np(gt)
    rec_coeff_rms = cocoa.seidel_coefficient_rms_np(rec)
    metrics.update(
        {
            "stage": "oracle_controls",
            "run_root": str(output_root),
            "metrics_path": rel_to_output_base(metrics_path),
            "image": image,
            "seed": int(seed),
            "candidate_id": candidate.candidate_id,
            "direction": candidate.direction,
            "target_wavefront_rms": float(candidate.target_rms),
            "actual_wavefront_rms": float(candidate.actual_rms),
            "target_coeff_rms": float(candidate.coeff_rms),
            "wavefront_gt_rms": float(gt_rms),
            "wavefront_recovered_rms": float(rec_rms),
            "wavefront_recovered_over_gt_rms": float(rec_rms / max(gt_rms, 1e-12)),
            "wavefront_error_rms": float(err_rms),
            "relative_wavefront_error": float(err_rms / max(gt_rms, 1e-12)),
            "coeff_gt_rms": float(gt_coeff_rms),
            "coeff_recovered_rms": float(rec_coeff_rms),
            "coeff_recovered_over_gt_rms": float(rec_coeff_rms / max(gt_coeff_rms, 1e-12)),
            "seidel_convention": seidel_convention,
            "dimension": "4D" if seidel_convention == "classical4d" else "6D",
            "oracle_mode": oracle_mode,
            "runner_mode": runner_mode,
            "loss_family": "direct_no_RMS_original",
            "lambda": 0.0,
            **cocoa.convention_metadata(seidel_convention),
        }
    )
    config = dict(metrics.get("config", {}))
    config.update(
        {
            "oracle_mode": oracle_mode,
            "runner_mode": runner_mode,
            "loss_family": "direct_no_RMS_original",
        }
    )
    metrics["config"] = config
    return metrics


def run_case(
    *,
    output_root: Path,
    args: argparse.Namespace,
    seidel_convention: str,
    image: str,
    candidate: accuracy.Candidate,
    seed: int,
    oracle_mode: str,
    force: bool,
) -> dict[str, Any]:
    runner_mode = MODE_TO_RUNNER_MODE[oracle_mode]
    metrics_path = case_metrics_path(
        output_root,
        seidel_convention=seidel_convention,
        seed=seed,
        image=image,
        candidate_id=candidate.candidate_id,
        oracle_mode=oracle_mode,
    )
    if metrics_path.is_file() and not force:
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("oracle_mode") != oracle_mode or "metrics_path" not in metrics:
            metrics = augment_metrics(
                metrics,
                output_root=output_root,
                metrics_path=metrics_path,
                seidel_convention=seidel_convention,
                oracle_mode=oracle_mode,
                runner_mode=runner_mode,
                image=image,
                candidate=candidate,
                seed=seed,
            )
            metrics_path.write_text(json.dumps(metrics, indent=2))
        print(
            f"[skip] {seidel_convention} seed={seed} {image} {candidate.candidate_id} {oracle_mode}",
            flush=True,
        )
        return metrics

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_args = run_args_for_case(
        args=args,
        image=image,
        candidate=candidate,
        seidel_convention=seidel_convention,
        seed=seed,
    )
    gt_vec = torch.tensor(candidate.seidel, device=device, dtype=torch.float32)
    sharp_gt = cocoa.load_baboon_gt(args.size, path=cocoa.IMAGE_PATHS[image], device=device)
    meas_gt = cocoa.synthesize_measurement(sharp_gt, gt_vec, cocoa.SYS_PARAMS)
    case_root = case_parent_dir(
        output_root,
        seidel_convention=seidel_convention,
        seed=seed,
        image=image,
        candidate_id=candidate.candidate_id,
        oracle_mode=oracle_mode,
    )
    case_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[case] {seidel_convention} seed={seed} image={image} "
        f"candidate={candidate.candidate_id} oracle={oracle_mode} "
        f"size={args.size} pre={args.pretrain_iter} joint={args.num_iter}",
        flush=True,
    )
    result, metrics = cocoa.run_one_mode(
        run_args,
        mode=runner_mode,
        sharp_gt=sharp_gt,
        meas_gt=meas_gt,
        gt_vec=gt_vec,
        gt_np=candidate.seidel,
        root_dir=case_root,
        device=device,
    )
    metrics = augment_metrics(
        metrics,
        output_root=output_root,
        metrics_path=metrics_path,
        seidel_convention=seidel_convention,
        oracle_mode=oracle_mode,
        runner_mode=runner_mode,
        image=image,
        candidate=candidate,
        seed=seed,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2))
    cocoa.save_summary_figure(case_root, sharp_gt, meas_gt, [(oracle_mode, result, metrics)])
    (case_root / "summary.json").write_text(
        json.dumps(
            {
                "seidel_convention": seidel_convention,
                "oracle_mode": oracle_mode,
                "runner_mode": runner_mode,
                "image": image,
                "seed": int(seed),
                "candidate": {
                    "candidate_id": candidate.candidate_id,
                    "direction": candidate.direction,
                    "target_rms": candidate.target_rms,
                    "actual_rms": candidate.actual_rms,
                    "coeff_rms": candidate.coeff_rms,
                    "seidel": candidate.seidel.tolist(),
                },
                "metrics_path": rel_to_output_base(metrics_path),
            },
            indent=2,
        )
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    preferred = [
        "stage",
        "run_root",
        "metrics_path",
        "seidel_convention",
        "dimension",
        "oracle_mode",
        "runner_mode",
        "loss_family",
        "image",
        "seed",
        "candidate_id",
        "direction",
        "target_wavefront_rms",
        "actual_wavefront_rms",
        "target_coeff_rms",
        "wavefront_gt_rms",
        "wavefront_recovered_rms",
        "wavefront_recovered_over_gt_rms",
        "coeff_gt_rms",
        "coeff_recovered_rms",
        "coeff_recovered_over_gt_rms",
        "relative_wavefront_error",
        "wavefront_error_rms",
        "operator_error_calibrated",
        "operator_error_phys_equiv",
        "operator_error_coord_diagnostic",
        "best_physical_transform",
        "ssim_recon_gain_vs_gt",
        "nrmse_recon_gain_vs_gt",
        "ssim_meas_pred_vs_meas",
        "nrmse_meas_pred_vs_meas",
        "l2_seidel_vs_gt",
        "seidel_parameterization",
        "seidel_rms_floor_weight",
        "fixed_seidel_indices",
        "no_defocus",
        "no_w311_no_defocus",
        "seidel_gt",
        "seidel_final",
    ]
    extra = []
    seen = set(preferred)
    for row in rows:
        for key, value in row.items():
            if key in seen or isinstance(value, (dict, list, tuple)):
                continue
            extra.append(key)
            seen.add(key)
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + sorted(extra)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, separators=(",", ":"))
                out[key] = value
            writer.writerow(out)


def build_all_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    strengths = parse_float_list(args.strengths, STRENGTHS)
    for seidel_convention in args.dimensions:
        candidates = accuracy.make_candidates(
            args.directions,
            strengths,
            seidel_convention=seidel_convention,
        )
        for seed in args.seeds:
            for image in args.images:
                for candidate in candidates:
                    for oracle_mode in args.oracle_modes:
                        cases.append(
                            {
                                "seidel_convention": seidel_convention,
                                "seed": int(seed),
                                "image": image,
                                "candidate": candidate,
                                "oracle_mode": oracle_mode,
                            }
                        )
    return cases


def collect_expected_metrics(output_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in build_all_cases(args):
        path = case_metrics_path(
            output_root,
            seidel_convention=case["seidel_convention"],
            seed=case["seed"],
            image=case["image"],
            candidate_id=case["candidate"].candidate_id,
            oracle_mode=case["oracle_mode"],
        )
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text())
        rows.append(
            augment_metrics(
                metrics,
                output_root=output_root,
                metrics_path=path,
                seidel_convention=case["seidel_convention"],
                oracle_mode=case["oracle_mode"],
                runner_mode=MODE_TO_RUNNER_MODE[case["oracle_mode"]],
                image=case["image"],
                candidate=case["candidate"],
                seed=case["seed"],
            )
        )
    return sort_rows(rows)


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image_order = {name: idx for idx, name in enumerate(IMAGES)}
    direction_order = {name: idx for idx, name in enumerate(DIRECTIONS)}
    mode_order = {name: idx for idx, name in enumerate(ORACLE_MODES)}
    dim_order = {name: idx for idx, name in enumerate(DIMENSIONS)}
    return sorted(
        rows,
        key=lambda row: (
            dim_order.get(str(row.get("seidel_convention")), 99),
            int(row.get("seed", 0)),
            image_order.get(str(row.get("image")), 99),
            direction_order.get(str(row.get("direction")), 99),
            float(row.get("target_wavefront_rms", 0.0)),
            mode_order.get(str(row.get("oracle_mode")), 99),
        ),
    )


def write_report_csvs(output_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = collect_expected_metrics(output_root, args)
    write_csv(rows, output_root / "oracle_controls_operator_input.csv")
    for seidel_convention in args.dimensions:
        dim_rows = [row for row in rows if row.get("seidel_convention") == seidel_convention]
        write_csv(dim_rows, output_root / f"oracle_controls_{seidel_convention}_operator_input.csv")
    expected = len(build_all_cases(args))
    summary = {
        "run_name": args.run_name,
        "output_root": str(output_root),
        "expected_cases": expected,
        "completed_cases": len(rows),
        "missing_cases": max(0, expected - len(rows)),
        "dimensions": args.dimensions,
        "images": args.images,
        "directions": args.directions,
        "strengths": parse_float_list(args.strengths, STRENGTHS),
        "seeds": args.seeds,
        "oracle_modes": args.oracle_modes,
        "loss_family": "direct_no_RMS_original",
    }
    (output_root / "run_status.json").write_text(json.dumps(summary, indent=2))
    print(f"[report] completed {len(rows)}/{expected} cases", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--images", nargs="+", choices=sorted(cocoa.IMAGE_PATHS), default=IMAGES)
    parser.add_argument("--directions", nargs="+", choices=sorted(accuracy.DIRECTIONS), default=DIRECTIONS)
    parser.add_argument("--strengths", nargs="+", default=[str(v) for v in STRENGTHS])
    parser.add_argument("--dimensions", nargs="+", choices=list(cocoa.CLASSICAL_CONVENTIONS), default=DIMENSIONS)
    parser.add_argument("--oracle-modes", nargs="+", choices=ORACLE_MODES, default=ORACLE_MODES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--pretrain-iter", type=int, default=400)
    parser.add_argument("--num-iter", type=int, default=1000)
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
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train-verbose", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--skip-config-write", action="store_true")
    args = parser.parse_args()
    args.scheduler = None if args.scheduler == "none" else args.scheduler
    args.nerf_skips = cocoa.parse_nerf_skips(args.nerf_skips)
    args.strengths = [str(value) for value in parse_float_list(args.strengths, STRENGTHS)]
    args.dimensions = list(args.dimensions)
    args.oracle_modes = list(args.oracle_modes)
    args.directions = list(args.directions)
    args.images = list(args.images)
    args.seeds = [int(seed) for seed in args.seeds]
    if args.run_name is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"seidel_oracle_controls_4D_6D_noRMS_{stamp}"
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    return args


def write_run_config(output_root: Path, args: argparse.Namespace) -> None:
    strengths = parse_float_list(args.strengths, STRENGTHS)
    candidates_by_dim = {
        convention: [
            {
                "candidate_id": candidate.candidate_id,
                "direction": candidate.direction,
                "target_rms": candidate.target_rms,
                "actual_rms": candidate.actual_rms,
                "coeff_rms": candidate.coeff_rms,
                "seidel": candidate.seidel.tolist(),
            }
            for candidate in accuracy.make_candidates(
                args.directions,
                strengths,
                seidel_convention=convention,
            )
        ]
        for convention in args.dimensions
    }
    (output_root / "oracle_controls_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "loss_family": "direct_no_RMS_original",
                "mode_mapping": MODE_TO_RUNNER_MODE,
                "candidates_by_dimension": candidates_by_dim,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    output_root = cocoa.PROJECT_ROOT / "outputs" / "cocoa_like_2d_mechanism" / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_config_write:
        write_run_config(output_root, args)

    all_cases = build_all_cases(args)
    selected_cases = shard_items(all_cases, shard_index=args.shard_index, num_shards=args.num_shards)
    print(
        f"[start] run={args.run_name} shard={args.shard_index}/{args.num_shards} "
        f"selected_cases={len(selected_cases)} total_cases={len(all_cases)}",
        flush=True,
    )
    if not args.report_only:
        for case in selected_cases:
            run_case(
                output_root=output_root,
                args=args,
                seidel_convention=case["seidel_convention"],
                image=case["image"],
                candidate=case["candidate"],
                seed=case["seed"],
                oracle_mode=case["oracle_mode"],
                force=args.force,
            )
    rows = write_report_csvs(output_root, args)
    print(f"[done] wrote reports for {len(rows)} completed cases under {output_root}", flush=True)


if __name__ == "__main__":
    main()
