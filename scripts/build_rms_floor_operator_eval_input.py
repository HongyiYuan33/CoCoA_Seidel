"""Build operator-evaluator input CSVs from RMS-floor sweep metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PREFERRED_FIELDS = [
    "profile",
    "candidate_id",
    "lambda",
    "alpha",
    "target_wavefront_rms",
    "target_coeff_rms",
    "direction",
    "pretrain_iter",
    "pretrain_scalar",
    "image",
    "size",
    "seidel_convention",
    "seidel_rms_prior_mode",
    "seidel_rms_prior_measure",
    "seidel_parameterization",
    "seidel_amplitude_final",
    "seidel_direction_rms_final",
    "fixed_seidel_indices",
    "gt_fixed_seidel_indices",
    "gt_fixed_seidel_values",
    "seidel_lr_multipliers",
    "seidel_gt",
    "seidel_final",
    "ssim_recon_gain_vs_gt",
    "nrmse_recon_gain_vs_gt",
    "nrmse_meas_pred_vs_meas",
    "recon_raw_hf_ratio",
    "l2_seidel_vs_gt",
    "wavefront_gt_rms",
    "wavefront_recovered_rms",
    "wavefront_recovered_over_gt_rms",
    "coeff_gt_rms",
    "coeff_recovered_rms",
    "coeff_recovered_over_gt_rms",
    "relative_wavefront_error",
    "final_seidel_rms_floor_loss",
    "final_seidel_wavefront_rms_floor_estimate",
    "final_seidel_wavefront_rms_estimate",
    "final_seidel_coeff_rms_estimate",
    "nerf_depth",
    "nerf_width",
    "nerf_skips",
    "fourier_num_angles",
    "fourier_num_octaves",
    "output_mode",
    "max_val",
    "rsd_weight",
    "nerf_beta",
    "metrics_path",
    "run_root",
]


def dump_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return value


def row_from_metrics(metrics_path: Path, output_root: Path) -> dict[str, Any]:
    with metrics_path.open() as f:
        metrics = json.load(f)
    config = metrics.get("config", {})
    case_key = metrics_path.parents[1].name
    run_root = metrics_path.parents[3]
    weight = float(metrics.get("seidel_rms_floor_weight", config.get("seidel_rms_floor_weight", 0.0)))
    alpha = float(metrics.get("seidel_rms_floor_alpha", config.get("seidel_rms_floor_alpha", 0.8)))
    profile = f"lambda{weight:g}__{case_key}"
    row = {
        "profile": profile,
        "candidate_id": profile,
        "lambda": weight,
        "alpha": alpha,
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
            config.get("seidel_rms_prior_mode", "floor"),
        ),
        "seidel_rms_prior_measure": metrics.get(
            "seidel_rms_prior_measure",
            config.get("seidel_rms_prior_measure", "wavefront"),
        ),
        "seidel_parameterization": metrics.get(
            "seidel_parameterization",
            config.get("seidel_parameterization", "direct"),
        ),
        "seidel_amplitude_final": metrics.get("seidel_amplitude_final"),
        "seidel_direction_rms_final": metrics.get("seidel_direction_rms_final"),
        "fixed_seidel_indices": metrics.get("fixed_seidel_indices", config.get("fixed_seidel_indices", [])),
        "gt_fixed_seidel_indices": metrics.get(
            "gt_fixed_seidel_indices",
            config.get("gt_fixed_seidel_indices", []),
        ),
        "gt_fixed_seidel_values": metrics.get("gt_fixed_seidel_values", []),
        "seidel_lr_multipliers": metrics.get(
            "seidel_lr_multipliers",
            config.get("seidel_lr_multipliers", None),
        ),
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
        "run_root": run_root.name,
    }
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cocoa_like_2d_mechanism"))
    parser.add_argument("--run-root", action="append", required=True, help="Sweep run root directory name.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for run_root_name in args.run_root:
        run_root = args.output_root / run_root_name
        if not run_root.is_dir():
            raise FileNotFoundError(run_root)
        paths = sorted((run_root / "stage1").glob("*/joint/metrics.json"))
        for path in paths:
            rows.append(row_from_metrics(path, args.output_root))
    rows.sort(
        key=lambda row: (
            float(row.get("lambda") or 0.0),
            float(row.get("target_wavefront_rms") or 0.0),
            str(row.get("image") or ""),
            str(row.get("run_root") or ""),
        )
    )
    if args.expected is not None and len(rows) != int(args.expected):
        raise RuntimeError(f"Expected {args.expected} rows, found {len(rows)}")

    fieldnames = list(PREFERRED_FIELDS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: dump_value(row.get(key, "")) for key in fieldnames})
    print(f"[build-input] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
