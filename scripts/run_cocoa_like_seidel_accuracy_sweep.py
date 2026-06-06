"""Sweep Seidel wavefront recoverability for the CoCoA-like 2-D runner."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_cocoa_like_2d_mechanism as cocoa  # noqa: E402


IMAGES = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]
STRENGTHS = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
DIRECTIONS = {
    "pos_balanced": [0.20, 0.10, 0.05, 0.05, 0.00, 0.10],
    "signed_balanced": [0.24, -0.08, 0.07, 0.06, 0.00, 0.08],
    "cocoa_signed": [0.30, -0.10, 0.05, 0.08, 0.00, 0.08],
    "coma_dominant": [0.05, 0.20, 0.02, 0.04, 0.00, 0.05],
    "astig_field": [0.08, 0.04, 0.16, 0.10, 0.00, 0.05],
    "spherical_defocus": [0.22, 0.00, 0.00, 0.00, 0.00, 0.12],
    "distortion_mixed": [0.16, 0.06, 0.04, 0.04, 0.08, 0.06],
}
STAGE1 = {"size": 128, "pretrain_iter": 200, "num_iter": 500}
STAGE2 = {"size": 256, "pretrain_iter": 400, "num_iter": 1000}
STAGE3 = {"size": 256, "pretrain_iter": 400, "num_iter": 1000}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    direction: str
    target_rms: float
    seidel: np.ndarray
    actual_rms: float


def tag_float(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_float_list(values: Iterable[str] | None, defaults: list[float]) -> list[float]:
    if values is None:
        return list(defaults)
    return [float(v) for v in values]


def get_pupil_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    X, Y = np.meshgrid(x, x, indexing="xy")
    mask = (X * X + Y * Y) <= 1.0
    return X, Y, mask


def seidel_wavefront(coeffs: np.ndarray, X: np.ndarray, Y: np.ndarray, H: float) -> np.ndarray:
    rho2 = X * X + Y * Y
    return (
        coeffs[0] * rho2**2
        + coeffs[1] * H * rho2 * X
        + coeffs[2] * H**2 * X**2
        + coeffs[3] * H**2 * rho2
        + coeffs[4] * H**3 * X
        + coeffs[5] * rho2
    )


def field_weighted_wavefront_rms(
    coeffs: np.ndarray,
    *,
    field_samples: int = 51,
    pupil_samples: int = 201,
) -> float:
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(6)
    X, Y, mask = get_pupil_grid(pupil_samples)
    hs = np.linspace(0.0, 1.0, field_samples, dtype=np.float64)
    weights = hs.copy()
    weights[0] = 0.0
    rms_values = []
    for H in hs:
        W = seidel_wavefront(coeffs, X, Y, float(H))[mask]
        W = W - float(np.mean(W))
        rms_values.append(math.sqrt(float(np.mean(W * W))))
    rms_values = np.asarray(rms_values, dtype=np.float64)
    denom = float(np.sum(weights))
    if denom <= 0:
        return float(rms_values[-1])
    return float(np.sum(rms_values * weights) / denom)


def make_candidates(
    directions: list[str],
    strengths: list[float],
    *,
    seidel_convention: str = "backend6",
) -> list[Candidate]:
    candidates: list[Candidate] = []
    fixed = cocoa.fixed_seidel_indices_for_convention(seidel_convention)
    for direction in directions:
        base = np.asarray(DIRECTIONS[direction], dtype=np.float64)
        if fixed:
            base = base.copy()
            base[fixed] = 0.0
        base_rms = field_weighted_wavefront_rms(base)
        if base_rms <= 1e-12:
            raise ValueError(f"Direction {direction} has near-zero wavefront RMS")
        for target in strengths:
            seidel = base * (target / base_rms)
            actual = field_weighted_wavefront_rms(seidel)
            candidate_id = f"{direction}__rms{tag_float(target)}"
            candidates.append(
                Candidate(
                    candidate_id=candidate_id,
                    direction=direction,
                    target_rms=float(target),
                    seidel=seidel.astype(np.float32),
                    actual_rms=actual,
                )
            )
    return candidates


def shard_items(items: list, *, shard_index: int, num_shards: int) -> list:
    if num_shards <= 1:
        return list(items)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_index]


def case_metrics_path(
    output_root: Path,
    stage: str,
    image: str,
    candidate_id: str,
    *,
    seed: int = 0,
) -> Path:
    seed_dir = f"seed{seed}" if stage == "stage3" else ""
    case_key = f"{image}__{candidate_id}"
    case_dir = output_root / stage / seed_dir / case_key if seed_dir else output_root / stage / case_key
    return case_dir / "joint" / "metrics.json"


def run_args_for_case(
    *,
    image: str,
    candidate: Candidate,
    size: int,
    pretrain_iter: int,
    num_iter: int,
    seed: int,
    train_verbose: bool,
    seidel_convention: str,
    sweep_args: argparse.Namespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        image=image,
        size=size,
        modes=["joint"],
        run_name=None,
        num_iter=num_iter,
        pretrain_iter=pretrain_iter,
        lr_obj=sweep_args.lr_obj,
        lr_seidel=sweep_args.lr_seidel,
        rsd_weight=sweep_args.rsd_weight,
        tv_weight=sweep_args.tv_weight,
        pretrain_scalar=sweep_args.pretrain_scalar,
        defocus_anchor_weight=sweep_args.defocus_anchor_weight,
        defocus_index=sweep_args.defocus_index,
        seidel_parameterization=sweep_args.seidel_parameterization,
        seidel_rms_prior_mode=sweep_args.seidel_rms_prior_mode,
        seidel_rms_floor_weight=sweep_args.seidel_rms_floor_weight,
        seidel_rms_floor_alpha=sweep_args.seidel_rms_floor_alpha,
        seidel_rms_floor_target=candidate.actual_rms,
        seidel_rms_floor_field_samples=sweep_args.seidel_rms_floor_field_samples,
        seidel_rms_floor_pupil_samples=sweep_args.seidel_rms_floor_pupil_samples,
        gt_fixed_seidel_indices=list(sweep_args.gt_fixed_seidel_indices),
        seidel_lr_multipliers=(
            None
            if sweep_args.seidel_lr_multipliers is None
            else list(sweep_args.seidel_lr_multipliers)
        ),
        scheduler=sweep_args.scheduler,
        eta_min_ratio=sweep_args.eta_min_ratio,
        max_val=sweep_args.max_val,
        nerf_beta=sweep_args.nerf_beta,
        output_mode=sweep_args.output_mode,
        nerf_depth=sweep_args.nerf_depth,
        nerf_width=sweep_args.nerf_width,
        nerf_skips=sweep_args.nerf_skips,
        fourier_num_angles=sweep_args.fourier_num_angles,
        fourier_num_octaves=sweep_args.fourier_num_octaves,
        seidel_convention=seidel_convention,
        gt_preset="custom",
        gt_seidel_json=json.dumps(candidate.seidel.astype(float).tolist()),
        gt_label=candidate.candidate_id,
        gt_source="custom",
        seed=seed,
        verbose=train_verbose,
    )


def augment_metrics(
    metrics: dict,
    *,
    stage: str,
    image: str,
    candidate: Candidate,
    seed: int,
    seidel_convention: str,
) -> dict:
    gt = np.asarray(metrics["seidel_gt"], dtype=np.float64)
    rec = np.asarray(metrics["seidel_final"], dtype=np.float64)
    gt_rms = field_weighted_wavefront_rms(gt)
    rec_rms = field_weighted_wavefront_rms(rec)
    err_rms = field_weighted_wavefront_rms(rec - gt)
    signblind_err_rms = min(err_rms, field_weighted_wavefront_rms((-rec) - gt))
    coeff_l2 = float(np.linalg.norm(rec - gt))
    coeff_rel = coeff_l2 / max(float(np.linalg.norm(gt)), 1e-12)
    gt_hf = float(metrics.get("gt_hf_ratio", 0.0))
    meas_hf = float(metrics.get("measurement_hf_ratio", 0.0))
    config = metrics.get("config", {})
    metrics.update(
        {
            "stage": stage,
            "image": image,
            "seed": seed,
            "candidate_id": candidate.candidate_id,
            "direction": candidate.direction,
            "target_wavefront_rms": candidate.target_rms,
            "actual_wavefront_rms": candidate.actual_rms,
            "wavefront_gt_rms": gt_rms,
            "wavefront_recovered_rms": rec_rms,
            "wavefront_error_rms": err_rms,
            "relative_wavefront_error": err_rms / max(gt_rms, 1e-12),
            "signblind_wavefront_error_rms": signblind_err_rms,
            "signblind_relative_wavefront_error": signblind_err_rms / max(gt_rms, 1e-12),
            "seidel_l2_relative": coeff_rel,
            "measurement_hf_drop": 1.0 - (meas_hf / max(gt_hf, 1e-12)),
            "seidel_convention": seidel_convention,
            "seidel_rms_prior_mode": str(config.get("seidel_rms_prior_mode", "floor")),
            "seidel_rms_floor_weight": float(config.get("seidel_rms_floor_weight", 0.0)),
            "seidel_rms_floor_alpha": float(config.get("seidel_rms_floor_alpha", 0.8)),
            "seidel_rms_floor_target": config.get("seidel_rms_floor_target"),
            "seidel_parameterization": str(config.get("seidel_parameterization", "direct")),
            "seidel_amplitude_final": metrics.get("seidel_amplitude_final"),
            "seidel_direction_rms_final": metrics.get("seidel_direction_rms_final"),
            "gt_fixed_seidel_indices": list(config.get("gt_fixed_seidel_indices", [])),
            "gt_fixed_seidel_values": [
                float(gt[int(idx)]) for idx in config.get("gt_fixed_seidel_indices", [])
            ],
            "seidel_lr_multipliers": config.get("seidel_lr_multipliers"),
            "wavefront_recovered_over_gt_rms": rec_rms / max(gt_rms, 1e-12),
            **cocoa.convention_metadata(seidel_convention),
        }
    )
    return metrics


def run_case(
    *,
    output_root: Path,
    stage: str,
    image: str,
    candidate: Candidate,
    size: int,
    pretrain_iter: int,
    num_iter: int,
    seed: int,
    force: bool,
    train_verbose: bool,
    seidel_convention: str,
    sweep_args: argparse.Namespace,
) -> dict:
    metrics_path = case_metrics_path(output_root, stage, image, candidate.candidate_id, seed=seed)
    case_dir = metrics_path.parents[1]
    if metrics_path.is_file() and not force:
        metrics = json.loads(metrics_path.read_text())
        if "relative_wavefront_error" not in metrics or "fixed_seidel_indices" not in metrics:
            metrics = augment_metrics(
                metrics,
                stage=stage,
                image=image,
                candidate=candidate,
                seed=seed,
                seidel_convention=seidel_convention,
            )
            metrics_path.write_text(json.dumps(metrics, indent=2))
        return metrics

    case_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_args = run_args_for_case(
        image=image,
        candidate=candidate,
        size=size,
        pretrain_iter=pretrain_iter,
        num_iter=num_iter,
        seed=seed,
        train_verbose=train_verbose,
        seidel_convention=seidel_convention,
        sweep_args=sweep_args,
    )
    gt_vec = torch.tensor(candidate.seidel, device=device, dtype=torch.float32)
    sharp_gt = cocoa.load_baboon_gt(size, path=cocoa.IMAGE_PATHS[image], device=device)
    meas_gt = cocoa.synthesize_measurement(sharp_gt, gt_vec, cocoa.SYS_PARAMS)

    print(
        f"[case] {stage} seed={seed} image={image} candidate={candidate.candidate_id} "
        f"size={size} pre={pretrain_iter} joint={num_iter}",
        flush=True,
    )
    result, metrics = cocoa.run_one_mode(
        run_args,
        mode="joint",
        sharp_gt=sharp_gt,
        meas_gt=meas_gt,
        gt_vec=gt_vec,
        gt_np=candidate.seidel,
        root_dir=case_dir,
        device=device,
    )
    metrics = augment_metrics(
        metrics,
        stage=stage,
        image=image,
        candidate=candidate,
        seed=seed,
        seidel_convention=seidel_convention,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2))
    cocoa.save_summary_figure(case_dir, sharp_gt, meas_gt, [("joint", result, metrics)])
    (case_dir / "summary.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "image": image,
                "seed": seed,
                "candidate": candidate.__dict__ | {"seidel": candidate.seidel.tolist()},
                "metrics_path": str(metrics_path),
            },
            indent=2,
        )
    )
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return metrics


def collect_metrics(output_root: Path, stage: str) -> list[dict]:
    if stage in {"stage1", "stage2"}:
        paths = sorted((output_root / stage).glob("*/joint/metrics.json"))
    elif stage == "stage3":
        paths = sorted((output_root / stage).glob("seed*/*/joint/metrics.json"))
    else:
        raise ValueError(stage)
    return [json.loads(path.read_text()) for path in paths]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    preferred = [
        "stage",
        "image",
        "seed",
        "candidate_id",
        "direction",
        "target_wavefront_rms",
        "actual_wavefront_rms",
        "wavefront_gt_rms",
        "wavefront_recovered_rms",
        "wavefront_recovered_over_gt_rms",
        "relative_wavefront_error",
        "wavefront_error_rms",
        "signblind_relative_wavefront_error",
        "seidel_l2_relative",
        "seidel_rms_prior_mode",
        "seidel_rms_floor_weight",
        "seidel_rms_floor_alpha",
        "seidel_rms_floor_target",
        "seidel_parameterization",
        "seidel_amplitude_final",
        "seidel_direction_rms_final",
        "gt_fixed_seidel_indices",
        "gt_fixed_seidel_values",
        "seidel_lr_multipliers",
        "final_seidel_rms_floor_loss",
        "final_seidel_wavefront_rms_floor_estimate",
        "l2_seidel_vs_gt",
        "ssim_recon_gain_vs_gt",
        "nrmse_recon_gain_vs_gt",
        "gt_hf_ratio",
        "measurement_hf_ratio",
        "measurement_hf_drop",
        "recon_raw_hf_ratio",
        "final_ssim_loss",
        "elapsed_s",
        "seidel_gt",
        "seidel_final",
    ]
    extra = sorted({k for row in rows for k in row if k not in preferred and not isinstance(row.get(k), (dict, list))})
    fieldnames = [k for k in preferred if any(k in row for row in rows)] + extra
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                out[key] = value
            writer.writerow(out)


def mean_metric(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return float("nan")
    return float(np.mean(values))


def candidate_summary(rows: list[dict], *, images: list[str], min_object_ssim: float) -> list[dict]:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        if float(row.get("ssim_recon_gain_vs_gt") or 0.0) < min_object_ssim:
            continue
        grouped.setdefault(row["candidate_id"], {}).setdefault(row["image"], []).append(row)
    summaries = []
    for candidate_id, by_image in grouped.items():
        if not all(by_image.get(image) for image in images):
            continue
        image_rows = [by_image[image] for image in images]
        image_means = [
            {
                "relative_wavefront_error": mean_metric(group, "relative_wavefront_error"),
                "ssim_recon_gain_vs_gt": mean_metric(group, "ssim_recon_gain_vs_gt"),
                "nrmse_recon_gain_vs_gt": mean_metric(group, "nrmse_recon_gain_vs_gt"),
            }
            for group in image_rows
        ]
        all_rows = [row for group in image_rows for row in group]
        rel_all = [float(row["relative_wavefront_error"]) for row in all_rows]
        summaries.append(
            {
                "candidate_id": candidate_id,
                "direction": all_rows[0]["direction"],
                "target_wavefront_rms": float(all_rows[0]["target_wavefront_rms"]),
                "mean_relative_wavefront_error": float(np.mean([r["relative_wavefront_error"] for r in image_means])),
                "median_relative_wavefront_error": float(np.median([r["relative_wavefront_error"] for r in image_means])),
                "max_relative_wavefront_error": float(np.max([r["relative_wavefront_error"] for r in image_means])),
                "std_relative_wavefront_error": float(np.std(rel_all)),
                "mean_ssim": float(np.mean([r["ssim_recon_gain_vs_gt"] for r in image_means])),
                "mean_nrmse": float(np.mean([r["nrmse_recon_gain_vs_gt"] for r in image_means])),
                "num_images": len(image_rows),
                "num_runs": len(all_rows),
            }
        )
    summaries.sort(key=lambda r: (r["mean_relative_wavefront_error"], r["max_relative_wavefront_error"]))
    return summaries


def summarize_stability_group(rows: list[dict], *, image: str) -> dict:
    rel = np.asarray([float(row["relative_wavefront_error"]) for row in rows], dtype=np.float64)
    ssim = np.asarray([float(row["ssim_recon_gain_vs_gt"]) for row in rows], dtype=np.float64)
    nrmse = np.asarray([float(row["nrmse_recon_gain_vs_gt"]) for row in rows], dtype=np.float64)
    return {
        "stage": "stage3",
        "image": image,
        "candidate_id": rows[0]["candidate_id"],
        "direction": rows[0]["direction"],
        "target_wavefront_rms": float(rows[0]["target_wavefront_rms"]),
        "actual_wavefront_rms": float(rows[0]["actual_wavefront_rms"]),
        "seeds": ",".join(str(int(row["seed"])) for row in sorted(rows, key=lambda r: int(r["seed"]))),
        "num_runs": len(rows),
        "mean_relative_wavefront_error": float(np.mean(rel)),
        "std_relative_wavefront_error": float(np.std(rel)),
        "min_relative_wavefront_error": float(np.min(rel)),
        "max_relative_wavefront_error": float(np.max(rel)),
        "mean_ssim": float(np.mean(ssim)),
        "std_ssim": float(np.std(ssim)),
        "mean_nrmse": float(np.mean(nrmse)),
        "std_nrmse": float(np.std(nrmse)),
    }


def stage3_stability_rows(rows: list[dict], images: list[str]) -> list[dict]:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        grouped.setdefault(row["candidate_id"], {}).setdefault(row["image"], []).append(row)

    out: list[dict] = []
    for candidate_id in sorted(grouped):
        by_image = grouped[candidate_id]
        all_rows = []
        for image in images:
            if not by_image.get(image):
                continue
            image_rows = by_image[image]
            out.append(summarize_stability_group(image_rows, image=image))
            all_rows.extend(image_rows)
        if all_rows:
            out.append(summarize_stability_group(all_rows, image="ALL"))
    out.sort(key=lambda row: (row["candidate_id"], row["image"] != "ALL", row["image"]))
    return out


def select_stage2_candidates(rows: list[dict], args: argparse.Namespace) -> list[str]:
    summaries = candidate_summary(rows, images=args.images, min_object_ssim=args.min_object_ssim)
    if not summaries:
        raise RuntimeError("No complete Stage 1 candidates passed the object SSIM filter")
    global_top = [row["candidate_id"] for row in summaries[: args.stage2_global_top]]
    rank = {row["candidate_id"]: idx for idx, row in enumerate(summaries)}
    selected = list(global_top)
    for image in args.images:
        image_rows = [
            row
            for row in rows
            if row["image"] == image
            and float(row.get("ssim_recon_gain_vs_gt") or 0.0) >= args.min_object_ssim
        ]
        if not image_rows:
            continue
        image_rows.sort(key=lambda r: r["relative_wavefront_error"])
        selected.append(image_rows[0]["candidate_id"])
    selected = sorted(set(selected), key=lambda cid: rank.get(cid, 10**9))
    return selected[: args.max_stage2_candidates]


def select_stage3_candidates(rows: list[dict], args: argparse.Namespace) -> list[str]:
    summaries = candidate_summary(rows, images=args.images, min_object_ssim=args.min_object_ssim)
    if not summaries:
        raise RuntimeError("No complete Stage 2 candidates passed the object SSIM filter")
    return [row["candidate_id"] for row in summaries[: args.stage3_global_top]]


def candidate_lookup(candidates: list[Candidate]) -> dict[str, Candidate]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def stage_cases(images: list[str], candidates: list[Candidate]) -> list[tuple[str, Candidate]]:
    return [(image, candidate) for image in images for candidate in candidates]


def plot_heatmap(rows: list[dict], output_root: Path, images: list[str]) -> None:
    if not rows:
        return
    summaries = sorted(
        {(row["candidate_id"], row["direction"], float(row["target_wavefront_rms"])) for row in rows},
        key=lambda x: (x[1], x[2], x[0]),
    )
    candidate_ids = [x[0] for x in summaries]
    labels = [f"{direction}\n{target:.2f}" for _, direction, target in summaries]
    matrix = np.full((len(candidate_ids), len(images)), np.nan, dtype=np.float64)
    for row in rows:
        if row["candidate_id"] in candidate_ids and row["image"] in images:
            i = candidate_ids.index(row["candidate_id"])
            j = images.index(row["image"])
            matrix[i, j] = row["relative_wavefront_error"]
    fig_h = max(8, 0.28 * len(candidate_ids))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    im = ax.imshow(matrix, cmap="viridis_r", aspect="auto", vmin=0, vmax=np.nanpercentile(matrix, 95))
    ax.set_xticks(range(len(images)), labels=images, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=7)
    ax.set_title("Stage 1 relative wavefront error (lower is better)")
    fig.colorbar(im, ax=ax, label="relative wavefront error")
    fig.tight_layout()
    fig.savefig(output_root / "heatmap_relative_wavefront_error.png", dpi=150)
    plt.close(fig)


def plot_scatter(rows: list[dict], output_root: Path) -> None:
    if not rows:
        return
    directions = sorted({row["direction"] for row in rows})
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, direction in enumerate(directions):
        group = [row for row in rows if row["direction"] == direction]
        ax.scatter(
            [row["target_wavefront_rms"] for row in group],
            [row["relative_wavefront_error"] for row in group],
            s=20,
            alpha=0.35,
            color=cmap(idx % 10),
            label=direction,
        )
    ax.set_xlabel("target wavefront RMS (waves)")
    ax.set_ylabel("relative wavefront error")
    ax.set_title("Recoverability vs wavefront strength")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_root / "scatter_recoverability_vs_rms.png", dpi=150)
    plt.close(fig)


def norm01(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, [1.0, 99.7])
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def choose_overview_row(rows: list[dict], *, image: str, candidate_id: str) -> dict | None:
    matches = [row for row in rows if row["image"] == image and row["candidate_id"] == candidate_id]
    if not matches:
        return None
    rel = np.asarray([float(row["relative_wavefront_error"]) for row in matches], dtype=np.float64)
    median = float(np.median(rel))
    return min(matches, key=lambda row: abs(float(row["relative_wavefront_error"]) - median))


def load_tensor_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def plot_overview(output_root: Path, rows: list[dict], images: list[str], candidates: list[str]) -> None:
    if not rows or not candidates:
        return
    stage = rows[0]["stage"]
    fig, ax = plt.subplots(len(images), 1 + len(candidates), figsize=(4 * (1 + len(candidates)), 4 * len(images)))
    if ax.ndim == 1:
        ax = ax.reshape(len(images), 1 + len(candidates))
    for row_idx, image in enumerate(images):
        first_row = choose_overview_row(rows, image=image, candidate_id=candidates[0])
        first_path = find_tensor_path(
            output_root,
            stage,
            image,
            candidates[0],
            seed=int(first_row.get("seed", 0)) if first_row else 0,
        )
        if first_path is None:
            continue
        first = load_tensor_dict(first_path)
        ax[row_idx, 0].imshow(norm01(first["sharp_gt"].numpy()), cmap="gray")
        ax[row_idx, 0].set_title(f"{image}\nGT")
        ax[row_idx, 0].axis("off")
        for col_idx, candidate_id in enumerate(candidates, start=1):
            row = choose_overview_row(rows, image=image, candidate_id=candidate_id)
            tensor_path = find_tensor_path(output_root, stage, image, candidate_id, seed=int(row.get("seed", 0)) if row else 0)
            if row is None or tensor_path is None:
                ax[row_idx, col_idx].axis("off")
                continue
            tensors = load_tensor_dict(tensor_path)
            ax[row_idx, col_idx].imshow(norm01(tensors["sharp_recon"].numpy()), cmap="gray")
            ax[row_idx, col_idx].set_title(
                f"{candidate_id}\nrelWF {row['relative_wavefront_error']:.3f} "
                f"SSIM {row['ssim_recon_gain_vs_gt']:.3f}",
                fontsize=8,
            )
            ax[row_idx, col_idx].axis("off")
    fig.suptitle(f"Top candidates overview ({stage})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_root / "overview_top_candidates.png", dpi=150)
    plt.close(fig)


def find_tensor_path(output_root: Path, stage: str, image: str, candidate_id: str, seed: int = 0) -> Path | None:
    if stage == "stage3":
        path = output_root / stage / f"seed{seed}" / f"{image}__{candidate_id}" / "joint" / "tensors.pt"
    else:
        path = output_root / stage / f"{image}__{candidate_id}" / "joint" / "tensors.pt"
    return path if path.is_file() else None


def write_best_candidates(output_root: Path, args: argparse.Namespace) -> None:
    lines = ["# Seidel Recovery Sweep Candidates", ""]
    best_stage = None
    best_summaries: list[dict] = []
    for stage in ["stage1", "stage2", "stage3"]:
        rows = collect_metrics(output_root, stage)
        if not rows:
            continue
        summaries = candidate_summary(rows, images=args.images, min_object_ssim=args.min_object_ssim)
        if summaries:
            best_stage = stage
            best_summaries = summaries
        lines += [f"## {stage}", ""]
        lines.append(
            "| rank | candidate | direction | target RMS | mean rel WF err | "
            "max rel WF err | std rel WF err | mean SSIM |"
        )
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
        for idx, row in enumerate(summaries[:15], start=1):
            lines.append(
                f"| {idx} | {row['candidate_id']} | {row['direction']} | "
                f"{row['target_wavefront_rms']:.3f} | "
                f"{row['mean_relative_wavefront_error']:.3f} | "
                f"{row['max_relative_wavefront_error']:.3f} | "
                f"{row['std_relative_wavefront_error']:.3f} | {row['mean_ssim']:.3f} |"
            )
        lines.append("")

    if best_summaries:
        primary = [row for row in best_summaries if float(row["target_wavefront_rms"]) > 0.040001]
        primary = primary or best_summaries
        top = primary[: min(5, len(primary))]
        directions = sorted({row["direction"] for row in top})
        rms_values = [float(row["target_wavefront_rms"]) for row in top]
        lines += ["## Recoverability region", ""]
        lines.append(
            f"Based on {best_stage}, the easiest recoverable region is concentrated in "
            f"{', '.join(directions)} with target wavefront RMS about "
            f"{min(rms_values):.3f}-{max(rms_values):.3f} waves."
        )
        lines.append(
            "The 0.040 waves cases are kept as small-aberration controls and are not prioritized "
            "for the main region when larger candidates are available."
        )
        lines.append("")
    output_root.joinpath("best_candidates.md").write_text("\n".join(lines))


def generate_reports(output_root: Path, args: argparse.Namespace) -> None:
    stage1 = collect_metrics(output_root, "stage1")
    stage2 = collect_metrics(output_root, "stage2")
    stage3 = collect_metrics(output_root, "stage3")
    write_csv(stage1, output_root / "stage1_metrics.csv")
    write_csv(stage2, output_root / "stage2_metrics.csv")
    write_csv(stage3, output_root / "stage3_metrics_raw.csv")
    write_csv(stage3_stability_rows(stage3, args.images), output_root / "stage3_seed_stability.csv")
    if stage1:
        plot_heatmap(stage1, output_root, args.images)
        plot_scatter(stage1, output_root)
    overview_rows = stage3 or stage2 or stage1
    if overview_rows:
        summaries = candidate_summary(overview_rows, images=args.images, min_object_ssim=args.min_object_ssim)
        top_candidates = [row["candidate_id"] for row in summaries[:3]]
        plot_overview(output_root, overview_rows, args.images, top_candidates)
    write_best_candidates(output_root, args)


def run_stage1_case_subprocess(
    output_root: Path,
    args: argparse.Namespace,
    *,
    image: str,
    candidate: Candidate,
) -> None:
    metrics_path = case_metrics_path(output_root, "stage1", image, candidate.candidate_id)
    if metrics_path.is_file() and not args.force:
        print(f"[skip-subprocess] stage1 image={image} candidate={candidate.candidate_id}", flush=True)
        return

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-name",
        args.run_name,
        "--stage",
        "stage1",
        "--images",
        image,
        "--directions",
        candidate.direction,
        "--strengths",
        f"{candidate.target_rms:.12g}",
        "--seidel-convention",
        args.seidel_convention,
        "--stage1-size",
        str(args.stage1_size),
        "--stage1-pretrain-iter",
        str(args.stage1_pretrain_iter),
        "--stage1-num-iter",
        str(args.stage1_num_iter),
        "--lr-obj",
        str(args.lr_obj),
        "--lr-seidel",
        str(args.lr_seidel),
        "--rsd-weight",
        str(args.rsd_weight),
        "--tv-weight",
        str(args.tv_weight),
        "--pretrain-scalar",
        str(args.pretrain_scalar),
        "--defocus-anchor-weight",
        str(args.defocus_anchor_weight),
        "--defocus-index",
        str(args.defocus_index),
        "--seidel-parameterization",
        args.seidel_parameterization,
        "--scheduler",
        args.scheduler or "none",
        "--eta-min-ratio",
        str(args.eta_min_ratio),
        "--max-val",
        str(args.max_val),
        "--nerf-beta",
        str(args.nerf_beta),
        "--output-mode",
        args.output_mode,
        "--nerf-depth",
        str(args.nerf_depth),
        "--nerf-width",
        str(args.nerf_width),
        "--nerf-skips",
        cocoa.format_nerf_skips(args.nerf_skips),
        "--fourier-num-angles",
        str(args.fourier_num_angles),
        "--fourier-num-octaves",
        str(args.fourier_num_octaves),
        "--seidel-rms-prior-mode",
        args.seidel_rms_prior_mode,
        "--seidel-rms-floor-weight",
        str(args.seidel_rms_floor_weight),
        "--seidel-rms-floor-alpha",
        str(args.seidel_rms_floor_alpha),
        "--seidel-rms-floor-field-samples",
        str(args.seidel_rms_floor_field_samples),
        "--seidel-rms-floor-pupil-samples",
        str(args.seidel_rms_floor_pupil_samples),
        "--skip-report",
        "--skip-config-write",
    ]
    if args.gt_fixed_seidel_indices:
        cmd.append("--gt-fixed-seidel-indices")
        cmd.extend(str(int(idx)) for idx in args.gt_fixed_seidel_indices)
    if args.seidel_lr_multipliers is not None:
        cmd.extend(
            [
                "--seidel-lr-multipliers-json",
                json.dumps([float(value) for value in args.seidel_lr_multipliers]),
            ]
        )
    if args.force:
        cmd.append("--force")
    if args.train_verbose:
        cmd.append("--train-verbose")

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[subprocess] stage1 image={image} candidate={candidate.candidate_id}", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def run_stage1(output_root: Path, args: argparse.Namespace, candidates: list[Candidate]) -> None:
    cases = stage_cases(args.images, candidates)
    cases = shard_items(cases, shard_index=args.shard_index, num_shards=args.num_shards)
    for image, candidate in cases:
        if args.case_subprocess:
            run_stage1_case_subprocess(output_root, args, image=image, candidate=candidate)
            continue
        run_case(
            output_root=output_root,
            stage="stage1",
            image=image,
            candidate=candidate,
            size=args.stage1_size,
            pretrain_iter=args.stage1_pretrain_iter,
            num_iter=args.stage1_num_iter,
            seed=0,
            force=args.force,
            train_verbose=args.train_verbose,
            seidel_convention=args.seidel_convention,
            sweep_args=args,
        )


def run_stage2(output_root: Path, args: argparse.Namespace, candidates: list[Candidate]) -> list[str]:
    stage1_rows = collect_metrics(output_root, "stage1")
    selected = select_stage2_candidates(stage1_rows, args)
    lookup = candidate_lookup(candidates)
    cases = stage_cases(args.images, [lookup[cid] for cid in selected])
    cases = shard_items(cases, shard_index=args.shard_index, num_shards=args.num_shards)
    for image, candidate in cases:
        run_case(
            output_root=output_root,
            stage="stage2",
            image=image,
            candidate=candidate,
            size=args.stage2_size,
            pretrain_iter=args.stage2_pretrain_iter,
            num_iter=args.stage2_num_iter,
            seed=0,
            force=args.force,
            train_verbose=args.train_verbose,
            seidel_convention=args.seidel_convention,
            sweep_args=args,
        )
    return selected


def run_stage3(output_root: Path, args: argparse.Namespace, candidates: list[Candidate]) -> list[str]:
    stage2_rows = collect_metrics(output_root, "stage2")
    selected = select_stage3_candidates(stage2_rows, args)
    lookup = candidate_lookup(candidates)
    all_cases = []
    for seed in args.stage3_seeds:
        for image, candidate in stage_cases(args.images, [lookup[cid] for cid in selected]):
            all_cases.append((seed, image, candidate))
    all_cases = shard_items(all_cases, shard_index=args.shard_index, num_shards=args.num_shards)
    for seed, image, candidate in all_cases:
        run_case(
            output_root=output_root,
            stage="stage3",
            image=image,
            candidate=candidate,
            size=args.stage3_size,
            pretrain_iter=args.stage3_pretrain_iter,
            num_iter=args.stage3_num_iter,
            seed=seed,
            force=args.force,
            train_verbose=args.train_verbose,
            seidel_convention=args.seidel_convention,
            sweep_args=args,
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--stage", choices=["all", "stage1", "stage2", "stage3", "report"], default="all")
    parser.add_argument("--images", nargs="+", choices=sorted(cocoa.IMAGE_PATHS), default=IMAGES)
    parser.add_argument("--directions", nargs="+", choices=sorted(DIRECTIONS), default=list(DIRECTIONS))
    parser.add_argument("--strengths", nargs="+", default=[str(v) for v in STRENGTHS])
    parser.add_argument(
        "--seidel-convention",
        choices=list(cocoa.CLASSICAL_CONVENTIONS),
        default="backend6",
        help=(
            "Seidel recovery convention. Use classical5d for Wd fixed to zero "
            "and classical4d for W311,Wd fixed to zero."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train-verbose", action="store_true")
    parser.add_argument(
        "--case-subprocess",
        action="store_true",
        help="Run each Stage 1 case in a fresh Python process to avoid CUDA allocator fragmentation.",
    )
    parser.add_argument("--skip-report", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-config-write", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--min-object-ssim", type=float, default=0.50)
    parser.add_argument("--stage2-global-top", type=int, default=6)
    parser.add_argument("--max-stage2-candidates", type=int, default=10)
    parser.add_argument("--stage3-global-top", type=int, default=3)
    parser.add_argument("--stage3-seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--stage1-size", type=int, default=STAGE1["size"])
    parser.add_argument("--stage1-pretrain-iter", type=int, default=STAGE1["pretrain_iter"])
    parser.add_argument("--stage1-num-iter", type=int, default=STAGE1["num_iter"])
    parser.add_argument("--stage2-size", type=int, default=STAGE2["size"])
    parser.add_argument("--stage2-pretrain-iter", type=int, default=STAGE2["pretrain_iter"])
    parser.add_argument("--stage2-num-iter", type=int, default=STAGE2["num_iter"])
    parser.add_argument("--stage3-size", type=int, default=STAGE3["size"])
    parser.add_argument("--stage3-pretrain-iter", type=int, default=STAGE3["pretrain_iter"])
    parser.add_argument("--stage3-num-iter", type=int, default=STAGE3["num_iter"])
    parser.add_argument("--lr-obj", type=float, default=5e-3)
    parser.add_argument("--lr-seidel", type=float, default=1e-2)
    parser.add_argument(
        "--seidel-lr-multipliers-json",
        default=None,
        help=(
            "Optional JSON/list multipliers applied to Seidel gradients before "
            "the optimizer step. Example '[10,1,1,1,1,1]' makes W040 use 10x "
            "the base Seidel learning rate."
        ),
    )
    parser.add_argument("--rsd-weight", type=float, default=5e-4)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--pretrain-scalar", type=float, default=5.0)
    parser.add_argument("--defocus-anchor-weight", type=float, default=1.0)
    parser.add_argument("--defocus-index", type=int, default=5)
    parser.add_argument(
        "--seidel-parameterization",
        choices=cocoa.SEIDEL_PARAMETERIZATIONS,
        default="direct",
        help=(
            "Internal Seidel optimization parameterization. 'direct' preserves "
            "the historical coefficient training; amp_direction variants train "
            "theta=a*u with unit wavefront-RMS direction u."
        ),
    )
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--eta-min-ratio", type=float, default=1.0 / 25.0)
    parser.add_argument("--max-val", type=float, default=40.0)
    parser.add_argument("--nerf-beta", type=float, default=1.0)
    parser.add_argument("--output-mode", choices=["softplus", "sigmoid"], default="softplus")
    parser.add_argument("--nerf-depth", type=int, default=6)
    parser.add_argument("--nerf-width", type=int, default=128)
    parser.add_argument(
        "--nerf-skips",
        default="2,4,6",
        help="Comma-separated NeuralObject2D skip indices, or 'none'.",
    )
    parser.add_argument("--fourier-num-angles", type=int, default=60)
    parser.add_argument("--fourier-num-octaves", type=int, default=7)
    parser.add_argument(
        "--seidel-rms-prior-mode",
        choices=["floor", "ratio_target"],
        default="floor",
        help=(
            "Seidel RMS prior form. 'floor' uses max(0, alpha*target - recovered)^2; "
            "'ratio_target' uses (recovered/target - alpha)^2."
        ),
    )
    parser.add_argument(
        "--seidel-rms-floor-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a hinge prior penalizing recovered Seidel wavefront RMS "
            "below alpha * target RMS. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--seidel-rms-floor-alpha",
        type=float,
        default=0.8,
        help="Floor fraction of the candidate target wavefront RMS.",
    )
    parser.add_argument("--seidel-rms-floor-field-samples", type=int, default=21)
    parser.add_argument("--seidel-rms-floor-pupil-samples", type=int, default=51)
    parser.add_argument(
        "--gt-fixed-seidel-indices",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Backend-6 Seidel indices to lock to GT values while recovering "
            "the remaining coefficients. Example: 0 fixes W040 to GT."
        ),
    )
    args = parser.parse_args()
    args.scheduler = None if args.scheduler == "none" else args.scheduler
    args.nerf_skips = cocoa.parse_nerf_skips(args.nerf_skips)
    args.strengths = parse_float_list(args.strengths, STRENGTHS)
    if args.seidel_lr_multipliers_json:
        args.seidel_lr_multipliers = cocoa.parse_float_vector_json(
            args.seidel_lr_multipliers_json,
            name="--seidel-lr-multipliers-json",
        )
        expected_dim = cocoa.active_backend_dim(args.seidel_convention)
        if len(args.seidel_lr_multipliers) != expected_dim:
            raise ValueError(
                "--seidel-lr-multipliers-json length must match trainable Seidel "
                f"dimension {expected_dim}, got {len(args.seidel_lr_multipliers)}"
            )
        if any(value < 0.0 for value in args.seidel_lr_multipliers):
            raise ValueError("--seidel-lr-multipliers-json values must be non-negative")
    else:
        args.seidel_lr_multipliers = None
    args.gt_fixed_seidel_indices = sorted({int(idx) for idx in args.gt_fixed_seidel_indices})
    invalid_gt_fixed = [idx for idx in args.gt_fixed_seidel_indices if idx < 0 or idx >= 6]
    if invalid_gt_fixed:
        raise ValueError(f"--gt-fixed-seidel-indices out of backend-6 range: {invalid_gt_fixed}")
    if args.seidel_parameterization != "direct":
        if args.seidel_convention not in {"classical6d", "backend6"}:
            raise ValueError("--seidel-parameterization amp_direction* requires classical6d/backend6")
        if args.gt_fixed_seidel_indices:
            raise ValueError("--seidel-parameterization amp_direction* does not support --gt-fixed-seidel-indices")
        if args.seidel_lr_multipliers is not None:
            raise ValueError("--seidel-parameterization amp_direction* does not support --seidel-lr-multipliers-json")
    if args.seidel_rms_floor_weight < 0.0:
        raise ValueError("--seidel-rms-floor-weight must be non-negative")
    if args.seidel_rms_floor_alpha < 0.0:
        raise ValueError("--seidel-rms-floor-alpha must be non-negative")
    if args.seidel_rms_floor_field_samples < 2:
        raise ValueError("--seidel-rms-floor-field-samples must be >= 2")
    if args.seidel_rms_floor_pupil_samples < 3:
        raise ValueError("--seidel-rms-floor-pupil-samples must be >= 3")
    if args.run_name is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"seidel_recovery_sweep_{stamp}"
    if args.stage == "all" and args.num_shards > 1:
        raise ValueError("Use --stage stage1/stage2/stage3 for sharded execution, not --stage all")
    if args.case_subprocess and args.stage != "stage1":
        raise ValueError("--case-subprocess currently supports --stage stage1 only")
    return args


def write_sweep_config(output_root: Path, args: argparse.Namespace, candidates: list[Candidate]) -> None:
    (output_root / "sweep_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                **cocoa.convention_metadata(args.seidel_convention),
                "directions": DIRECTIONS,
                "candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "direction": candidate.direction,
                        "target_rms": candidate.target_rms,
                        "actual_rms": candidate.actual_rms,
                        "seidel": candidate.seidel.tolist(),
                    }
                    for candidate in candidates
                ],
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    candidates = make_candidates(
        args.directions,
        args.strengths,
        seidel_convention=args.seidel_convention,
    )
    output_root = cocoa.PROJECT_ROOT / "outputs" / "cocoa_like_2d_mechanism" / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_config_write:
        write_sweep_config(output_root, args, candidates)

    if args.stage in {"all", "stage1"}:
        run_stage1(output_root, args, candidates)
        if args.case_subprocess and not args.skip_config_write:
            write_sweep_config(output_root, args, candidates)
        if not args.skip_report:
            generate_reports(output_root, args)
    if args.stage in {"all", "stage2"}:
        selected = run_stage2(output_root, args, candidates)
        print(f"[stage2] selected candidates: {selected}", flush=True)
        if not args.skip_report:
            generate_reports(output_root, args)
    if args.stage in {"all", "stage3"}:
        selected = run_stage3(output_root, args, candidates)
        print(f"[stage3] selected candidates: {selected}", flush=True)
        if not args.skip_report:
            generate_reports(output_root, args)
    if args.stage == "report":
        generate_reports(output_root, args)
    print(f"[done] sweep root: {output_root}", flush=True)


if __name__ == "__main__":
    main()
