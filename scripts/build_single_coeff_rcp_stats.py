"""Build RCP panels and statistics for the single-coefficient Seidel sweep."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import (  # noqa: E402
    field_weighted_wavefront_rms,
)


RUN_NAME = "single_coeff_recovery_6d_size256_four_images_pre400_joint1000_20260607"
COEFF_LABELS = ["W040", "W131", "W222", "W220", "W311", "Wd"]
VALUE_ORDER = [0.1, 0.2, 0.4, -0.1, -0.2, -0.4]
IMAGE_ORDER = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple, int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def parse_vector(value: Any) -> np.ndarray:
    parsed = parse_jsonish(value)
    if parsed is None:
        raise ValueError("Missing vector value")
    arr = np.asarray([float(item) for item in parsed], dtype=np.float64)
    if arr.shape != (6,):
        raise ValueError(f"Expected a 6D Seidel vector, got shape {arr.shape}")
    return arr


def parse_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value in (None, ""):
        return float(default)
    return float(value)


def tag_float(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")


def short_float(value: float, digits: int = 4) -> str:
    if math.isnan(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def load_tensors(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def percentile01(arr: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if float(hi) <= float(lo):
        return normalize01(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def wrapped(text: str, width: int = 100) -> str:
    return "\n".join(wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_operator_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for coeff in COEFF_LABELS:
        csv_path = run_dir / f"single_coeff_operator_eval_{coeff}_dim256" / "seidel_physical_operator_metrics.csv"
        coeff_rows = read_csv(csv_path)
        for row in coeff_rows:
            row["_operator_eval_coeff"] = coeff
        rows.extend(coeff_rows)
    rows.sort(
        key=lambda row: (
            IMAGE_ORDER.index(row["image"]) if row["image"] in IMAGE_ORDER else 999,
            COEFF_LABELS.index(row["active_seidel_name"]),
            VALUE_ORDER.index(round(float(row["active_seidel_value"]), 4)),
        )
    )
    return rows


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        active_idx = int(float(row["active_seidel_index"]))
        seidel_gt = parse_vector(row["seidel_gt"])
        seidel_raw = parse_vector(row["seidel_final"])
        seidel_aligned = parse_vector(row["aligned_seidel_physical"])
        out = dict(row)
        out["active_idx"] = active_idx
        out["active_gt"] = float(seidel_gt[active_idx])
        out["active_raw"] = float(seidel_raw[active_idx])
        out["active_aligned"] = float(seidel_aligned[active_idx])
        out["active_aligned_error"] = out["active_aligned"] - out["active_gt"]
        out["active_aligned_abs_error"] = abs(out["active_aligned_error"])
        out["gt_rms"] = field_weighted_wavefront_rms(seidel_gt)
        out["rec_raw_rms"] = field_weighted_wavefront_rms(seidel_raw)
        out["rec_aligned_rms"] = field_weighted_wavefront_rms(seidel_aligned)
        out["operator_error_calibrated_f"] = parse_float(row, "operator_error_calibrated")
        out["operator_error_phys_equiv_f"] = parse_float(row, "operator_error_phys_equiv")
        out["operator_error_coord_diagnostic_f"] = parse_float(row, "operator_error_coord_diagnostic")
        out["ssim_f"] = parse_float(row, "ssim_recon_gain_vs_gt")
        out["nrmse_f"] = parse_float(row, "nrmse_recon_gain_vs_gt")
        out["relative_wavefront_error_f"] = parse_float(row, "relative_wavefront_error")
        enriched.append(out)
    return enriched


def mean(values: list[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else math.nan


def grouped_summary(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, group in sorted(groups.items(), key=lambda item: item[0]):
        summary = {key: value for key, value in zip(keys, key_values)}
        summary.update(
            {
                "rows": len(group),
                "operator_error_calibrated_mean": mean([r["operator_error_calibrated_f"] for r in group]),
                "operator_error_phys_equiv_mean": mean([r["operator_error_phys_equiv_f"] for r in group]),
                "operator_error_coord_diagnostic_mean": mean([r["operator_error_coord_diagnostic_f"] for r in group]),
                "relative_wavefront_error_mean": mean([r["relative_wavefront_error_f"] for r in group]),
                "active_aligned_abs_error_mean": mean([r["active_aligned_abs_error"] for r in group]),
                "ssim_mean": mean([r["ssim_f"] for r in group]),
                "nrmse_mean": mean([r["nrmse_f"] for r in group]),
            }
        )
        out.append(summary)
    return out


def pivot_mean(
    rows: list[dict[str, Any]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
    row_order: list[Any],
    col_order: list[Any],
) -> np.ndarray:
    matrix = np.full((len(row_order), len(col_order)), np.nan, dtype=np.float64)
    for r_idx, row_value in enumerate(row_order):
        for c_idx, col_value in enumerate(col_order):
            vals = [
                float(row[value_key])
                for row in rows
                if row[row_key] == row_value and round(float(row[col_key]), 4) == round(float(col_value), 4)
            ]
            if vals:
                matrix[r_idx, c_idx] = float(np.mean(vals))
    return matrix


def save_heatmap(
    matrix: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_title(title)
    ax.set_xlabel("active coefficient value")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8, color="white")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_stat_plots(rows: list[dict[str, Any]], out_dir: Path) -> None:
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    value_labels = [f"{v:g}" for v in VALUE_ORDER]
    metrics = [
        ("operator_error_calibrated_f", "mean operator_error_calibrated", "operator_error_heatmap_by_coeff_value.png", "viridis_r"),
        ("active_aligned_abs_error", "mean active aligned abs error", "active_coeff_abs_error_heatmap.png", "magma_r"),
        ("ssim_f", "mean SSIM_gain", "ssim_heatmap_by_coeff_value.png", "viridis"),
        ("relative_wavefront_error_f", "mean relative wavefront error", "relative_wavefront_error_heatmap.png", "viridis_r"),
    ]
    for value_key, label, filename, cmap in metrics:
        matrix = pivot_mean(
            rows,
            row_key="active_seidel_name",
            col_key="active_seidel_value",
            value_key=value_key,
            row_order=COEFF_LABELS,
            col_order=VALUE_ORDER,
        )
        save_heatmap(
            matrix,
            row_labels=COEFF_LABELS,
            col_labels=value_labels,
            title=label,
            cbar_label=label,
            out_path=stats_dir / filename,
            cmap=cmap,
        )

    coeff_summary = grouped_summary(rows, ["active_seidel_name"])
    image_summary = grouped_summary(rows, ["image"])
    coeff_value_summary = grouped_summary(rows, ["active_seidel_name", "active_seidel_value"])
    image_coeff_summary = grouped_summary(rows, ["image", "active_seidel_name"])
    write_csv(coeff_summary, stats_dir / "coefficient_summary.csv")
    write_csv(image_summary, stats_dir / "image_summary.csv")
    write_csv(coeff_value_summary, stats_dir / "coefficient_value_summary.csv")
    write_csv(image_coeff_summary, stats_dir / "image_coefficient_summary.csv")

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    xs = np.arange(len(COEFF_LABELS))
    means = [
        mean([r["operator_error_calibrated_f"] for r in rows if r["active_seidel_name"] == coeff])
        for coeff in COEFF_LABELS
    ]
    ax.bar(xs, means, color="#3B82B8")
    ax.set_xticks(xs, labels=COEFF_LABELS)
    ax.set_ylabel("mean operator_error_calibrated")
    ax.set_title("Single-coeff recovery difficulty by coefficient")
    ax.grid(axis="y", alpha=0.25)
    for x, val in zip(xs, means):
        ax.text(x, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(stats_dir / "operator_error_by_coefficient.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    width = 0.18
    xs = np.arange(len(COEFF_LABELS))
    colors = ["#3B82B8", "#7CB342", "#F58518", "#C44E52"]
    for idx, image in enumerate(IMAGE_ORDER):
        vals = [
            mean([r["operator_error_calibrated_f"] for r in rows if r["active_seidel_name"] == coeff and r["image"] == image])
            for coeff in COEFF_LABELS
        ]
        ax.bar(xs + (idx - 1.5) * width, vals, width, label=image, color=colors[idx])
    ax.set_xticks(xs, labels=COEFF_LABELS)
    ax.set_ylabel("mean operator_error_calibrated")
    ax.set_title("Operator error by image and coefficient")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(stats_dir / "operator_error_by_image_and_coefficient.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    for coeff in COEFF_LABELS:
        group = [r for r in rows if r["active_seidel_name"] == coeff]
        ax.scatter(
            [r["active_gt"] for r in group],
            [r["active_aligned"] for r in group],
            s=24,
            alpha=0.72,
            label=coeff,
        )
    lim = 0.44
    ax.plot([-lim, lim], [-lim, lim], color="0.35", linewidth=1.0, linestyle="--")
    ax.axhline(0.0, color="0.78", linewidth=0.8)
    ax.axvline(0.0, color="0.78", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("GT active coefficient")
    ax.set_ylabel("aligned recovered active coefficient")
    ax.set_title("Active coefficient recovery scatter")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(stats_dir / "active_coefficient_recovery_scatter.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    abs_values = [0.1, 0.2, 0.4]
    for coeff in COEFF_LABELS:
        pos = []
        neg = []
        for abs_value in abs_values:
            pos.append(mean([r["operator_error_calibrated_f"] for r in rows if r["active_seidel_name"] == coeff and round(float(r["active_seidel_value"]), 4) == abs_value]))
            neg.append(mean([r["operator_error_calibrated_f"] for r in rows if r["active_seidel_name"] == coeff and round(float(r["active_seidel_value"]), 4) == -abs_value]))
        ax.plot(abs_values, pos, marker="o", label=f"{coeff} +")
        ax.plot(abs_values, neg, marker="x", linestyle="--", label=f"{coeff} -")
    ax.set_xlabel("|coefficient value|")
    ax.set_ylabel("mean operator_error_calibrated")
    ax.set_title("Positive vs negative coefficient recoverability")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(stats_dir / "positive_negative_operator_symmetry.png", dpi=160)
    plt.close(fig)


def case_dir_for(run_dir: Path, row: dict[str, Any]) -> Path:
    return run_dir / "stage1" / f"{row['image']}__{row['candidate_id']}" / "joint"


def make_rcp(row: dict[str, Any], *, run_dir: Path, out_dir: Path, rank: int) -> dict[str, Any]:
    case_dir = case_dir_for(run_dir, row)
    metrics = json.loads((case_dir / "metrics.json").read_text())
    tensors = load_tensors(case_dir / "tensors.pt")

    gt = as_array(tensors["sharp_gt"])
    meas = as_array(tensors["measurement_gt"])
    recon = as_array(tensors["sharp_recon"])
    pred = as_array(tensors["measurement_pred"])
    gain = float(metrics.get("best_gain_recon_to_gt", 1.0))
    recon_gain = recon * gain
    err = np.abs(recon_gain - gt)

    seidel_gt = parse_vector(row["seidel_gt"])
    seidel_raw = parse_vector(row["seidel_final"])
    seidel_aligned = parse_vector(row["aligned_seidel_physical"])
    gt_rms = float(row["gt_rms"])
    raw_rms = float(row["rec_raw_rms"])
    aligned_rms = float(row["rec_aligned_rms"])
    active_name = str(row["active_seidel_name"])
    active_value = float(row["active_seidel_value"])
    image = str(row["image"])
    candidate = str(row["candidate_id"])

    out_subdir = out_dir / "rcp_all" / image / active_name
    out_subdir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"rank{rank:03d}__opcal{tag_float(row['operator_error_calibrated_f'])}__"
        f"{image}__{candidate}__four_panel_plus_coeff_similarity.png"
    )
    out_path = out_subdir / filename

    fig = plt.figure(figsize=(20.0, 6.0), dpi=150)
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 1.0],
        left=0.025,
        right=0.985,
        top=0.92,
        bottom=0.075,
        wspace=0.06,
    )
    left = outer[0, 0].subgridspec(2, 3, wspace=0.08, hspace=0.16)
    image_items = [
        ("Sharp GT", gt, "gray", normalize01),
        ("Measurement", meas, "gray", normalize01),
        ("Recon raw clipped", recon, "gray", normalize01),
        ("Recon percentile", recon, "gray", percentile01),
        ("Predicted measurement", pred, "gray", normalize01),
        ("Gain-aligned abs error", err, "magma", percentile01),
    ]
    for idx, (title, arr, cmap, norm_fn) in enumerate(image_items):
        ax = fig.add_subplot(left[idx // 3, idx % 3])
        ax.imshow(norm_fn(arr), cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.text(
        0.27,
        0.972,
        (
            f"SSIM_gain={short_float(row['ssim_f'])}  "
            f"NRMSE_gain={short_float(row['nrmse_f'])}  "
            f"HF raw={short_float(float(metrics.get('recon_raw_hf_ratio', math.nan)))}  "
            f"HF meas={short_float(float(metrics.get('measurement_hf_ratio', math.nan)))}"
        ),
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    right = outer[0, 1].subgridspec(3, 1, height_ratios=[0.72, 2.05, 0.68], hspace=0.22)
    ax_text = fig.add_subplot(right[0, 0])
    ax_text.axis("off")
    title = f"Single-coeff RCP rank {rank:03d} | {image} | {active_name}={active_value:g}"
    lines = [
        f"op_cal={short_float(row['operator_error_calibrated_f'])} | "
        f"phys={short_float(row['operator_error_phys_equiv_f'])} | "
        f"coord={short_float(row['operator_error_coord_diagnostic_f'])}",
        (
            f"RMS waves: GT={short_float(gt_rms)} | "
            f"rec_aligned={short_float(aligned_rms)} | rec_raw={short_float(raw_rms)}"
        ),
        (
            f"active: GT={short_float(row['active_gt'])} | "
            f"raw={short_float(row['active_raw'])} | aligned={short_float(row['active_aligned'])} | "
            f"abs_err={short_float(row['active_aligned_abs_error'])}"
        ),
        (
            f"best_phys={row.get('best_physical_transform', '?')} | "
            f"aligned_WFrel={short_float(parse_float(row, 'aligned_wavefront_error_physical'))} | "
            f"relWF={short_float(row['relative_wavefront_error_f'])}"
        ),
        wrapped(f"case={image}__{candidate}", width=96),
    ]
    ax_text.text(0.0, 0.98, title, ha="left", va="top", fontsize=11, fontweight="bold")
    ax_text.text(0.0, 0.74, "\n".join(lines), ha="left", va="top", fontsize=8.8)

    ax_bar = fig.add_subplot(right[1, 0])
    x = np.arange(len(COEFF_LABELS), dtype=np.float64)
    width = 0.34
    ax_bar.bar(x - width / 2, seidel_gt, width, label="GT", color="#55b8b0")
    ax_bar.bar(x + width / 2, seidel_aligned, width, label="aligned recovered", color="#ef7d55")
    ax_bar.scatter(x, seidel_raw, marker="x", color="black", s=28, label="raw recovered", zorder=4)
    ax_bar.axhline(0.0, color="0.55", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(COEFF_LABELS, fontsize=8)
    ax_bar.set_ylabel("coefficient", fontsize=8)
    ax_bar.set_title("Seidel coefficients: GT vs raw/aligned recovered", fontsize=9)
    ax_bar.grid(axis="y", alpha=0.22)
    ax_bar.legend(loc="upper right", ncol=3, fontsize=7, frameon=False)
    ax_bar.tick_params(axis="y", labelsize=8)

    ax_table = fig.add_subplot(right[2, 0])
    ax_table.axis("off")
    table_values = [
        [short_float(v, 3) for v in seidel_gt],
        [short_float(v, 3) for v in seidel_raw],
        [short_float(v, 3) for v in seidel_aligned],
    ]
    table = ax_table.table(
        cellText=table_values,
        rowLabels=["GT", "raw", "aligned"],
        colLabels=COEFF_LABELS,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.15)
    for cell in table.get_celld().values():
        cell.set_linewidth(0.25)
        cell.set_edgecolor("0.75")

    fig.savefig(out_path)
    plt.close(fig)
    return {
        "rank": rank,
        "image": image,
        "candidate_id": candidate,
        "active_seidel_name": active_name,
        "active_seidel_value": active_value,
        "operator_error_calibrated": row["operator_error_calibrated_f"],
        "operator_error_phys_equiv": row["operator_error_phys_equiv_f"],
        "relative_wavefront_error": row["relative_wavefront_error_f"],
        "ssim": row["ssim_f"],
        "gt_rms": gt_rms,
        "rec_aligned_rms": aligned_rms,
        "rec_raw_rms": raw_rms,
        "path": display_path(out_path),
    }


def make_overview(manifest_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    selected: list[dict[str, Any]] = []
    for coeff in COEFF_LABELS:
        group = [row for row in manifest_rows if row["active_seidel_name"] == coeff]
        if not group:
            continue
        group_sorted = sorted(group, key=lambda row: float(row["operator_error_calibrated"]))
        selected.append(group_sorted[0])
        selected.append(group_sorted[-1])

    images = [Image.open(PROJECT_ROOT / row["path"]).convert("RGB") for row in selected]
    target_width = 1500
    resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    resized = []
    for image in images:
        height = int(round(image.height * target_width / image.width))
        resized.append(image.resize((target_width, height), resample_filter))
    gap = 24
    total_height = sum(image.height for image in resized) + gap * max(0, len(resized) - 1)
    canvas = Image.new("RGB", (target_width, total_height), "white")
    y = 0
    for image in resized:
        canvas.paste(image, (0, y))
        y += image.height + gap
    out_path = out_dir / "single_coeff_RCP_best_worst_overview.png"
    canvas.save(out_path)
    return out_path


def write_readme(out_dir: Path, manifest_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Single-Coefficient RCP And Stats",
        "",
        f"Generated RCP panels: {len(manifest_rows)}",
        "",
        "Key outputs:",
        "- `manifest.csv`: all RCP panel paths and metrics",
        "- `stats/`: coefficient/value/image summary CSVs and statistical plots",
        "- `single_coeff_RCP_best_worst_overview.png`: one best and one worst operator case per coefficient",
        "- `rcp_all/<image>/<coefficient>/`: all full RCP panels",
        "",
        "RCP panels report GT RMS, raw recovered RMS, and aligned recovered RMS using `field_weighted_wavefront_rms`.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/cocoa_like_2d_mechanism") / RUN_NAME,
    )
    parser.add_argument("--output-name", default="single_coeff_rcp_stats")
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = run_dir / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = enrich_rows(load_operator_rows(run_dir))
    rows_ranked = sorted(rows, key=lambda row: (row["operator_error_calibrated_f"], row["image"], row["candidate_id"]))
    full_rows: list[dict[str, Any]] = []
    for row in rows_ranked:
        full_rows.append(
            {
                "image": row["image"],
                "candidate_id": row["candidate_id"],
                "active_seidel_name": row["active_seidel_name"],
                "active_seidel_value": row["active_seidel_value"],
                "operator_error_calibrated": row["operator_error_calibrated_f"],
                "operator_error_phys_equiv": row["operator_error_phys_equiv_f"],
                "operator_error_coord_diagnostic": row["operator_error_coord_diagnostic_f"],
                "relative_wavefront_error": row["relative_wavefront_error_f"],
                "active_gt": row["active_gt"],
                "active_raw": row["active_raw"],
                "active_aligned": row["active_aligned"],
                "active_aligned_abs_error": row["active_aligned_abs_error"],
                "ssim": row["ssim_f"],
                "nrmse": row["nrmse_f"],
                "gt_rms": row["gt_rms"],
                "rec_raw_rms": row["rec_raw_rms"],
                "rec_aligned_rms": row["rec_aligned_rms"],
            }
        )
    write_csv(full_rows, out_dir / "combined_single_coeff_metrics.csv")
    make_stat_plots(rows, out_dir)

    manifest_rows = [
        make_rcp(row, run_dir=run_dir, out_dir=out_dir, rank=rank)
        for rank, row in enumerate(rows_ranked, start=1)
    ]
    write_csv(manifest_rows, out_dir / "manifest.csv")
    overview = make_overview(manifest_rows, out_dir)
    write_readme(out_dir, manifest_rows)
    print(f"[done] wrote {len(manifest_rows)} RCP panels")
    print(f"[done] overview {overview}")
    print(f"[done] stats {out_dir / 'stats'}")


if __name__ == "__main__":
    main()
