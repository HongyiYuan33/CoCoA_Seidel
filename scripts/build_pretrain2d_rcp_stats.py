"""Build RCP panels and statistics for the 4D pretrain iter/scalar sweep."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from collections import Counter, defaultdict
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


DEFAULT_PREFIX = "pretrain2d_tunedadam4d_size256_three_images_joint1000_20260608"
DEFAULT_OUTPUT_DIR = (
    "outputs/cocoa_like_2d_mechanism/"
    "pretrain2d_tunedadam4d_size256_three_images_joint1000_20260608_rcp_stats"
)
DEFAULT_CAPACITY_STATS_CSV = (
    "outputs/cocoa_like_2d_mechanism/"
    "capacity4d_dirrms_tunedprior_size256_four_images_20260607_rcp_stats/"
    "stats/combined_capacity4d_metrics.csv"
)
DEFAULT_CAPACITY_BASELINE_EVAL_CSV = (
    "outputs/cocoa_like_2d_mechanism/"
    "capacity4d_dirrms_tunedprior_size256_four_images_20260607__baseline/"
    "stage1_operator_eval_dim256/seidel_physical_operator_metrics.csv"
)

PRETRAIN_GRID = [
    (100, 1.0, "pre100_scalar1"),
    (100, 5.0, "pre100_scalar5"),
    (100, 7.5, "pre100_scalar7p5"),
    (400, 1.0, "pre400_scalar1"),
    (400, 5.0, "pre400_scalar5"),
    (400, 7.5, "pre400_scalar7p5"),
    (800, 1.0, "pre800_scalar1"),
    (800, 5.0, "pre800_scalar5"),
    (800, 7.5, "pre800_scalar7p5"),
]
IMAGE_ORDER = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]
DIRECTION_ORDER = ["cocoa_signed", "signed_balanced"]
RMS_ORDER = [0.06, 0.20, 0.40]
COEFF_LABELS = ["W040", "W131", "W222", "W220", "W311", "Wd"]


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
    return f"{float(value):.4f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")


def rms_tag(value: float) -> str:
    return f"rms{tag_float(value)}"


def scalar_tag(value: float) -> str:
    return f"scalar{tag_float(value)}"


def short_float(value: float, digits: int = 4) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_tensors(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def blank_like(reference: np.ndarray) -> np.ndarray:
    return np.zeros_like(reference, dtype=np.float32)


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


def wrapped(text: str, width: int = 98) -> str:
    return "\n".join(wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else math.nan


def median(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else math.nan


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("image", "")),
        str(row.get("direction", "")),
        str(row.get("candidate_id", "")),
    )


def case_group_key(row: dict[str, Any]) -> tuple[str, str, float]:
    return (
        str(row["image"]),
        str(row["direction"]),
        round(float(row["target_rms_f"]), 6),
    )


def run_dir_for(output_root: Path, prefix: str, tag: str) -> Path:
    return output_root / f"{prefix}__{tag}"


def case_dir_for(output_root: Path, prefix: str, row: dict[str, Any]) -> Path:
    return (
        run_dir_for(output_root, prefix, str(row["pretrain_tag"]))
        / "stage1"
        / f"{row['image']}__{row['candidate_id']}"
        / "joint"
    )


def stage1_lookup(run_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read_csv(run_dir / "stage1_metrics.csv"):
        out[row_key(row)] = row
    return out


def load_rows(output_root: Path, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pre_iter, scalar, tag in PRETRAIN_GRID:
        run_dir = run_dir_for(output_root, prefix, tag)
        eval_csv = run_dir / "stage1_operator_eval_dim256" / "seidel_physical_operator_metrics.csv"
        metrics_lookup = stage1_lookup(run_dir)
        for row_idx, row in enumerate(read_csv(eval_csv)):
            merged = dict(row)
            merged.update({k: v for k, v in metrics_lookup.get(row_key(row), {}).items() if k not in merged})
            merged["pretrain_iter"] = pre_iter
            merged["pretrain_scalar"] = scalar
            merged["pretrain_tag"] = tag
            merged["_source_row_index"] = row_idx
            rows.append(merged)
    return rows


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        seidel_gt = parse_vector(row["seidel_gt"])
        seidel_raw = parse_vector(row["seidel_final"])
        seidel_aligned = parse_vector(row["aligned_seidel_physical"])
        row = dict(row)
        row["target_rms_f"] = parse_float(row, "target_wavefront_rms")
        row["gt_rms"] = field_weighted_wavefront_rms(seidel_gt)
        row["rec_raw_rms"] = field_weighted_wavefront_rms(seidel_raw)
        row["rec_aligned_rms"] = field_weighted_wavefront_rms(seidel_aligned)
        row["operator_error_calibrated_f"] = parse_float(row, "operator_error_calibrated")
        row["operator_error_phys_equiv_f"] = parse_float(row, "operator_error_phys_equiv")
        row["operator_error_coord_diagnostic_f"] = parse_float(row, "operator_error_coord_diagnostic")
        row["ssim_f"] = parse_float(row, "ssim_recon_gain_vs_gt")
        row["nrmse_f"] = parse_float(row, "nrmse_recon_gain_vs_gt")
        row["relative_wavefront_error_f"] = parse_float(row, "relative_wavefront_error")
        row["aligned_wavefront_error_physical_f"] = parse_float(row, "aligned_wavefront_error_physical")
        row["aligned_coeff_relative_error_physical_f"] = parse_float(
            row,
            "aligned_coeff_relative_error_physical",
        )
        row["pretrain_final_loss_f"] = parse_float(row, "pretrain_final_loss")
        row["pretrain_render_ssim_vs_target_f"] = parse_float(
            row,
            "pretrain_render_ssim_vs_target",
        )
        row["pretrain_render_nrmse_vs_target_f"] = parse_float(
            row,
            "pretrain_render_nrmse_vs_target",
        )
        row["pretrain_render_hf_ratio_f"] = parse_float(row, "pretrain_render_hf_ratio")
        out.append(row)
    out.sort(
        key=lambda row: (
            IMAGE_ORDER.index(row["image"]) if row["image"] in IMAGE_ORDER else 999,
            DIRECTION_ORDER.index(row["direction"]) if row["direction"] in DIRECTION_ORDER else 999,
            row["target_rms_f"],
            int(row["pretrain_iter"]),
            float(row["pretrain_scalar"]),
        )
    )
    return out


def grouped_summary(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        record = {name: value for name, value in zip(keys, key)}
        record["count"] = len(group)
        for metric in [
            "operator_error_calibrated_f",
            "aligned_wavefront_error_physical_f",
            "aligned_coeff_relative_error_physical_f",
            "ssim_f",
            "nrmse_f",
            "pretrain_final_loss_f",
            "pretrain_render_ssim_vs_target_f",
            "pretrain_render_nrmse_vs_target_f",
            "pretrain_render_hf_ratio_f",
        ]:
            vals = [parse_float(row, metric) for row in group]
            record[f"{metric}_mean"] = mean(vals)
            record[f"{metric}_median"] = median(vals)
        out.append(record)
    return out


def best_rows_by_metric(rows: list[dict[str, Any]], metric_key: str, *, higher: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[case_group_key(row)].append(row)
    out = []
    for key in sorted(groups):
        group = groups[key]
        finite = [row for row in group if math.isfinite(parse_float(row, metric_key))]
        if not finite:
            continue
        best = max(finite, key=lambda row: parse_float(row, metric_key)) if higher else min(
            finite,
            key=lambda row: parse_float(row, metric_key),
        )
        out.append(dict(best))
    return out


def write_best_and_winner_stats(rows: list[dict[str, Any]], stats_dir: Path) -> dict[str, list[dict[str, Any]]]:
    best_op = best_rows_by_metric(rows, "operator_error_calibrated_f", higher=False)
    best_ssim = best_rows_by_metric(rows, "ssim_f", higher=True)
    best_nrmse = best_rows_by_metric(rows, "nrmse_f", higher=False)
    write_csv(best_op, stats_dir / "best_by_operator_error_calibrated.csv")
    write_csv(best_ssim, stats_dir / "best_by_ssim.csv")
    write_csv(best_nrmse, stats_dir / "best_by_nrmse.csv")

    winner_rows = []
    for metric_name, winners in [
        ("operator_error_calibrated", best_op),
        ("ssim", best_ssim),
        ("nrmse", best_nrmse),
    ]:
        counter = Counter(str(row["pretrain_tag"]) for row in winners)
        for tag, count in sorted(counter.items()):
            iter_value = next(pre for pre, _, item_tag in PRETRAIN_GRID if item_tag == tag)
            scalar_value = next(scalar for _, scalar, item_tag in PRETRAIN_GRID if item_tag == tag)
            winner_rows.append(
                {
                    "metric": metric_name,
                    "pretrain_tag": tag,
                    "pretrain_iter": iter_value,
                    "pretrain_scalar": scalar_value,
                    "winner_count": count,
                    "case_count": len(winners),
                }
            )
    write_csv(winner_rows, stats_dir / "winner_counts_by_pretrain_setting.csv")
    return {
        "operator_error_calibrated": best_op,
        "ssim": best_ssim,
        "nrmse": best_nrmse,
    }


def value_grid(rows: list[dict[str, Any]], metric_key: str) -> np.ndarray:
    iter_values = [item[0] for item in PRETRAIN_GRID[0::3]]
    scalar_values = [1.0, 5.0, 7.5]
    grid = np.full((len(iter_values), len(scalar_values)), np.nan, dtype=np.float64)
    for i, pre_iter in enumerate(iter_values):
        for j, scalar in enumerate(scalar_values):
            group = [
                row
                for row in rows
                if int(row["pretrain_iter"]) == pre_iter
                and abs(float(row["pretrain_scalar"]) - scalar) < 1e-9
            ]
            grid[i, j] = mean([parse_float(row, metric_key) for row in group])
    return grid


def plot_heatmaps(rows: list[dict[str, Any]], stats_dir: Path) -> None:
    iter_values = [100, 400, 800]
    scalar_values = [1.0, 5.0, 7.5]
    metrics = [
        ("operator_error_calibrated_f", "Mean operator error", "operator_error_heatmap_iter_scalar.png", "viridis_r"),
        ("ssim_f", "Mean SSIM gain", "ssim_heatmap_iter_scalar.png", "viridis"),
        ("nrmse_f", "Mean NRMSE gain", "nrmse_heatmap_iter_scalar.png", "viridis_r"),
    ]
    for metric_key, title, filename, cmap in metrics:
        grid = value_grid(rows, metric_key)
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        im = ax.imshow(grid, cmap=cmap)
        ax.set_xticks(np.arange(len(scalar_values)))
        ax.set_yticks(np.arange(len(iter_values)))
        ax.set_xticklabels([f"{v:g}" for v in scalar_values])
        ax.set_yticklabels([str(v) for v in iter_values])
        ax.set_xlabel("pretrain_scalar")
        ax.set_ylabel("pretrain_iter")
        ax.set_title(title)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, short_float(grid[i, j], 3), ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(stats_dir / filename, dpi=170)
        plt.close(fig)


def plot_pretrain_scatter(rows: list[dict[str, Any]], stats_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True)
    x = [parse_float(row, "pretrain_render_nrmse_vs_target_f") for row in rows]
    panels = [
        ("operator_error_calibrated_f", "operator error", False),
        ("ssim_f", "SSIM gain", True),
        ("nrmse_f", "NRMSE gain", False),
    ]
    colors = {tag: idx for idx, (_, _, tag) in enumerate(PRETRAIN_GRID)}
    cmap = plt.cm.tab10
    for ax, (metric_key, label, _) in zip(axes, panels):
        y = [parse_float(row, metric_key) for row in rows]
        c = [colors[str(row["pretrain_tag"])] for row in rows]
        ax.scatter(x, y, c=c, cmap=cmap, s=28, alpha=0.72)
        ax.set_xlabel("pretrain render NRMSE vs target")
        ax.set_ylabel(label)
        ax.grid(alpha=0.22)
    handles = []
    for tag, idx in colors.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=tag,
                markerfacecolor=cmap(idx % 10),
                markersize=7,
            )
        )
    axes[-1].legend(handles=handles, fontsize=7, frameon=False, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(stats_dir / "pretrain_quality_vs_final_metrics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_stats(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, stats_dir / "combined_pretrain_sweep_metrics.csv")
    write_csv(grouped_summary(rows, ["pretrain_iter", "pretrain_scalar", "pretrain_tag"]), stats_dir / "summary_by_iter_scalar.csv")
    write_csv(grouped_summary(rows, ["direction", "target_rms_f"]), stats_dir / "summary_by_direction_rms.csv")
    write_csv(grouped_summary(rows, ["image"]), stats_dir / "summary_by_image.csv")
    best = write_best_and_winner_stats(rows, stats_dir)
    plot_heatmaps(rows, stats_dir)
    plot_pretrain_scatter(rows, stats_dir)
    return best


def make_rcp(row: dict[str, Any], *, output_root: Path, prefix: str, out_dir: Path, rank: int) -> dict[str, Any]:
    case_dir = case_dir_for(output_root, prefix, row)
    metrics = json.loads((case_dir / "metrics.json").read_text())
    tensors = load_tensors(case_dir / "tensors.pt")

    gt = as_array(tensors["sharp_gt"])
    meas = as_array(tensors["measurement_gt"])
    pre_target = as_array(tensors.get("pretrain_target", blank_like(meas)))
    pre_render = as_array(tensors.get("pretrain_render", blank_like(meas)))
    pre_error = as_array(tensors.get("pretrain_abs_error", np.abs(pre_render - pre_target)))
    recon = as_array(tensors["sharp_recon"])
    pred = as_array(tensors["measurement_pred"])
    gain = float(metrics.get("best_gain_recon_to_gt", 1.0))
    recon_gain = recon * gain
    err = np.abs(recon_gain - gt)

    seidel_gt = parse_vector(row["seidel_gt"])
    seidel_raw = parse_vector(row["seidel_final"])
    seidel_aligned = parse_vector(row["aligned_seidel_physical"])
    image = str(row["image"])
    direction = str(row["direction"])
    target_rms = float(row["target_rms_f"])
    candidate = str(row["candidate_id"])
    tag = str(row["pretrain_tag"])

    out_subdir = out_dir / "rcp_all" / image / direction / rms_tag(target_rms)
    out_subdir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"rank{rank:03d}__opcal{tag_float(row['operator_error_calibrated_f'])}__"
        f"{tag}__{image}__{candidate}__pretrain_RCP.png"
    )
    out_path = out_subdir / filename

    fig = plt.figure(figsize=(20.6, 7.8), dpi=145)
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.08, 1.0],
        left=0.024,
        right=0.987,
        top=0.925,
        bottom=0.065,
        wspace=0.055,
    )
    left = outer[0, 0].subgridspec(3, 3, wspace=0.08, hspace=0.20)
    image_items = [
        ("Sharp GT", gt, "gray", normalize01),
        ("Measurement", meas, "gray", normalize01),
        ("Pretrain target", pre_target, "gray", percentile01),
        ("Pretrain render", pre_render, "gray", percentile01),
        ("Pretrain abs error", pre_error, "magma", percentile01),
        ("Recon raw clipped", recon, "gray", normalize01),
        ("Recon percentile", recon, "gray", percentile01),
        ("Predicted measurement", pred, "gray", normalize01),
        ("Final gain-aligned abs error", err, "magma", percentile01),
    ]
    for idx, (title, arr, cmap, norm_fn) in enumerate(image_items):
        ax = fig.add_subplot(left[idx // 3, idx % 3])
        ax.imshow(norm_fn(arr), cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=8.8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.text(
        0.275,
        0.975,
        (
            f"pre_loss={short_float(row['pretrain_final_loss_f'])}  "
            f"pre_NRMSE={short_float(row['pretrain_render_nrmse_vs_target_f'])}  "
            f"SSIM_gain={short_float(row['ssim_f'])}  "
            f"NRMSE_gain={short_float(row['nrmse_f'])}"
        ),
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    right = outer[0, 1].subgridspec(3, 1, height_ratios=[0.82, 2.05, 0.68], hspace=0.22)
    ax_text = fig.add_subplot(right[0, 0])
    ax_text.axis("off")
    title = (
        f"Pretrain2D RCP rank {rank:03d} | {image} | {direction} | "
        f"RMS={target_rms:g} | pre={int(row['pretrain_iter'])}, scalar={float(row['pretrain_scalar']):g}"
    )
    lines = [
        f"op_cal={short_float(row['operator_error_calibrated_f'])} | "
        f"phys={short_float(row['operator_error_phys_equiv_f'])} | "
        f"coord={short_float(row['operator_error_coord_diagnostic_f'])}",
        (
            f"RMS waves: GT={short_float(row['gt_rms'])} | "
            f"rec_aligned={short_float(row['rec_aligned_rms'])} | rec_raw={short_float(row['rec_raw_rms'])}"
        ),
        (
            f"pretrain: loss={short_float(row['pretrain_final_loss_f'])} | "
            f"SSIM_target={short_float(row['pretrain_render_ssim_vs_target_f'])} | "
            f"NRMSE_target={short_float(row['pretrain_render_nrmse_vs_target_f'])}"
        ),
        (
            f"aligned_WFrel={short_float(row['aligned_wavefront_error_physical_f'])} | "
            f"coeff_rel={short_float(row['aligned_coeff_relative_error_physical_f'])} | "
            f"best_phys={row.get('best_physical_transform', '?')}"
        ),
        wrapped(f"case={image}__{candidate}", width=96),
    ]
    ax_text.text(0.0, 0.98, title, ha="left", va="top", fontsize=11, fontweight="bold")
    ax_text.text(0.0, 0.73, "\n".join(lines), ha="left", va="top", fontsize=8.8)

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
        "direction": direction,
        "target_rms": target_rms,
        "candidate_id": candidate,
        "pretrain_iter": row["pretrain_iter"],
        "pretrain_scalar": row["pretrain_scalar"],
        "pretrain_tag": tag,
        "operator_error_calibrated": row["operator_error_calibrated_f"],
        "operator_error_phys_equiv": row["operator_error_phys_equiv_f"],
        "ssim": row["ssim_f"],
        "nrmse": row["nrmse_f"],
        "pretrain_final_loss": row["pretrain_final_loss_f"],
        "pretrain_render_nrmse_vs_target": row["pretrain_render_nrmse_vs_target_f"],
        "gt_rms": row["gt_rms"],
        "rec_aligned_rms": row["rec_aligned_rms"],
        "rec_raw_rms": row["rec_raw_rms"],
        "path": display_path(out_path),
    }


def make_overview(rows: list[dict[str, Any]], out_path: Path, *, target_width: int = 1600) -> None:
    if not rows:
        return
    images = [Image.open(PROJECT_ROOT / row["path"]).convert("RGB") for row in rows]
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_drift_check(rows: list[dict[str, Any]], reference_csv: Path | None, stats_dir: Path) -> list[dict[str, Any]]:
    if reference_csv is None or not reference_csv.is_file():
        write_csv([], stats_dir / "baseline_drift_check_pre400_scalar5_vs_Seidel4D-TunedAdam-256-v1.csv")
        return []
    ref_rows = []
    for row in read_csv(reference_csv):
        if row.get("profile") and row.get("profile") != "baseline":
            continue
        ref = dict(row)
        ref["target_rms_f"] = parse_float(ref, "target_wavefront_rms", parse_float(ref, "target_rms_f"))
        ref_rows.append(ref)
    refs = {row_key(row): row for row in ref_rows}
    new_rows = [row for row in rows if str(row["pretrain_tag"]) == "pre400_scalar5"]
    out = []
    for row in new_rows:
        ref = refs.get(row_key(row))
        if ref is None:
            continue
        out.append(
            {
                "image": row["image"],
                "direction": row["direction"],
                "candidate_id": row["candidate_id"],
                "target_rms": row["target_rms_f"],
                "new_operator_error_calibrated": row["operator_error_calibrated_f"],
                "ref_operator_error_calibrated": parse_float(ref, "operator_error_calibrated"),
                "delta_operator_error_calibrated": row["operator_error_calibrated_f"]
                - parse_float(ref, "operator_error_calibrated"),
                "new_ssim": row["ssim_f"],
                "ref_ssim": parse_float(ref, "ssim_recon_gain_vs_gt", parse_float(ref, "ssim_f")),
                "delta_ssim": row["ssim_f"]
                - parse_float(ref, "ssim_recon_gain_vs_gt", parse_float(ref, "ssim_f")),
                "new_nrmse": row["nrmse_f"],
                "ref_nrmse": parse_float(ref, "nrmse_recon_gain_vs_gt", parse_float(ref, "nrmse_f")),
                "delta_nrmse": row["nrmse_f"]
                - parse_float(ref, "nrmse_recon_gain_vs_gt", parse_float(ref, "nrmse_f")),
                "reference_csv": display_path(reference_csv.resolve()),
            }
        )
    write_csv(out, stats_dir / "baseline_drift_check_pre400_scalar5_vs_Seidel4D-TunedAdam-256-v1.csv")
    return out


def write_summary(out_dir: Path, rows: list[dict[str, Any]], drift_rows: list[dict[str, Any]]) -> None:
    stats_dir = out_dir / "stats"
    summary_rows = grouped_summary(rows, ["pretrain_iter", "pretrain_scalar", "pretrain_tag"])
    best_op = min(summary_rows, key=lambda row: float(row["operator_error_calibrated_f_mean"]))
    best_ssim = max(summary_rows, key=lambda row: float(row["ssim_f_mean"]))
    best_nrmse = min(summary_rows, key=lambda row: float(row["nrmse_f_mean"]))
    drift_op = mean([float(row["delta_operator_error_calibrated"]) for row in drift_rows])
    drift_ssim = mean([float(row["delta_ssim"]) for row in drift_rows])
    drift_nrmse = mean([float(row["delta_nrmse"]) for row in drift_rows])
    lines = [
        "# Pretrain Iter/Scalar Sweep RCP And Stats",
        "",
        f"Generated rows: {len(rows)}",
        "",
        "Best mean settings:",
        (
            f"- Operator error: {best_op['pretrain_tag']} "
            f"(mean={short_float(best_op['operator_error_calibrated_f_mean'])})"
        ),
        f"- SSIM: {best_ssim['pretrain_tag']} (mean={short_float(best_ssim['ssim_f_mean'])})",
        f"- NRMSE: {best_nrmse['pretrain_tag']} (mean={short_float(best_nrmse['nrmse_f_mean'])})",
        "",
        "Baseline drift check for pre400_scalar5 vs Seidel4D-TunedAdam-256-v1:",
        (
            f"- matched cases={len(drift_rows)}, "
            f"mean delta operator={short_float(drift_op)}, "
            f"mean delta SSIM={short_float(drift_ssim)}, "
            f"mean delta NRMSE={short_float(drift_nrmse)}"
        ),
        "",
        "Key outputs:",
        "- `manifest.csv`: all RCP panels",
        "- `RCP_best_operator_overview.png`: best operator-error panel per image/direction/RMS",
        "- `stats/summary_by_iter_scalar.csv`: aggregate table by pretrain setting",
        "- `stats/*_heatmap_iter_scalar.png`: 2D trends across pretrain_iter and pretrain_scalar",
        "- `stats/pretrain_quality_vs_final_metrics.png`: pretrain quality against final recovery metrics",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    (stats_dir / "summary.md").write_text("\n".join(lines) + "\n")


def find_default_reference_csv() -> Path | None:
    for rel in [DEFAULT_CAPACITY_STATS_CSV, DEFAULT_CAPACITY_BASELINE_EVAL_CSV]:
        path = PROJECT_ROOT / rel
        if path.is_file():
            return path
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/cocoa_like_2d_mechanism")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-reference-csv", type=Path, default=None)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = enrich_rows(load_rows(output_root, args.prefix))
    if not rows:
        raise SystemExit(f"No completed evaluator rows found for prefix {args.prefix!r}")

    best = make_stats(rows, out_dir)
    drift_reference = args.baseline_reference_csv.resolve() if args.baseline_reference_csv else find_default_reference_csv()
    drift_rows = write_drift_check(rows, drift_reference, out_dir / "stats")

    rows_by_identity = {
        (
            row["image"],
            row["direction"],
            round(float(row["target_rms_f"]), 6),
            row["pretrain_tag"],
        ): row
        for row in rows
    }
    manifest_rows = []
    ranked = sorted(rows, key=lambda row: row["operator_error_calibrated_f"])
    for rank, row in enumerate(ranked, start=1):
        manifest_rows.append(make_rcp(row, output_root=output_root, prefix=args.prefix, out_dir=out_dir, rank=rank))
    write_csv(manifest_rows, out_dir / "manifest.csv")

    manifest_by_identity = {
        (
            row["image"],
            row["direction"],
            round(float(row["target_rms"]), 6),
            row["pretrain_tag"],
        ): row
        for row in manifest_rows
    }
    overview = []
    for row in best["operator_error_calibrated"]:
        key = (
            row["image"],
            row["direction"],
            round(float(row["target_rms_f"]), 6),
            row["pretrain_tag"],
        )
        if key in rows_by_identity and key in manifest_by_identity:
            overview.append(manifest_by_identity[key])
    overview.sort(
        key=lambda row: (
            IMAGE_ORDER.index(row["image"]) if row["image"] in IMAGE_ORDER else 999,
            DIRECTION_ORDER.index(row["direction"]) if row["direction"] in DIRECTION_ORDER else 999,
            float(row["target_rms"]),
        )
    )
    make_overview(overview, out_dir / "RCP_best_operator_overview.png")
    write_summary(out_dir, rows, drift_rows)
    print(f"[done] rows={len(rows)} rcp={len(manifest_rows)} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
