"""Build RCP panels and statistics for the object-pretrain contrast mini-experiment."""

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


DEFAULT_PREFIX = "pretrain_contrast_init4d_size256_three_images_rms020_pre400_joint1000_20260608"
DEFAULT_OUTPUT_DIR = (
    "outputs/cocoa_like_2d_mechanism/"
    "pretrain_contrast_init4d_size256_three_images_rms020_pre400_joint1000_20260608_rcp_stats"
)

METHODS = [
    {
        "method": "baseline_scalar1",
        "label": "baseline: scalar=1",
        "pretrain_scalar": 1.0,
        "target_transform": "none",
        "contrast_alpha": 1.0,
        "pretrain_rsd_weight": 0.0,
        "pretrain_edge_weight": 0.0,
        "percentile_lo": 1.0,
        "percentile_hi": 99.0,
        "gamma": 1.0,
    },
    {
        "method": "A_scalar3",
        "label": "A: scalar=3",
        "pretrain_scalar": 3.0,
        "target_transform": "none",
        "contrast_alpha": 1.0,
        "pretrain_rsd_weight": 0.0,
        "pretrain_edge_weight": 0.0,
        "percentile_lo": 1.0,
        "percentile_hi": 99.0,
        "gamma": 1.0,
    },
    {
        "method": "B_scalar1_pretrain_rsd1e3",
        "label": "B: scalar=1 + RSD loss 1e-3",
        "pretrain_scalar": 1.0,
        "target_transform": "none",
        "contrast_alpha": 1.0,
        "pretrain_rsd_weight": 1e-3,
        "pretrain_edge_weight": 0.0,
        "percentile_lo": 1.0,
        "percentile_hi": 99.0,
        "gamma": 1.0,
    },
    {
        "method": "C_scalar1_contrast_alpha2",
        "label": "C: linear contrast alpha=2",
        "pretrain_scalar": 1.0,
        "target_transform": "linear_contrast",
        "contrast_alpha": 2.0,
        "pretrain_rsd_weight": 0.0,
        "pretrain_edge_weight": 0.0,
        "percentile_lo": 1.0,
        "percentile_hi": 99.0,
        "gamma": 1.0,
    },
    {
        "method": "D_scalar1_sobel_edge0p1",
        "label": "D: Sobel edge L1 0.1",
        "pretrain_scalar": 1.0,
        "target_transform": "none",
        "contrast_alpha": 1.0,
        "pretrain_rsd_weight": 0.0,
        "pretrain_edge_weight": 0.1,
        "percentile_lo": 1.0,
        "percentile_hi": 99.0,
        "gamma": 1.0,
    },
    {
        "method": "E_scalar1_p1p99_gamma0p7",
        "label": "E: p1/p99 + gamma 0.7",
        "pretrain_scalar": 1.0,
        "target_transform": "percentile_gamma",
        "contrast_alpha": 1.0,
        "pretrain_rsd_weight": 0.0,
        "pretrain_edge_weight": 0.0,
        "percentile_lo": 1.0,
        "percentile_hi": 99.0,
        "gamma": 0.7,
    },
]
METHOD_BY_NAME = {item["method"]: item for item in METHODS}
METHOD_ORDER = [item["method"] for item in METHODS]
IMAGE_ORDER = ["Iksung_beads", "dendrites", "dendrites_dense"]
COEFF_LABELS = ["W040", "W131", "W222", "W220", "W311", "Wd"]


def refresh_method_indexes() -> None:
    global METHOD_BY_NAME, METHOD_ORDER
    METHOD_BY_NAME = {str(item["method"]): item for item in METHODS}
    METHOD_ORDER = [str(item["method"]) for item in METHODS]


def load_settings_manifest(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Settings manifest must be a JSON list: {path}")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entry {idx} is not an object")
        row = dict(item)
        row.setdefault("label", row.get("method", f"setting {idx:03d}"))
        row.setdefault("family", "unspecified")
        row.setdefault("pretrain_scalar", 1.0)
        row.setdefault("target_transform", "none")
        row.setdefault("contrast_alpha", 1.0)
        row.setdefault("pretrain_rsd_weight", 0.0)
        row.setdefault("pretrain_edge_weight", 0.0)
        row.setdefault("percentile_lo", 1.0)
        row.setdefault("percentile_hi", 99.0)
        row.setdefault("gamma", 1.0)
        if "method" not in row:
            raise ValueError(f"Manifest entry {idx} has no method field")
        out.append(row)
    seen: set[str] = set()
    for row in out:
        method = str(row["method"])
        if method in seen:
            raise ValueError(f"Duplicate method in manifest: {method}")
        seen.add(method)
    return out


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
    if arr.size < 6:
        arr = np.pad(arr, (0, 6 - arr.size), constant_values=0.0)
    if arr.shape != (6,):
        raise ValueError(f"Expected a Seidel vector compatible with 6D display, got shape {arr.shape}")
    return arr


def parse_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def tag_float(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")


def rms_tag(value: float) -> str:
    return f"rms{tag_float(value)}"


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


def fixed01(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((np.asarray(arr, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0)


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


def case_group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["image"]),
        str(row["direction"]),
        str(row["candidate_id"]),
    )


def run_dir_for(output_root: Path, prefix: str, method: str) -> Path:
    return output_root / f"{prefix}__{method}"


def case_dir_for(output_root: Path, prefix: str, row: dict[str, Any]) -> Path:
    return (
        run_dir_for(output_root, prefix, str(row["pretrain_method"]))
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
    for method in METHODS:
        method_name = str(method["method"])
        run_dir = run_dir_for(output_root, prefix, method_name)
        eval_csv = run_dir / "stage1_operator_eval_dim256" / "seidel_physical_operator_metrics.csv"
        metrics_lookup = stage1_lookup(run_dir)
        for row_idx, row in enumerate(read_csv(eval_csv)):
            merged = dict(row)
            merged.update({k: v for k, v in metrics_lookup.get(row_key(row), {}).items() if k not in merged})
            merged["pretrain_method"] = method_name
            merged["method_label"] = method["label"]
            merged["_source_row_index"] = row_idx
            for key, value in method.items():
                merged.setdefault(key, value)
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
        row["pretrain_final_ssim_loss_f"] = parse_float(row, "pretrain_final_ssim_loss")
        row["pretrain_final_rsd_loss_f"] = parse_float(row, "pretrain_final_rsd_loss")
        row["pretrain_final_edge_loss_f"] = parse_float(row, "pretrain_final_edge_loss")
        row["pretrain_final_weighted_rsd_loss_f"] = parse_float(row, "pretrain_final_weighted_rsd_loss")
        row["pretrain_final_weighted_edge_loss_f"] = parse_float(row, "pretrain_final_weighted_edge_loss")
        row["pretrain_render_ssim_vs_target_f"] = parse_float(row, "pretrain_render_ssim_vs_target")
        row["pretrain_render_nrmse_vs_target_f"] = parse_float(row, "pretrain_render_nrmse_vs_target")
        row["pretrain_render_hf_ratio_f"] = parse_float(row, "pretrain_render_hf_ratio")
        out.append(row)
    out.sort(
        key=lambda row: (
            IMAGE_ORDER.index(row["image"]) if row["image"] in IMAGE_ORDER else 999,
            METHOD_ORDER.index(row["pretrain_method"]) if row["pretrain_method"] in METHOD_ORDER else 999,
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
            "pretrain_final_ssim_loss_f",
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
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
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


def comparison_by_case(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {
        case_group_key(row): row
        for row in rows
        if str(row["pretrain_method"]) == "baseline_scalar1"
    }
    out = []
    for row in rows:
        ref = baseline.get(case_group_key(row))
        record = {
            "image": row["image"],
            "direction": row["direction"],
            "candidate_id": row["candidate_id"],
            "target_rms": row["target_rms_f"],
            "pretrain_method": row["pretrain_method"],
            "method_label": row["method_label"],
            "operator_error_calibrated": row["operator_error_calibrated_f"],
            "ssim": row["ssim_f"],
            "nrmse": row["nrmse_f"],
            "aligned_wavefront_error_physical": row["aligned_wavefront_error_physical_f"],
            "aligned_coeff_relative_error_physical": row["aligned_coeff_relative_error_physical_f"],
            "pretrain_final_loss": row["pretrain_final_loss_f"],
            "pretrain_render_nrmse_vs_target": row["pretrain_render_nrmse_vs_target_f"],
        }
        if ref is not None:
            record.update(
                {
                    "baseline_operator_error_calibrated": ref["operator_error_calibrated_f"],
                    "delta_operator_error_calibrated": row["operator_error_calibrated_f"]
                    - ref["operator_error_calibrated_f"],
                    "baseline_ssim": ref["ssim_f"],
                    "delta_ssim": row["ssim_f"] - ref["ssim_f"],
                    "baseline_nrmse": ref["nrmse_f"],
                    "delta_nrmse": row["nrmse_f"] - ref["nrmse_f"],
                }
            )
        out.append(record)
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
        counter = Counter(str(row["pretrain_method"]) for row in winners)
        for method, count in sorted(counter.items(), key=lambda item: METHOD_ORDER.index(item[0])):
            winner_rows.append(
                {
                    "metric": metric_name,
                    "pretrain_method": method,
                    "method_label": METHOD_BY_NAME[method]["label"],
                    "winner_count": count,
                    "case_count": len(winners),
                }
            )
    write_csv(winner_rows, stats_dir / "winner_counts_by_method.csv")
    family_rows = []
    for metric_name, winners in [
        ("operator_error_calibrated", best_op),
        ("ssim", best_ssim),
        ("nrmse", best_nrmse),
    ]:
        counter = Counter(str(row.get("family", "unspecified")) for row in winners)
        for family, count in sorted(counter.items()):
            family_rows.append(
                {
                    "metric": metric_name,
                    "family": family,
                    "winner_count": count,
                    "case_count": len(winners),
                }
            )
    write_csv(family_rows, stats_dir / "winner_counts_by_family.csv")
    return {
        "operator_error_calibrated": best_op,
        "ssim": best_ssim,
        "nrmse": best_nrmse,
    }


def plot_method_metric_means(rows: list[dict[str, Any]], stats_dir: Path) -> None:
    summary = grouped_summary(rows, ["pretrain_method", "method_label"])
    summary.sort(key=lambda row: METHOD_ORDER.index(row["pretrain_method"]))
    x = np.arange(len(summary))
    panels = [
        ("operator_error_calibrated_f_mean", "Mean operator error", "lower is better"),
        ("ssim_f_mean", "Mean SSIM", "higher is better"),
        ("nrmse_f_mean", "Mean NRMSE", "lower is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.5))
    colors = plt.cm.Set2(np.linspace(0.05, 0.85, len(summary)))
    for ax, (metric, title, subtitle) in zip(axes, panels):
        vals = [float(row[metric]) for row in summary]
        ax.bar(x, vals, color=colors)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([str(row["pretrain_method"]).replace("_", "\n") for row in summary], fontsize=7)
        ax.grid(axis="y", alpha=0.2)
        for xi, val in zip(x, vals):
            ax.text(xi, val, short_float(val, 3), ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(stats_dir / "method_metric_means.png", dpi=170)
    plt.close(fig)


def plot_pretrain_scatter(rows: list[dict[str, Any]], stats_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True)
    x = [parse_float(row, "pretrain_render_nrmse_vs_target_f") for row in rows]
    panels = [
        ("operator_error_calibrated_f", "operator error"),
        ("ssim_f", "SSIM gain"),
        ("nrmse_f", "NRMSE gain"),
    ]
    colors = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    cmap = plt.cm.Set2
    for ax, (metric_key, label) in zip(axes, panels):
        y = [parse_float(row, metric_key) for row in rows]
        c = [colors[str(row["pretrain_method"])] for row in rows]
        ax.scatter(x, y, c=c, cmap=cmap, s=48, alpha=0.78)
        ax.set_xlabel("pretrain render NRMSE vs target")
        ax.set_ylabel(label)
        ax.grid(alpha=0.22)
    handles = []
    for method, idx in colors.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=method,
                markerfacecolor=cmap(idx % 8),
                markersize=7,
            )
        )
    axes[-1].legend(handles=handles, fontsize=7, frameon=False, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(stats_dir / "pretrain_quality_vs_final_metrics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def setting_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = grouped_summary(rows, ["pretrain_method", "method_label"])
    by_method = {str(row["pretrain_method"]): row for row in summary}
    out = []
    for method in METHOD_ORDER:
        if method not in by_method:
            continue
        row = dict(by_method[method])
        row.update(METHOD_BY_NAME.get(method, {}))
        out.append(row)
    return out


def plot_heatmap(
    summary_rows: list[dict[str, Any]],
    *,
    family: str,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    filename: str,
    stats_dir: Path,
) -> None:
    rows = [row for row in summary_rows if row.get("family") == family]
    if not rows:
        return
    x_vals = sorted({float(row[x_key]) for row in rows})
    y_vals = sorted({float(row[y_key]) for row in rows})
    if not x_vals or not y_vals:
        return
    grid = np.full((len(y_vals), len(x_vals)), np.nan, dtype=np.float64)
    for row in rows:
        x = x_vals.index(float(row[x_key]))
        y = y_vals.index(float(row[y_key]))
        grid[y, x] = float(row["operator_error_calibrated_f_mean"])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([f"{v:g}" for v in x_vals])
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_yticklabels([f"{v:g}" for v in y_vals])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{family}: mean operator error")
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if math.isfinite(grid[y, x]):
                ax.text(x, y, short_float(grid[y, x], 3), ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(stats_dir / filename, dpi=170)
    plt.close(fig)


def plot_family_metric_means(rows: list[dict[str, Any]], stats_dir: Path) -> None:
    summary = grouped_summary(rows, ["family"])
    if not summary:
        return
    summary.sort(key=lambda row: str(row["family"]))
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    panels = [
        ("operator_error_calibrated_f_mean", "Mean operator error"),
        ("ssim_f_mean", "Mean SSIM"),
        ("nrmse_f_mean", "Mean NRMSE"),
    ]
    for ax, (metric, title) in zip(axes, panels):
        vals = [float(row[metric]) for row in summary]
        ax.bar(x, vals, color="#6aa6d8")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(row["family"]).replace("_", "\n") for row in summary], fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(stats_dir / "family_metric_means.png", dpi=170)
    plt.close(fig)


def plot_parameter_trends(rows: list[dict[str, Any]], stats_dir: Path) -> None:
    summary = setting_means(rows)
    plot_family_metric_means(rows, stats_dir)
    plot_heatmap(
        summary,
        family="scalar_alpha",
        x_key="pretrain_scalar",
        y_key="contrast_alpha",
        x_label="pretrain_scalar",
        y_label="contrast_alpha",
        filename="heatmap_scalar_alpha_operator_error.png",
        stats_dir=stats_dir,
    )
    plot_heatmap(
        summary,
        family="scalar_rsd",
        x_key="pretrain_scalar",
        y_key="pretrain_rsd_weight",
        x_label="pretrain_scalar",
        y_label="pretrain_rsd_weight",
        filename="heatmap_scalar_rsd_operator_error.png",
        stats_dir=stats_dir,
    )
    plot_heatmap(
        summary,
        family="alpha_rsd",
        x_key="contrast_alpha",
        y_key="pretrain_rsd_weight",
        x_label="contrast_alpha",
        y_label="pretrain_rsd_weight",
        filename="heatmap_alpha_rsd_operator_error.png",
        stats_dir=stats_dir,
    )
    plot_heatmap(
        summary,
        family="percentile_gamma",
        x_key="gamma",
        y_key="percentile_lo",
        x_label="gamma",
        y_label="percentile_lo",
        filename="heatmap_percentile_gamma_operator_error.png",
        stats_dir=stats_dir,
    )


def make_stats(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, stats_dir / "combined_pretrain_contrast_metrics.csv")
    write_csv(comparison_by_case(rows), stats_dir / "comparison_by_case.csv")
    write_csv(grouped_summary(rows, ["pretrain_method", "method_label"]), stats_dir / "summary_by_method.csv")
    write_csv(setting_means(rows), stats_dir / "summary_by_setting.csv")
    write_csv(grouped_summary(rows, ["family"]), stats_dir / "summary_by_family.csv")
    write_csv(grouped_summary(rows, ["image"]), stats_dir / "summary_by_image.csv")
    best = write_best_and_winner_stats(rows, stats_dir)
    plot_method_metric_means(rows, stats_dir)
    plot_pretrain_scatter(rows, stats_dir)
    plot_parameter_trends(rows, stats_dir)
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
    method = str(row["pretrain_method"])

    out_subdir = out_dir / "rcp_all" / image / direction / rms_tag(target_rms)
    out_subdir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"rank{rank:03d}__opcal{tag_float(row['operator_error_calibrated_f'])}__"
        f"{method}__{image}__{candidate}__pretrain_contrast_RCP.png"
    )
    out_path = out_subdir / filename

    fig = plt.figure(figsize=(20.8, 7.9), dpi=145)
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
            f"{row['method_label']}  "
            f"pre_loss={short_float(row['pretrain_final_loss_f'])}  "
            f"pre_NRMSE={short_float(row['pretrain_render_nrmse_vs_target_f'])}  "
            f"SSIM_gain={short_float(row['ssim_f'])}  "
            f"NRMSE_gain={short_float(row['nrmse_f'])}"
        ),
        ha="center",
        va="top",
        fontsize=9.6,
        fontweight="bold",
    )

    right = outer[0, 1].subgridspec(3, 1, height_ratios=[0.86, 2.05, 0.68], hspace=0.22)
    ax_text = fig.add_subplot(right[0, 0])
    ax_text.axis("off")
    title = (
        f"Pretrain Contrast RCP rank {rank:03d} | {image} | {direction} | "
        f"RMS={target_rms:g} | {method}"
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
            f"ssim_loss={short_float(row['pretrain_final_ssim_loss_f'])} | "
            f"RSD={short_float(row['pretrain_final_weighted_rsd_loss_f'])} | "
            f"edge={short_float(row['pretrain_final_weighted_edge_loss_f'])}"
        ),
        (
            f"target={row.get('pretrain_target_transform', row.get('target_transform'))} | "
            f"scalar={row.get('pretrain_scalar')} | alpha={row.get('pretrain_contrast_alpha', row.get('contrast_alpha'))} | "
            f"gamma={row.get('pretrain_gamma', row.get('gamma'))}"
        ),
        (
            f"aligned_WFrel={short_float(row['aligned_wavefront_error_physical_f'])} | "
            f"coeff_rel={short_float(row['aligned_coeff_relative_error_physical_f'])} | "
            f"best_phys={row.get('best_physical_transform', '?')}"
        ),
        wrapped(f"case={image}__{candidate}", width=96),
    ]
    ax_text.text(0.0, 0.98, title, ha="left", va="top", fontsize=10.7, fontweight="bold")
    ax_text.text(0.0, 0.75, "\n".join(lines), ha="left", va="top", fontsize=8.4)

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
        "pretrain_method": method,
        "method_label": row["method_label"],
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


def make_fixed_scale_panels(rows: list[dict[str, Any]], *, output_root: Path, prefix: str, out_dir: Path) -> None:
    panels_dir = out_dir / "stats" / "fixed_scale_pretrain_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    rows_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_image[str(row["image"])].append(row)

    for image, image_rows in rows_by_image.items():
        image_rows.sort(key=lambda row: METHOD_ORDER.index(row["pretrain_method"]))
        target_arrays = []
        render_arrays = []
        error_arrays = []
        loaded: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
        for row in image_rows:
            tensors = load_tensors(case_dir_for(output_root, prefix, row) / "tensors.pt")
            target = as_array(tensors["pretrain_target"])
            render = as_array(tensors["pretrain_render"])
            error = as_array(tensors.get("pretrain_abs_error", np.abs(render - target)))
            target_arrays.append(target)
            render_arrays.append(render)
            error_arrays.append(error)
            loaded.append((row, target, render, error))
        value_lo = float(min(np.nanmin(arr) for arr in target_arrays + render_arrays))
        value_hi = float(max(np.nanmax(arr) for arr in target_arrays + render_arrays))
        err_lo = 0.0
        err_hi = float(max(np.nanmax(arr) for arr in error_arrays))

        fig, axes = plt.subplots(len(loaded), 3, figsize=(8.4, 2.15 * len(loaded)), dpi=170)
        if len(loaded) == 1:
            axes = np.asarray([axes])
        for row_idx, (row, target, render, error) in enumerate(loaded):
            items = [
                ("target fixed scale", fixed01(target, value_lo, value_hi), "gray"),
                ("render fixed scale", fixed01(render, value_lo, value_hi), "gray"),
                ("abs error fixed scale", fixed01(error, err_lo, err_hi), "magma"),
            ]
            for col_idx, (title, arr, cmap) in enumerate(items):
                ax = axes[row_idx, col_idx]
                ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0)
                if row_idx == 0:
                    ax.set_title(title, fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(str(row["pretrain_method"]).replace("_", "\n"), fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(
            f"{image}: fixed scale pretrain target/render/error "
            f"(value range {short_float(value_lo, 3)} to {short_float(value_hi, 3)})",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(panels_dir / f"{image}_fixed_scale_pretrain_methods.png")
        plt.close(fig)


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    stats_dir = out_dir / "stats"
    summary_rows = grouped_summary(rows, ["pretrain_method", "method_label"])
    summary_rows.sort(key=lambda row: METHOD_ORDER.index(row["pretrain_method"]))
    best_op = min(summary_rows, key=lambda row: float(row["operator_error_calibrated_f_mean"]))
    best_ssim = max(summary_rows, key=lambda row: float(row["ssim_f_mean"]))
    best_nrmse = min(summary_rows, key=lambda row: float(row["nrmse_f_mean"]))
    comp_rows = comparison_by_case(rows)
    improved = [
        row
        for row in comp_rows
        if row["pretrain_method"] != "baseline_scalar1"
        and math.isfinite(parse_float(row, "delta_operator_error_calibrated"))
        and parse_float(row, "delta_operator_error_calibrated") < 0
    ]
    lines = [
        "# Object-Pretrain Contrast Initialization Mini-Experiment",
        "",
        f"Generated rows: {len(rows)}",
        "",
        "Best mean methods:",
        (
            f"- Operator error: {best_op['pretrain_method']} "
            f"(mean={short_float(best_op['operator_error_calibrated_f_mean'])})"
        ),
        f"- SSIM: {best_ssim['pretrain_method']} (mean={short_float(best_ssim['ssim_f_mean'])})",
        f"- NRMSE: {best_nrmse['pretrain_method']} (mean={short_float(best_nrmse['nrmse_f_mean'])})",
        "",
        (
            "Cases with lower operator error than baseline_scalar1: "
            f"{len(improved)} / {max(0, len(comp_rows) - 3)} non-baseline comparisons"
        ),
        "",
        "Method means:",
    ]
    for row in summary_rows:
        lines.append(
            "- "
            f"{row['pretrain_method']}: op={short_float(row['operator_error_calibrated_f_mean'])}, "
            f"SSIM={short_float(row['ssim_f_mean'])}, "
            f"NRMSE={short_float(row['nrmse_f_mean'])}, "
            f"pre_NRMSE={short_float(row['pretrain_render_nrmse_vs_target_f_mean'])}"
        )
    lines.extend(
        [
            "",
            "Key outputs:",
            "- `manifest.csv`: all RCP panels",
            "- `settings_manifest.csv`: setting definitions used for this run",
            "- `RCP_best_operator_overview.png`: best operator-error panel per image",
            "- `stats/comparison_by_case.csv`: method results and deltas vs baseline",
            "- `stats/summary_by_setting.csv`: aggregate table by setting",
            "- `stats/summary_by_family.csv`: aggregate table by method family",
            "- `stats/method_metric_means.png`: compact metric bars",
            "- `stats/*heatmap*_operator_error.png`: coarse combination trends",
            "- `stats/fixed_scale_pretrain_panels/*.png`: fixed-scale target/render/error comparison",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    (stats_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    global METHODS
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/cocoa_like_2d_mechanism")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--settings-manifest",
        type=Path,
        default=None,
        help="JSON list of pretrain contrast settings. Defaults to the built-in mini-experiment settings.",
    )
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.settings_manifest is not None:
        METHODS = load_settings_manifest(args.settings_manifest.resolve())
        refresh_method_indexes()
    write_csv(METHODS, out_dir / "settings_manifest.csv")
    (out_dir / "settings_manifest.json").write_text(json.dumps(METHODS, indent=2) + "\n")

    rows = enrich_rows(load_rows(output_root, args.prefix))
    if not rows:
        raise SystemExit(f"No completed evaluator rows found for prefix {args.prefix!r}")

    best = make_stats(rows, out_dir)
    make_fixed_scale_panels(rows, output_root=output_root, prefix=args.prefix, out_dir=out_dir)

    manifest_rows = []
    ranked = sorted(rows, key=lambda row: row["operator_error_calibrated_f"])
    for rank, row in enumerate(ranked, start=1):
        manifest_rows.append(make_rcp(row, output_root=output_root, prefix=args.prefix, out_dir=out_dir, rank=rank))
    write_csv(manifest_rows, out_dir / "manifest.csv")

    manifest_by_identity = {
        (
            row["image"],
            row["direction"],
            row["candidate_id"],
            row["pretrain_method"],
        ): row
        for row in manifest_rows
    }
    overview = []
    for row in best["operator_error_calibrated"]:
        key = (
            row["image"],
            row["direction"],
            row["candidate_id"],
            row["pretrain_method"],
        )
        if key in manifest_by_identity:
            overview.append(manifest_by_identity[key])
    overview.sort(key=lambda row: IMAGE_ORDER.index(row["image"]) if row["image"] in IMAGE_ORDER else 999)
    make_overview(overview, out_dir / "RCP_best_operator_overview.png")
    write_summary(out_dir, rows)
    print(f"[done] rows={len(rows)} rcp={len(manifest_rows)} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
