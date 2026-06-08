"""Build RCP panels and statistics for the capacity4d direction/RMS sweep."""

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


DEFAULT_PREFIX = "capacity4d_dirrms_tunedprior_size256_four_images_20260607"
IMAGE_ORDER = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]
DIRECTION_ORDER = ["cocoa_signed", "signed_balanced"]
RMS_ORDER = [0.06, 0.20, 0.40]
COEFF_LABELS = ["W040", "W131", "W222", "W220", "W311", "Wd"]
PROFILES = [
    ("baseline", 6, 128, "2,4,6"),
    ("depth_only", 4, 128, "2,4,6"),
    ("width_only", 6, 64, "2,4,6"),
    ("medium", 4, 64, "2"),
    ("noskip_6x128", 6, 128, "none"),
    ("noskip_4x128", 4, 128, "none"),
    ("noskip_6x64", 6, 64, "none"),
    ("noskip_4x64", 4, 64, "none"),
    ("noskip_3x32", 3, 32, "none"),
]
PROFILE_NAMES = [item[0] for item in PROFILES]
PROFILE_META = {name: {"depth": depth, "width": width, "skips": skips} for name, depth, width, skips in PROFILES}


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


def short_float(value: float, digits: int = 4) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


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


def load_rows(output_root: Path, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in PROFILE_NAMES:
        run_dir = output_root / f"{prefix}__{profile}"
        eval_csv = run_dir / "stage1_operator_eval_dim256" / "seidel_physical_operator_metrics.csv"
        for row_idx, row in enumerate(read_csv(eval_csv)):
            enriched = dict(row)
            enriched["profile"] = profile
            enriched["profile_depth"] = PROFILE_META[profile]["depth"]
            enriched["profile_width"] = PROFILE_META[profile]["width"]
            enriched["profile_skips"] = PROFILE_META[profile]["skips"]
            enriched["_source_row_index"] = row_idx
            rows.append(enriched)
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
        out.append(row)
    out.sort(
        key=lambda row: (
            IMAGE_ORDER.index(row["image"]) if row["image"] in IMAGE_ORDER else 999,
            DIRECTION_ORDER.index(row["direction"]) if row["direction"] in DIRECTION_ORDER else 999,
            row["target_rms_f"],
            PROFILE_NAMES.index(row["profile"]),
        )
    )
    return out


def group_key(row: dict[str, Any]) -> tuple[str, str, float]:
    return (str(row["image"]), str(row["direction"]), round(float(row["target_rms_f"]), 6))


def best_rows_by_metric(rows: list[dict[str, Any]], metric_key: str, *, higher: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    best: list[dict[str, Any]] = []
    for key, group in groups.items():
        candidates = [row for row in group if math.isfinite(float(row[metric_key]))]
        if not candidates:
            continue
        chosen = max(candidates, key=lambda row: float(row[metric_key])) if higher else min(candidates, key=lambda row: float(row[metric_key]))
        out = dict(chosen)
        out["best_metric"] = metric_key
        out["best_metric_value"] = chosen[metric_key]
        out["group_image"], out["group_direction"], out["group_target_rms"] = key
        best.append(out)
    best.sort(
        key=lambda row: (
            IMAGE_ORDER.index(row["group_image"]) if row["group_image"] in IMAGE_ORDER else 999,
            DIRECTION_ORDER.index(row["group_direction"]) if row["group_direction"] in DIRECTION_ORDER else 999,
            float(row["group_target_rms"]),
        )
    )
    return best


def grouped_summary(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        summary = {key: value for key, value in zip(keys, key_values)}
        summary.update(
            {
                "rows": len(group),
                "operator_error_calibrated_mean": mean([r["operator_error_calibrated_f"] for r in group]),
                "operator_error_calibrated_min": min(float(r["operator_error_calibrated_f"]) for r in group),
                "operator_error_phys_equiv_mean": mean([r["operator_error_phys_equiv_f"] for r in group]),
                "operator_error_coord_diagnostic_mean": mean([r["operator_error_coord_diagnostic_f"] for r in group]),
                "ssim_mean": mean([r["ssim_f"] for r in group]),
                "ssim_max": max(float(r["ssim_f"]) for r in group),
                "nrmse_mean": mean([r["nrmse_f"] for r in group]),
                "nrmse_min": min(float(r["nrmse_f"]) for r in group),
                "relative_wavefront_error_mean": mean([r["relative_wavefront_error_f"] for r in group]),
                "rec_aligned_rms_mean": mean([r["rec_aligned_rms"] for r in group]),
            }
        )
        out.append(summary)
    return out


def matrix_mean(rows: list[dict[str, Any]], *, row_key: str, col_key: str, value_key: str, row_order: list[Any], col_order: list[Any]) -> np.ndarray:
    matrix = np.full((len(row_order), len(col_order)), np.nan, dtype=np.float64)
    for r_idx, row_value in enumerate(row_order):
        for c_idx, col_value in enumerate(col_order):
            vals = [
                float(row[value_key])
                for row in rows
                if row[row_key] == row_value
                and (
                    round(float(row[col_key]), 6) == round(float(col_value), 6)
                    if isinstance(col_value, float)
                    else row[col_key] == col_value
                )
            ]
            if vals:
                matrix[r_idx, c_idx] = float(np.mean(vals))
    return matrix


def save_heatmap(matrix: np.ndarray, *, row_labels: list[str], col_labels: list[str], title: str, cbar_label: str, out_path: Path, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7, color="white")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def profile_metric_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for profile in PROFILE_NAMES:
        group = [row for row in rows if row["profile"] == profile]
        if not group:
            continue
        summary[profile] = {
            "operator_error_calibrated_mean": mean([row["operator_error_calibrated_f"] for row in group]),
            "ssim_mean": mean([row["ssim_f"] for row in group]),
            "nrmse_mean": mean([row["nrmse_f"] for row in group]),
            "relative_wavefront_error_mean": mean([row["relative_wavefront_error_f"] for row in group]),
            "rec_aligned_rms_mean": mean([row["rec_aligned_rms"] for row in group]),
            "rows": float(len(group)),
        }
    return summary


def capacity_parameter_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = grouped_summary(rows, ["profile"])
    out: list[dict[str, Any]] = []
    for row in base:
        profile = str(row["profile"])
        meta = PROFILE_META[profile]
        enriched = {
            "profile": profile,
            "depth": meta["depth"],
            "width": meta["width"],
            "skips": meta["skips"],
            "skip_count": 0 if meta["skips"] == "none" else len(str(meta["skips"]).split(",")),
        }
        enriched.update(row)
        out.append(enriched)
    out.sort(key=lambda row: (int(row["depth"]), int(row["width"]), str(row["skips"]), str(row["profile"])))
    return out


def capacity_trend_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = profile_metric_summary(rows)
    comparisons = [
        ("depth_down_with_skips_width128", "baseline", "depth_only", "6x128 skips 2,4,6 -> 4x128 skips 2,4,6"),
        ("width_down_with_skips_depth6", "baseline", "width_only", "6x128 skips 2,4,6 -> 6x64 skips 2,4,6"),
        ("depth_width_down_sparse_skips", "baseline", "medium", "6x128 skips 2,4,6 -> 4x64 skip 2"),
        ("remove_skips_6x128", "baseline", "noskip_6x128", "6x128 skips 2,4,6 -> 6x128 no skips"),
        ("remove_skips_4x128", "depth_only", "noskip_4x128", "4x128 skips 2,4,6 -> 4x128 no skips"),
        ("remove_skips_6x64", "width_only", "noskip_6x64", "6x64 skips 2,4,6 -> 6x64 no skips"),
        ("remove_skip_4x64", "medium", "noskip_4x64", "4x64 skip 2 -> 4x64 no skips"),
        ("depth_down_noskip_width128", "noskip_6x128", "noskip_4x128", "6x128 no skips -> 4x128 no skips"),
        ("depth_down_noskip_width64", "noskip_6x64", "noskip_4x64", "6x64 no skips -> 4x64 no skips"),
        ("width_down_noskip_depth6", "noskip_6x128", "noskip_6x64", "6x128 no skips -> 6x64 no skips"),
        ("width_down_noskip_depth4", "noskip_4x128", "noskip_4x64", "4x128 no skips -> 4x64 no skips"),
        ("very_low_capacity_noskip", "noskip_4x64", "noskip_3x32", "4x64 no skips -> 3x32 no skips"),
    ]
    out: list[dict[str, Any]] = []
    for trend, from_profile, to_profile, change in comparisons:
        before = summary[from_profile]
        after = summary[to_profile]
        before_meta = PROFILE_META[from_profile]
        after_meta = PROFILE_META[to_profile]
        op_delta = after["operator_error_calibrated_mean"] - before["operator_error_calibrated_mean"]
        ssim_delta = after["ssim_mean"] - before["ssim_mean"]
        nrmse_delta = after["nrmse_mean"] - before["nrmse_mean"]
        improved_votes = int(op_delta < 0.0) + int(ssim_delta > 0.0) + int(nrmse_delta < 0.0)
        if improved_votes >= 2:
            interpretation = "improves_overall"
        elif improved_votes == 1:
            interpretation = "mixed_or_mostly_worse"
        else:
            interpretation = "worse_overall"
        out.append(
            {
                "trend": trend,
                "capacity_change": change,
                "from_profile": from_profile,
                "from_depth": before_meta["depth"],
                "from_width": before_meta["width"],
                "from_skips": before_meta["skips"],
                "to_profile": to_profile,
                "to_depth": after_meta["depth"],
                "to_width": after_meta["width"],
                "to_skips": after_meta["skips"],
                "from_operator_error_calibrated_mean": before["operator_error_calibrated_mean"],
                "to_operator_error_calibrated_mean": after["operator_error_calibrated_mean"],
                "delta_operator_error_calibrated_mean": op_delta,
                "from_ssim_mean": before["ssim_mean"],
                "to_ssim_mean": after["ssim_mean"],
                "delta_ssim_mean": ssim_delta,
                "from_nrmse_mean": before["nrmse_mean"],
                "to_nrmse_mean": after["nrmse_mean"],
                "delta_nrmse_mean": nrmse_delta,
                "interpretation": interpretation,
            }
        )
    return out


def save_capacity_trend_plot(trend_rows: list[dict[str, Any]], out_path: Path) -> None:
    labels = [str(row["trend"]) for row in trend_rows]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 7.2), sharey=True)
    specs = [
        ("delta_operator_error_calibrated_mean", "Delta operator error", "lower is better", lambda v: v < 0.0),
        ("delta_ssim_mean", "Delta SSIM", "higher is better", lambda v: v > 0.0),
        ("delta_nrmse_mean", "Delta NRMSE", "lower is better", lambda v: v < 0.0),
    ]
    for ax, (key, title, subtitle, improves) in zip(axes, specs):
        vals = np.asarray([float(row[key]) for row in trend_rows], dtype=np.float64)
        colors = ["#4CAF50" if improves(v) else "#D95F02" for v in vals]
        ax.barh(y, vals, color=colors)
        ax.axvline(0.0, color="0.35", linewidth=0.9)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        for yi, val in zip(y, vals):
            ha = "left" if val >= 0.0 else "right"
            pad = 0.001 if key != "delta_ssim_mean" else 0.0008
            ax.text(val + (pad if val >= 0.0 else -pad), yi, f"{val:+.4f}", va="center", ha=ha, fontsize=7)
    axes[0].set_yticks(y, labels=labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Capacity parameter changes: mean metric deltas", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_capacity_trend_notes(stats_dir: Path, rows: list[dict[str, Any]], trend_rows: list[dict[str, Any]]) -> None:
    profile_rows = capacity_parameter_summary(rows)
    best_op = min(profile_rows, key=lambda row: float(row["operator_error_calibrated_mean"]))
    best_ssim = max(profile_rows, key=lambda row: float(row["ssim_mean"]))
    best_nrmse = min(profile_rows, key=lambda row: float(row["nrmse_mean"]))

    def fmt(value: Any, digits: int = 4) -> str:
        return f"{float(value):.{digits}f}"

    lines = [
        "# 容量参数变化趋势",
        "",
        "本文件按具体容量参数汇总趋势：MLP depth、width 和 skip layout。",
        "",
        "## 各 Profile 的具体参数",
        "",
        "| profile | depth | width | skips | 平均 operator | 平均 SSIM | 平均 NRMSE |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in sorted(profile_rows, key=lambda item: PROFILE_NAMES.index(str(item["profile"]))):
        lines.append(
            "| {profile} | {depth} | {width} | {skips} | {op} | {ssim} | {nrmse} |".format(
                profile=row["profile"],
                depth=row["depth"],
                width=row["width"],
                skips=row["skips"],
                op=fmt(row["operator_error_calibrated_mean"]),
                ssim=fmt(row["ssim_mean"]),
                nrmse=fmt(row["nrmse_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## 主要趋势",
            "",
            (
                f"- 平均 operator error 最好的是 `{best_op['profile']}` "
                f"(depth={best_op['depth']}, width={best_op['width']}, skips={best_op['skips']}), "
                f"mean={fmt(best_op['operator_error_calibrated_mean'])}。"
            ),
            (
                f"- 平均 SSIM 最好的是 `{best_ssim['profile']}` "
                f"(depth={best_ssim['depth']}, width={best_ssim['width']}, skips={best_ssim['skips']}), "
                f"mean={fmt(best_ssim['ssim_mean'])}。"
            ),
            (
                f"- 平均 NRMSE 最好的是 `{best_nrmse['profile']}` "
                f"(depth={best_nrmse['depth']}, width={best_nrmse['width']}, skips={best_nrmse['skips']}), "
                f"mean={fmt(best_nrmse['nrmse_mean'])}。"
            ),
            "- delta 表示从 `from_profile` 变到 `to_profile` 后的均值变化；operator/NRMSE 越负越好，SSIM 越正越好。",
            "",
            "## 成对容量变化",
            "",
            "| trend | 容量变化 | delta operator | delta SSIM | delta NRMSE | 解读 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in trend_rows:
        lines.append(
            "| {trend} | {change} | {op:+.4f} | {ssim:+.4f} | {nrmse:+.4f} | {interp} |".format(
                trend=row["trend"],
                change=row["capacity_change"],
                op=float(row["delta_operator_error_calibrated_mean"]),
                ssim=float(row["delta_ssim_mean"]),
                nrmse=float(row["delta_nrmse_mean"]),
                interp=row["interpretation"],
            )
        )
    lines.extend(
        [
            "",
            "配套文件：",
            "- `capacity_parameter_summary.csv`",
            "- `capacity_parameter_trends.csv`",
            "- `capacity_parameter_trend_deltas.png`",
            "",
        ]
    )
    (stats_dir / "capacity_parameter_trends.md").write_text("\n".join(lines))


def write_stats_readme(stats_dir: Path) -> None:
    lines = [
        "# Capacity4D 统计文件",
        "",
        "重点文件：",
        "- `capacity_parameter_trends.md`: 按 depth/width/skips 写出的容量变化趋势说明",
        "- `capacity_parameter_summary.csv`: 每个 capacity profile 的具体参数和均值指标",
        "- `capacity_parameter_trends.csv`: 成对容量变化的 operator/SSIM/NRMSE delta",
        "- `capacity_parameter_trend_deltas.png`: 成对容量变化的可视化趋势图",
        "- `profile_metric_means.png`: 每个 profile 的平均 operator、SSIM、NRMSE",
        "- `winner_counts_by_profile.png`: 每个 profile 在不同指标下的 best-case 次数",
        "- `best_by_operator_error_calibrated.csv`, `best_by_ssim.csv`, `best_by_nrmse.csv`: 每个 image/direction/RMS 的最优行",
        "",
    ]
    (stats_dir / "README.md").write_text("\n".join(lines))


def make_stat_plots(rows: list[dict[str, Any]], out_dir: Path) -> None:
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, stats_dir / "combined_capacity4d_metrics.csv")
    write_csv(grouped_summary(rows, ["profile"]), stats_dir / "profile_summary.csv")
    write_csv(grouped_summary(rows, ["image"]), stats_dir / "image_summary.csv")
    write_csv(grouped_summary(rows, ["direction"]), stats_dir / "direction_summary.csv")
    write_csv(grouped_summary(rows, ["target_rms_f"]), stats_dir / "rms_summary.csv")
    write_csv(grouped_summary(rows, ["profile", "target_rms_f"]), stats_dir / "profile_rms_summary.csv")
    write_csv(grouped_summary(rows, ["profile", "image"]), stats_dir / "profile_image_summary.csv")
    parameter_summary = capacity_parameter_summary(rows)
    trend_rows = capacity_trend_rows(rows)
    write_csv(parameter_summary, stats_dir / "capacity_parameter_summary.csv")
    write_csv(trend_rows, stats_dir / "capacity_parameter_trends.csv")
    write_capacity_trend_notes(stats_dir, rows, trend_rows)
    write_stats_readme(stats_dir)

    op_best = best_rows_by_metric(rows, "operator_error_calibrated_f", higher=False)
    ssim_best = best_rows_by_metric(rows, "ssim_f", higher=True)
    nrmse_best = best_rows_by_metric(rows, "nrmse_f", higher=False)
    write_csv(op_best, stats_dir / "best_by_operator_error_calibrated.csv")
    write_csv(ssim_best, stats_dir / "best_by_ssim.csv")
    write_csv(nrmse_best, stats_dir / "best_by_nrmse.csv")
    winner_rows = []
    for label, best in [("operator_error_calibrated", op_best), ("ssim", ssim_best), ("nrmse", nrmse_best)]:
        counts = Counter(row["profile"] for row in best)
        for profile in PROFILE_NAMES:
            winner_rows.append({"metric": label, "profile": profile, "wins": counts.get(profile, 0)})
    write_csv(winner_rows, stats_dir / "winner_counts_by_metric.csv")

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), sharex=True)
    metric_specs = [
        ("operator_error_calibrated_f", "Mean operator error", "lower is better", "#3B82B8"),
        ("ssim_f", "Mean SSIM gain", "higher is better", "#7CB342"),
        ("nrmse_f", "Mean NRMSE gain", "lower is better", "#F58518"),
    ]
    xs = np.arange(len(PROFILE_NAMES))
    for ax, (metric, title, ylabel, color) in zip(axes, metric_specs):
        vals = [mean([row[metric] for row in rows if row["profile"] == profile]) for profile in PROFILE_NAMES]
        ax.bar(xs, vals, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs, labels=PROFILE_NAMES, rotation=42, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for x, val in zip(xs, vals):
            ax.text(x, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Capacity profile means across 216 cases", y=1.02)
    fig.tight_layout()
    fig.savefig(stats_dir / "profile_metric_means.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    save_capacity_trend_plot(trend_rows, stats_dir / "capacity_parameter_trend_deltas.png")

    for value_key, label, filename, cmap in [
        ("operator_error_calibrated_f", "mean operator_error_calibrated", "operator_error_heatmap_profile_rms.png", "viridis_r"),
        ("ssim_f", "mean SSIM_gain", "ssim_heatmap_profile_rms.png", "viridis"),
        ("nrmse_f", "mean NRMSE_gain", "nrmse_heatmap_profile_rms.png", "viridis_r"),
    ]:
        matrix = matrix_mean(
            rows,
            row_key="profile",
            col_key="target_rms_f",
            value_key=value_key,
            row_order=PROFILE_NAMES,
            col_order=RMS_ORDER,
        )
        save_heatmap(
            matrix,
            row_labels=PROFILE_NAMES,
            col_labels=[str(v) for v in RMS_ORDER],
            title=label,
            cbar_label=label,
            out_path=stats_dir / filename,
            cmap=cmap,
        )

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    width = 0.24
    for idx, (label, best, color) in enumerate(
        [
            ("operator", op_best, "#3B82B8"),
            ("SSIM", ssim_best, "#7CB342"),
            ("NRMSE", nrmse_best, "#F58518"),
        ]
    ):
        counts = Counter(row["profile"] for row in best)
        vals = [counts.get(profile, 0) for profile in PROFILE_NAMES]
        ax.bar(xs + (idx - 1) * width, vals, width, label=label, color=color)
    ax.set_xticks(xs, labels=PROFILE_NAMES, rotation=42, ha="right", fontsize=8)
    ax.set_ylabel("best-case wins out of 24")
    ax.set_title("Profile wins by best metric across image/direction/RMS groups")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(stats_dir / "winner_counts_by_profile.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(PROFILE_NAMES)))
    for profile, color in zip(PROFILE_NAMES, colors):
        group = [row for row in rows if row["profile"] == profile]
        ax.scatter(
            [row["operator_error_calibrated_f"] for row in group],
            [row["ssim_f"] for row in group],
            s=28,
            alpha=0.70,
            label=profile,
            color=color,
        )
    ax.set_xlabel("operator_error_calibrated")
    ax.set_ylabel("SSIM gain vs GT")
    ax.set_title("Operator error vs reconstruction SSIM")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(stats_dir / "operator_error_vs_ssim_scatter.png", dpi=170)
    plt.close(fig)


def case_dir_for(output_root: Path, prefix: str, row: dict[str, Any]) -> Path:
    return output_root / f"{prefix}__{row['profile']}" / "stage1" / f"{row['image']}__{row['candidate_id']}" / "joint"


def make_rcp(row: dict[str, Any], *, output_root: Path, prefix: str, out_dir: Path, rank: int) -> dict[str, Any]:
    case_dir = case_dir_for(output_root, prefix, row)
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
    profile = str(row["profile"])
    image = str(row["image"])
    direction = str(row["direction"])
    target_rms = float(row["target_rms_f"])
    candidate = str(row["candidate_id"])

    out_subdir = out_dir / "rcp_all" / image / direction / rms_tag(target_rms)
    out_subdir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"rank{rank:03d}__opcal{tag_float(row['operator_error_calibrated_f'])}__"
        f"{profile}__{image}__{candidate}__four_panel_plus_coeff_similarity.png"
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
    title = f"Capacity4D RCP rank {rank:03d} | {image} | {direction} | RMS={target_rms:g} | {profile}"
    lines = [
        f"op_cal={short_float(row['operator_error_calibrated_f'])} | "
        f"phys={short_float(row['operator_error_phys_equiv_f'])} | "
        f"coord={short_float(row['operator_error_coord_diagnostic_f'])}",
        (
            f"RMS waves: GT={short_float(row['gt_rms'])} | "
            f"rec_aligned={short_float(row['rec_aligned_rms'])} | rec_raw={short_float(row['rec_raw_rms'])}"
        ),
        (
            f"profile: depth={row['profile_depth']} width={row['profile_width']} skips={row['profile_skips']} | "
            f"best_phys={row.get('best_physical_transform', '?')}"
        ),
        (
            f"aligned_WFrel={short_float(row['aligned_wavefront_error_physical_f'])} | "
            f"relWF={short_float(row['relative_wavefront_error_f'])} | "
            f"actual_RMS={short_float(parse_float(row, 'actual_wavefront_rms'))}"
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
        "direction": direction,
        "target_rms": target_rms,
        "candidate_id": candidate,
        "profile": profile,
        "operator_error_calibrated": row["operator_error_calibrated_f"],
        "operator_error_phys_equiv": row["operator_error_phys_equiv_f"],
        "operator_error_coord_diagnostic": row["operator_error_coord_diagnostic_f"],
        "ssim": row["ssim_f"],
        "nrmse": row["nrmse_f"],
        "gt_rms": row["gt_rms"],
        "rec_aligned_rms": row["rec_aligned_rms"],
        "rec_raw_rms": row["rec_raw_rms"],
        "best_physical_transform": row.get("best_physical_transform", ""),
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


def write_readme(out_dir: Path, manifest_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Capacity4D Direction/RMS RCP And Stats",
        "",
        f"Generated RCP panels: {len(manifest_rows)}",
        "",
        "Key outputs:",
        "- `manifest.csv`: all RCP panel paths and metrics ranked by operator error",
        "- `stats/`: summary CSVs and statistical plots",
        "- `stats/capacity_parameter_trends.md`: concrete depth/width/skip trend notes",
        "- `stats/capacity_parameter_trend_deltas.png`: capacity-change delta plot",
        "- `RCP_best_operator_overview.png`: best operator-error RCP per image/direction/RMS group",
        "- `rcp_all/<image>/<direction>/<rms>/`: all full RCP panels",
        "",
        "RCP panels report GT RMS, raw recovered RMS, and aligned recovered RMS using `field_weighted_wavefront_rms`.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cocoa_like_2d_mechanism"))
    parser.add_argument("--run-prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cocoa_like_2d_mechanism") / f"{DEFAULT_PREFIX}_rcp_stats",
    )
    parser.add_argument("--skip-rcp", action="store_true")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = enrich_rows(load_rows(args.output_root, args.run_prefix))
    rows_ranked = sorted(rows, key=lambda row: (row["operator_error_calibrated_f"], row["image"], row["direction"], row["target_rms_f"], row["profile"]))
    make_stat_plots(rows, out_dir)

    manifest_rows: list[dict[str, Any]] = []
    if not args.skip_rcp:
        for rank, row in enumerate(rows_ranked, start=1):
            manifest_rows.append(make_rcp(row, output_root=args.output_root, prefix=args.run_prefix, out_dir=out_dir, rank=rank))
        write_csv(manifest_rows, out_dir / "manifest.csv")
        best_operator_rows = best_rows_by_metric(rows, "operator_error_calibrated_f", higher=False)
        manifest_by_key = {
            (row["image"], row["direction"], round(float(row["target_rms"]), 6), row["profile"]): row
            for row in manifest_rows
        }
        overview_rows = [
            manifest_by_key[(row["image"], row["direction"], round(float(row["target_rms_f"]), 6), row["profile"])]
            for row in best_operator_rows
        ]
        make_overview(overview_rows, out_dir / "RCP_best_operator_overview.png")
        write_readme(out_dir, manifest_rows)
    elif (out_dir / "manifest.csv").is_file():
        write_readme(out_dir, read_csv(out_dir / "manifest.csv"))

    print(f"[done] rows={len(rows)}")
    print(f"[done] stats={out_dir / 'stats'}")
    if manifest_rows:
        print(f"[done] rcp={len(manifest_rows)}")
        print(f"[done] overview={out_dir / 'RCP_best_operator_overview.png'}")


if __name__ == "__main__":
    main()
