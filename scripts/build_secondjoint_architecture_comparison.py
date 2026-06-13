"""Build baseline-centered comparison views for scalar5 second-joint variants."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs/cocoa_like_2d_mechanism"
DEFAULT_INPUT = OUTPUT_ROOT / "secondjoint_scalar5_combined_comparison_20260612" / "comparison_by_case_long.csv"
DEFAULT_OUT = OUTPUT_ROOT / "secondjoint_scalar5_architecture_comparison_20260612"
BASELINE = "single_joint"
VARIANT_ORDER = [
    "single_joint",
    "second_joint",
    "postobjraw_scalar5",
    "postobjraw_pg",
    "postreconpct_keepobj",
    "postreconpct_resetobj",
]
NON_BASELINE_VARIANTS = [variant for variant in VARIANT_ORDER if variant != BASELINE]
VARIANT_LABELS = {
    "single_joint": "baseline: scalar5 single joint",
    "second_joint": "second joint reset Seidel",
    "postobjraw_scalar5": "post object-raw scalar5",
    "postobjraw_pg": "post object-raw p0.1/p99.9 gamma1.5",
    "postreconpct_keepobj": "post recon-percentile keep object",
    "postreconpct_resetobj": "post recon-percentile reset object",
}
METRICS = [
    ("operator_error_calibrated", "op error", False),
    ("aligned_coeff_absolute_error_physical", "coeff abs", False),
    ("aligned_wavefront_error_physical", "wavefront", False),
    ("ssim", "SSIM", True),
    ("nrmse", "NRMSE", False),
]
IMAGES = ["Iksung_beads", "dendrites", "dendrites_dense"]
RMS_VALUES = [0.20, 0.30, 0.40]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt(value: Any) -> str:
    value = parse_float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.5f}" if abs(value) < 1 else f"{value:.4f}"


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(sum(clean) / len(clean)) if clean else math.nan


def rms_tag(value: float) -> str:
    return f"rms{value:g}".replace(".", "p")


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 54:
        raise RuntimeError(f"Expected 54 rows, got {len(rows)}")
    groups: dict[tuple[str, float], set[str]] = defaultdict(set)
    for row in rows:
        groups[(str(row["image"]), round(parse_float(row["target_rms"]), 6))].add(
            str(row["joint_variant"])
        )
    expected = set(VARIANT_ORDER)
    if len(groups) != 9:
        raise RuntimeError(f"Expected 9 image/RMS groups, got {len(groups)}")
    bad = {key: variants for key, variants in groups.items() if variants != expected}
    if bad:
        raise RuntimeError(f"Every image/RMS must have exactly {sorted(expected)}; bad={bad}")


def build_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, float, str], dict[str, Any]]:
    return {
        (
            str(row["image"]),
            round(parse_float(row["target_rms"]), 6),
            str(row["joint_variant"]),
        ): row
        for row in rows
    }


def build_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = build_lookup(rows)
    out: list[dict[str, Any]] = []
    for image in IMAGES:
        for rms in RMS_VALUES:
            baseline = lookup[(image, round(rms, 6), BASELINE)]
            for variant in NON_BASELINE_VARIANTS:
                row = lookup[(image, round(rms, 6), variant)]
                record: dict[str, Any] = {
                    "image": image,
                    "target_rms": rms,
                    "candidate_id": row["candidate_id"],
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "baseline_variant": BASELINE,
                    "pretrain_method": row["pretrain_method"],
                    "baseline_pretrain_method": baseline["pretrain_method"],
                }
                for metric, _label, higher in METRICS:
                    base_value = parse_float(baseline.get(metric))
                    variant_value = parse_float(row.get(metric))
                    delta = variant_value - base_value
                    record[f"baseline_{metric}"] = base_value
                    record[f"variant_{metric}"] = variant_value
                    record[f"delta_{metric}"] = delta
                    record[f"improves_{metric}"] = bool(delta > 0 if higher else delta < 0)
                out.append(record)
    return out


def summary_vs_baseline(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        groups[str(row["variant"])].append(row)
    out: list[dict[str, Any]] = []
    for variant in NON_BASELINE_VARIANTS:
        group = groups[variant]
        record: dict[str, Any] = {
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "count": len(group),
        }
        for metric, _label, _higher in METRICS:
            record[f"baseline_{metric}_mean"] = mean([parse_float(row[f"baseline_{metric}"]) for row in group])
            record[f"variant_{metric}_mean"] = mean([parse_float(row[f"variant_{metric}"]) for row in group])
            record[f"delta_{metric}_mean"] = mean([parse_float(row[f"delta_{metric}"]) for row in group])
            record[f"improves_{metric}_count"] = sum(
                1 for row in group if str(row[f"improves_{metric}"]).lower() == "true" or row[f"improves_{metric}"] is True
            )
        out.append(record)
    return out


def best_variant_by_case(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = build_lookup(rows)
    out: list[dict[str, Any]] = []
    for image in IMAGES:
        for rms in RMS_VALUES:
            group = [lookup[(image, round(rms, 6), variant)] for variant in VARIANT_ORDER]
            record: dict[str, Any] = {"image": image, "target_rms": rms, "variant_count": len(group)}
            for metric, _label, higher in METRICS:
                best = max(group, key=lambda row: parse_float(row.get(metric))) if higher else min(group, key=lambda row: parse_float(row.get(metric)))
                record[f"best_{metric}_variant"] = best["joint_variant"]
                record[f"best_{metric}_value"] = parse_float(best.get(metric))
            out.append(record)
    return out


def winner_counts(best_rows: list[dict[str, Any]], delta_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_ORDER:
        out[variant] = {"variant": variant, "variant_label": VARIANT_LABELS[variant]}
        for metric, _label, _higher in METRICS:
            out[variant][f"best_{metric}_count"] = 0
            out[variant][f"improves_baseline_{metric}_count"] = 0 if variant != BASELINE else ""
    for row in best_rows:
        for metric, _label, _higher in METRICS:
            out[str(row[f"best_{metric}_variant"])][f"best_{metric}_count"] += 1
    for row in delta_summary:
        variant = str(row["variant"])
        for metric, _label, _higher in METRICS:
            out[variant][f"improves_baseline_{metric}_count"] = row[f"improves_{metric}_count"]
    return [out[variant] for variant in VARIANT_ORDER]


def plot_variant_mean_metrics(rows: list[dict[str, Any]], out_dir: Path) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["joint_variant"])].append(row)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(22, 4.8), constrained_layout=True)
    colors = ["#5B6770" if variant == BASELINE else "#4C78A8" for variant in VARIANT_ORDER]
    for ax, (metric, label, higher) in zip(axes, METRICS):
        values = [mean([parse_float(row.get(metric)) for row in groups[variant]]) for variant in VARIANT_ORDER]
        ax.bar(range(len(VARIANT_ORDER)), values, color=colors)
        ax.set_title(f"mean {label}\n{'higher better' if higher else 'lower better'}")
        ax.set_xticks(range(len(VARIANT_ORDER)), VARIANT_ORDER, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            ax.text(idx, value, fmt(value), ha="center", va="bottom", fontsize=7)
    fig.savefig(out_dir / "figures" / "variant_mean_metrics.png", dpi=180)
    plt.close(fig)


def plot_variant_mean_metrics_by_image(rows: list[dict[str, Any]], out_dir: Path) -> None:
    image_dir = out_dir / "figures" / "by_image"
    image_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#5B6770" if variant == BASELINE else "#4C78A8" for variant in VARIANT_ORDER]
    for image in IMAGES:
        image_rows = [row for row in rows if str(row["image"]) == image]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in image_rows:
            groups[str(row["joint_variant"])].append(row)
        fig, axes = plt.subplots(1, len(METRICS), figsize=(22, 4.8), constrained_layout=True)
        fig.suptitle(f"{image}: mean over RMS 0.2 / 0.3 / 0.4", fontsize=13, fontweight="bold")
        for ax, (metric, label, higher) in zip(axes, METRICS):
            values = [
                mean([parse_float(row.get(metric)) for row in groups[variant]])
                for variant in VARIANT_ORDER
            ]
            ax.bar(range(len(VARIANT_ORDER)), values, color=colors)
            ax.set_title(f"mean {label}\n{'higher better' if higher else 'lower better'}")
            ax.set_xticks(range(len(VARIANT_ORDER)), VARIANT_ORDER, rotation=35, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            for idx, value in enumerate(values):
                ax.text(idx, value, fmt(value), ha="center", va="bottom", fontsize=7)
        fig.savefig(image_dir / f"variant_mean_metrics__{image}.png", dpi=180)
        plt.close(fig)


def plot_delta_heatmaps(delta_rows: list[dict[str, Any]], out_dir: Path) -> None:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    case_labels = [f"{image}\n{rms:g}" for image in IMAGES for rms in RMS_VALUES]
    for metric, label, higher in METRICS:
        grid = np.full((len(NON_BASELINE_VARIANTS), len(case_labels)), np.nan, dtype=np.float64)
        for row in delta_rows:
            i = NON_BASELINE_VARIANTS.index(str(row["variant"]))
            j = IMAGES.index(str(row["image"])) * len(RMS_VALUES) + RMS_VALUES.index(round(parse_float(row["target_rms"]), 2))
            grid[i, j] = parse_float(row[f"delta_{metric}"])
        vmax = np.nanmax(np.abs(grid))
        vmax = 1e-12 if not np.isfinite(vmax) or vmax <= 0 else vmax
        cmap = "BrBG" if higher else "BrBG_r"
        fig, ax = plt.subplots(figsize=(14.5, 5.2), constrained_layout=True)
        im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{label}: variant - baseline ({'positive better' if higher else 'negative better'})")
        ax.set_xticks(np.arange(len(case_labels)), case_labels, fontsize=8)
        ax.set_yticks(np.arange(len(NON_BASELINE_VARIANTS)), NON_BASELINE_VARIANTS, fontsize=8)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, fmt(grid[i, j]), ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
        fig.savefig(figures / f"delta_vs_baseline_{metric}.png", dpi=180)
        plt.close(fig)


def plot_winner_heatmap(best_rows: list[dict[str, Any]], out_dir: Path) -> None:
    variant_to_idx = {variant: idx for idx, variant in enumerate(VARIANT_ORDER)}
    metric_names = [metric for metric, _label, _higher in METRICS]
    fig, axes = plt.subplots(1, len(metric_names), figsize=(22, 4.6), constrained_layout=True)
    cmap = plt.get_cmap("tab10", len(VARIANT_ORDER))
    for ax, metric in zip(axes, metric_names):
        grid = np.full((len(IMAGES), len(RMS_VALUES)), np.nan)
        for row in best_rows:
            i = IMAGES.index(str(row["image"]))
            j = RMS_VALUES.index(round(parse_float(row["target_rms"]), 2))
            grid[i, j] = variant_to_idx[str(row[f"best_{metric}_variant"])]
        im = ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=len(VARIANT_ORDER)-0.5, aspect="auto")
        ax.set_title(metric)
        ax.set_xticks(np.arange(len(RMS_VALUES)), [f"{r:g}" for r in RMS_VALUES])
        ax.set_yticks(np.arange(len(IMAGES)), IMAGES)
        for i in range(len(IMAGES)):
            for j in range(len(RMS_VALUES)):
                variant = VARIANT_ORDER[int(grid[i, j])]
                ax.text(j, i, variant.replace("_", "\n"), ha="center", va="center", fontsize=6)
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.01, ticks=range(len(VARIANT_ORDER)))
    cbar.ax.set_yticklabels(VARIANT_ORDER, fontsize=7)
    fig.savefig(out_dir / "figures" / "per_case_winner_heatmap.png", dpi=180)
    plt.close(fig)


def plot_tradeoff(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for idx, variant in enumerate(VARIANT_ORDER):
        group = [row for row in rows if row["joint_variant"] == variant]
        axes[0].scatter(
            [parse_float(row["aligned_coeff_absolute_error_physical"]) for row in group],
            [parse_float(row["ssim"]) for row in group],
            label=variant,
            color=colors(idx),
            alpha=0.8,
        )
        axes[1].scatter(
            [parse_float(row["operator_error_calibrated"]) for row in group],
            [parse_float(row["nrmse"]) for row in group],
            label=variant,
            color=colors(idx),
            alpha=0.8,
        )
    axes[0].set_xlabel("coeff abs lower better")
    axes[0].set_ylabel("SSIM higher better")
    axes[0].set_title("Seidel coeff vs object SSIM")
    axes[1].set_xlabel("operator error lower better")
    axes[1].set_ylabel("NRMSE lower better")
    axes[1].set_title("operator vs object NRMSE")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    fig.savefig(out_dir / "figures" / "seidel_object_tradeoff_scatter.png", dpi=180)
    plt.close(fig)


def find_rcp(row: dict[str, Any]) -> Path:
    source = Path(str(row["source_rcp_dir"]))
    if not source.is_dir():
        source = OUTPUT_ROOT / source.name
    image = str(row["image"])
    rms = rms_tag(parse_float(row["target_rms"]))
    method = str(row["pretrain_method"])
    pattern = source / "rcp_all" / image / "signed_balanced" / rms / f"*__{method}__*.png"
    matches = sorted(pattern.parent.glob(pattern.name))
    if not matches:
        raise RuntimeError(f"Missing RCP for {method} {image} {rms}: {pattern}")
    return matches[0]


def link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)


def organize_rcps(rows: list[dict[str, Any]], out_dir: Path, *, copy: bool) -> None:
    for row in rows:
        src = find_rcp(row)
        variant = str(row["joint_variant"])
        image = str(row["image"])
        rms = rms_tag(parse_float(row["target_rms"]))
        filename = f"{variant}__{src.name}"
        link_or_copy(src, out_dir / "rcp_by_case" / image / rms / filename, copy=copy)
        link_or_copy(src, out_dir / "rcp_by_variant" / variant / image / rms / src.name, copy=copy)


def write_summary_md(out_dir: Path, variant_summary: list[dict[str, Any]], best_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Scalar5 Architecture Comparison",
        "",
        "Baseline: `scalar5_single_joint` = scalar5 pretrain 400 + joint 1000.",
        "",
        "## Mean Metrics",
        "",
        "| variant | op | coeff abs | wavefront | SSIM | NRMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in variant_summary:
        lines.append(
            f"| {row['variant']} | {fmt(row['variant_operator_error_calibrated_mean'])} "
            f"({fmt(row['delta_operator_error_calibrated_mean'])}) | "
            f"{fmt(row['variant_aligned_coeff_absolute_error_physical_mean'])} "
            f"({fmt(row['delta_aligned_coeff_absolute_error_physical_mean'])}) | "
            f"{fmt(row['variant_aligned_wavefront_error_physical_mean'])} "
            f"({fmt(row['delta_aligned_wavefront_error_physical_mean'])}) | "
            f"{fmt(row['variant_ssim_mean'])} ({fmt(row['delta_ssim_mean'])}) | "
            f"{fmt(row['variant_nrmse_mean'])} ({fmt(row['delta_nrmse_mean'])}) |"
        )
    lines.extend(["", "Delta in parentheses is `variant - baseline`; negative is better except SSIM.", ""])
    lines.extend(["## Best Variant Counts", ""])
    for metric, _label, _higher in METRICS:
        counts: dict[str, int] = defaultdict(int)
        for row in best_rows:
            counts[str(row[f"best_{metric}_variant"])] += 1
        chunks = ", ".join(f"{variant}: {counts.get(variant, 0)}" for variant in VARIANT_ORDER)
        lines.append(f"- `{metric}`: {chunks}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `baseline_delta_by_case.csv`",
            "- `summary_vs_baseline_by_variant.csv`",
            "- `winner_counts_vs_baseline.csv`",
            "- `best_variant_by_image_rms.csv`",
            "- `figures/*.png`",
            "- `rcp_by_case/` and `rcp_by_variant/`",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--copy-rcp", action="store_true", help="Copy RCP images instead of making relative symlinks.")
    args = parser.parse_args()

    rows = read_csv(args.input)
    validate_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)

    delta_rows = build_delta_rows(rows)
    variant_summary = summary_vs_baseline(delta_rows)
    best_rows = best_variant_by_case(rows)
    winner_rows = winner_counts(best_rows, variant_summary)

    write_csv(delta_rows, args.output_dir / "baseline_delta_by_case.csv")
    write_csv(variant_summary, args.output_dir / "summary_vs_baseline_by_variant.csv")
    write_csv(winner_rows, args.output_dir / "winner_counts_vs_baseline.csv")
    write_csv(best_rows, args.output_dir / "best_variant_by_image_rms.csv")

    plot_variant_mean_metrics(rows, args.output_dir)
    plot_variant_mean_metrics_by_image(rows, args.output_dir)
    plot_delta_heatmaps(delta_rows, args.output_dir)
    plot_winner_heatmap(best_rows, args.output_dir)
    plot_tradeoff(rows, args.output_dir)
    organize_rcps(rows, args.output_dir, copy=args.copy_rcp)
    write_summary_md(args.output_dir, variant_summary, best_rows)

    print(
        f"[done] rows={len(rows)} delta_rows={len(delta_rows)} "
        f"variants={len(variant_summary)} out={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
