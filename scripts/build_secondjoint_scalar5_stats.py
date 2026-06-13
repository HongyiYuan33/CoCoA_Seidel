"""Build paired stats for the scalar5 second-joint reset-Seidel experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs/cocoa_like_2d_mechanism"
DEFAULT_PREFIX = "secondjoint_scalar5_4d_size256_three_images_rms020_030_040_pre400_joint1000x2_20260612"
DEFAULT_RCP_DIR = OUTPUT_ROOT / f"{DEFAULT_PREFIX}_rcp_stats"
IMAGES = ["Iksung_beads", "dendrites", "dendrites_dense"]
RMS_VALUES = [0.20, 0.30, 0.40]
VARIANT_ORDER = ["single_joint", "second_joint"]
METHOD_TO_VARIANT = {
    "scalar5_single_joint": "single_joint",
    "scalar5_second_joint": "second_joint",
}
METRICS = [
    ("operator_error_calibrated", "op", False),
    ("aligned_coeff_absolute_error_physical", "coeff_abs", False),
    ("aligned_wavefront_error_physical", "wavefront", False),
    ("ssim", "SSIM", True),
    ("nrmse", "NRMSE", False),
    ("rec_aligned_rms", "rec_aligned_rms", False),
]


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
    numeric = parse_float(value)
    if not math.isfinite(numeric):
        return "nan"
    if abs(numeric) >= 1:
        return f"{numeric:.4f}"
    return f"{numeric:.5f}"


def mean(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return float(sum(clean) / len(clean)) if clean else math.nan


def median(values: list[float]) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return math.nan
    mid = len(clean) // 2
    if len(clean) % 2:
        return float(clean[mid])
    return float((clean[mid - 1] + clean[mid]) / 2.0)


def load_rows(rcp_dir: Path) -> list[dict[str, Any]]:
    comp = read_csv(rcp_dir / "stats" / "comparison_by_case.csv")
    manifest_lookup = {}
    manifest_path = rcp_dir / "manifest.csv"
    if manifest_path.is_file():
        for row in read_csv(manifest_path):
            key = (
                row["image"],
                row["pretrain_method"],
                round(parse_float(row["target_rms"]), 6),
            )
            manifest_lookup[key] = row

    out = []
    for row in comp:
        item: dict[str, Any] = dict(row)
        method = str(row["pretrain_method"])
        variant = METHOD_TO_VARIANT.get(method, method)
        item["joint_variant"] = variant
        key = (row["image"], method, round(parse_float(row["target_rms"]), 6))
        manifest = manifest_lookup.get(key, {})
        item["rec_aligned_rms"] = manifest.get("rec_aligned_rms", "")
        item["rec_raw_rms"] = manifest.get("rec_raw_rms", "")
        item["gt_rms"] = manifest.get("gt_rms", "")
        out.append(item)
    return out


def grouped_summary(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)

    summary = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        record = {name: value for name, value in zip(keys, key)}
        record["count"] = len(group)
        for metric, short, _higher in METRICS:
            vals = [parse_float(row.get(metric)) for row in group]
            record[f"{short}_mean"] = mean(vals)
            record[f"{short}_median"] = median(vals)
        summary.append(record)
    return summary


def paired_delta(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            row["image"],
            round(parse_float(row["target_rms"]), 6),
            row["joint_variant"],
        ): row
        for row in rows
    }
    out = []
    for image in IMAGES:
        for rms in RMS_VALUES:
            single = lookup.get((image, round(rms, 6), "single_joint"))
            second = lookup.get((image, round(rms, 6), "second_joint"))
            if single is None or second is None:
                raise RuntimeError(f"Missing pair image={image} rms={rms}")
            record: dict[str, Any] = {
                "image": image,
                "target_rms": rms,
                "single_method": single["pretrain_method"],
                "second_method": second["pretrain_method"],
            }
            for metric, short, higher in METRICS:
                a = parse_float(single.get(metric))
                b = parse_float(second.get(metric))
                delta = b - a
                record[f"single_{short}"] = a
                record[f"second_{short}"] = b
                record[f"delta_{short}_second_minus_single"] = delta
                record[f"second_improves_{short}"] = bool(delta > 0 if higher else delta < 0)
            out.append(record)
    return out


def make_delta_heatmaps(delta_rows: list[dict[str, Any]], out_dir: Path) -> None:
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("delta_op_second_minus_single", "operator error delta", False),
        ("delta_coeff_abs_second_minus_single", "coeff abs delta", False),
        ("delta_wavefront_second_minus_single", "wavefront delta", False),
        ("delta_SSIM_second_minus_single", "SSIM delta", True),
        ("delta_NRMSE_second_minus_single", "NRMSE delta", False),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9.5, 14.0), constrained_layout=True)
    for ax, (key, title, higher) in zip(axes, panels):
        grid = np.full((len(IMAGES), len(RMS_VALUES)), np.nan, dtype=np.float64)
        for row in delta_rows:
            i = IMAGES.index(str(row["image"]))
            j = RMS_VALUES.index(round(parse_float(row["target_rms"]), 2))
            grid[i, j] = parse_float(row[key])
        vmax = np.nanmax(np.abs(grid))
        vmax = 1e-12 if not np.isfinite(vmax) or vmax <= 0 else vmax
        cmap = "BrBG" if higher else "BrBG_r"
        im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{title}: second - single ({'positive better' if higher else 'negative better'})")
        ax.set_xticks(np.arange(len(RMS_VALUES)), [f"{r:g}" for r in RMS_VALUES])
        ax.set_yticks(np.arange(len(IMAGES)), IMAGES)
        for i in range(len(IMAGES)):
            for j in range(len(RMS_VALUES)):
                ax.text(j, i, fmt(grid[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    fig.savefig(stats_dir / "second_joint_delta_heatmaps.png", dpi=180)
    plt.close(fig)


def make_metric_bars(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    stats_dir = out_dir / "stats"
    variants = [row for row in summary_rows if row.get("joint_variant") in VARIANT_ORDER]
    variants.sort(key=lambda row: VARIANT_ORDER.index(str(row["joint_variant"])))
    x = np.arange(len(variants))
    fig, axes = plt.subplots(1, 5, figsize=(17.0, 4.2), constrained_layout=True)
    for ax, (_metric, short, higher) in zip(axes, METRICS[:5]):
        vals = [parse_float(row[f"{short}_mean"]) for row in variants]
        ax.bar(x, vals, color=["#4C78A8", "#F58518"])
        ax.set_xticks(x, [row["joint_variant"] for row in variants], rotation=20, ha="right")
        ax.set_title(f"mean {short}\n{'higher better' if higher else 'lower better'}")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(stats_dir / "second_joint_variant_metric_means.png", dpi=180)
    plt.close(fig)


def write_summary(
    *,
    out_dir: Path,
    variant_summary: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> None:
    stats_dir = out_dir / "stats"
    by_variant = {row["joint_variant"]: row for row in variant_summary}
    lines = [
        "# Scalar5 Second-Joint Reset-Seidel Experiment",
        "",
        "Setup: scalar=5, three images, signed_balanced, RMS 0.20/0.30/0.40.",
        "Second-joint variant inherits object weights, resets Seidel to zero, and rebuilds Adam optimizers.",
        "",
        "## Mean Metrics",
        "",
        "| variant | op | coeff abs | wavefront | SSIM | NRMSE | rec aligned RMS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        row = by_variant[variant]
        lines.append(
            "| "
            f"{variant} | {fmt(row['op_mean'])} | {fmt(row['coeff_abs_mean'])} | "
            f"{fmt(row['wavefront_mean'])} | {fmt(row['SSIM_mean'])} | "
            f"{fmt(row['NRMSE_mean'])} | {fmt(row['rec_aligned_rms_mean'])} |"
        )

    lines.extend(["", "## Paired Delta Counts", ""])
    for _metric, short, higher in METRICS[:5]:
        key = f"second_improves_{short}"
        count = sum(1 for row in delta_rows if str(row.get(key)).lower() == "true" or row.get(key) is True)
        direction = "higher is better" if higher else "lower is better"
        mean_delta = mean([parse_float(row[f"delta_{short}_second_minus_single"]) for row in delta_rows])
        lines.append(f"- {short}: second improves {count}/{len(delta_rows)} cases; mean delta={fmt(mean_delta)} ({direction}).")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `stats/second_joint_delta_by_case.csv`",
            "- `stats/summary_by_variant.csv`",
            "- `stats/summary_by_rms.csv`",
            "- `stats/summary_by_image.csv`",
            "- `stats/second_joint_delta_heatmaps.png`",
            "- `stats/second_joint_variant_metric_means.png`",
            "- `manifest.csv` and `rcp_all/` from the RCP builder",
        ]
    )
    text = "\n".join(lines) + "\n"
    (stats_dir / "summary.md").write_text(text)
    (stats_dir / "second_joint_summary.md").write_text(text)
    (out_dir / "README.md").write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RCP_DIR)
    args = parser.parse_args()

    rows = load_rows(args.output_dir)
    stats_dir = args.output_dir / "stats"
    variant_summary = grouped_summary(rows, ["joint_variant"])
    rms_summary = grouped_summary(rows, ["target_rms", "joint_variant"])
    image_summary = grouped_summary(rows, ["image", "joint_variant"])
    delta_rows = paired_delta(rows)

    write_csv(variant_summary, stats_dir / "summary_by_variant.csv")
    write_csv(rms_summary, stats_dir / "summary_by_rms.csv")
    write_csv(image_summary, stats_dir / "summary_by_image.csv")
    write_csv(delta_rows, stats_dir / "second_joint_delta_by_case.csv")
    make_delta_heatmaps(delta_rows, args.output_dir)
    make_metric_bars(variant_summary, args.output_dir)
    write_summary(out_dir=args.output_dir, variant_summary=variant_summary, delta_rows=delta_rows)
    print(f"[secondjoint-stats] rows={len(rows)} pairs={len(delta_rows)} out={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
