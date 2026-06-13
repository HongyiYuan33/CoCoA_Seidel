"""Combine scalar5 second-joint variants across multiple RCP stats directories."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs/cocoa_like_2d_mechanism"
DEFAULT_OUT_DIR = OUTPUT_ROOT / "secondjoint_scalar5_combined_comparison_20260612"
DEFAULT_RCP_DIRS = [
    OUTPUT_ROOT / "secondjoint_scalar5_4d_size256_three_images_rms020_030_040_pre400_joint1000x2_20260612_rcp_stats",
    OUTPUT_ROOT / "secondjoint_postobjraw_scalar5_4d_size256_three_images_rms020_030_040_pre400_joint1000x2_20260612_rcp_stats",
    OUTPUT_ROOT / "secondjoint_postobjraw_pg_p0p1_p99p9_g1p5_4d_size256_three_images_rms020_030_040_pre400_joint1000x2_20260612_rcp_stats",
    OUTPUT_ROOT / "secondjoint_postreconpct_4d_size256_three_images_rms020_030_040_pre400_joint1000x2_20260612_rcp_stats",
]
METHOD_TO_VARIANT = {
    "scalar5_single_joint": "single_joint",
    "scalar5_second_joint": "second_joint",
    "scalar5_second_joint_postobjraw_scalar5": "postobjraw_scalar5",
    "scalar5_second_joint_postobjraw_pg_p0p1_p99p9_g1p5": "postobjraw_pg",
    "scalar5_second_joint_postreconpct_keepobj": "postreconpct_keepobj",
    "scalar5_second_joint_postreconpct_resetobj": "postreconpct_resetobj",
}
VARIANT_ORDER = [
    "single_joint",
    "second_joint",
    "postobjraw_scalar5",
    "postobjraw_pg",
    "postreconpct_keepobj",
    "postreconpct_resetobj",
]
METRICS = [
    ("operator_error_calibrated", False),
    ("aligned_coeff_absolute_error_physical", False),
    ("aligned_wavefront_error_physical", False),
    ("ssim", True),
    ("nrmse", False),
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
    value = parse_float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.5f}" if abs(value) < 1 else f"{value:.4f}"


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(sum(clean) / len(clean)) if clean else math.nan


def load_rows(rcp_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rcp_dir in rcp_dirs:
        comp_path = rcp_dir / "stats" / "comparison_by_case.csv"
        if not comp_path.is_file():
            continue
        for row in read_csv(comp_path):
            method = str(row.get("pretrain_method", ""))
            variant = METHOD_TO_VARIANT.get(method)
            if variant is None:
                continue
            item: dict[str, Any] = dict(row)
            item["joint_variant"] = variant
            item["source_rcp_dir"] = str(rcp_dir)
            rows.append(item)
    rows.sort(
        key=lambda row: (
            str(row.get("image", "")),
            parse_float(row.get("target_rms")),
            VARIANT_ORDER.index(str(row["joint_variant"])),
        )
    )
    return rows


def summary_by_variant(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["joint_variant"])].append(row)
    out: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        group = groups.get(variant, [])
        if not group:
            continue
        record: dict[str, Any] = {"joint_variant": variant, "count": len(group)}
        for metric, _higher in METRICS:
            record[f"{metric}_mean"] = mean([parse_float(row.get(metric)) for row in group])
        out.append(record)
    return out


def best_by_case(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("image", "")), round(parse_float(row.get("target_rms")), 6))].append(row)
    out: list[dict[str, Any]] = []
    for (image, rms), group in sorted(groups.items()):
        record: dict[str, Any] = {"image": image, "target_rms": rms, "variant_count": len(group)}
        for metric, higher in METRICS:
            valid = [row for row in group if math.isfinite(parse_float(row.get(metric)))]
            if not valid:
                continue
            best = max(valid, key=lambda row: parse_float(row.get(metric))) if higher else min(valid, key=lambda row: parse_float(row.get(metric)))
            record[f"best_{metric}_variant"] = best["joint_variant"]
            record[f"best_{metric}_value"] = parse_float(best.get(metric))
        out.append(record)
    return out


def write_summary_md(out_dir: Path, rows: list[dict[str, Any]], variant_summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Scalar5 Second-Joint Combined Comparison",
        "",
        f"Rows loaded: {len(rows)}",
        "",
        "| variant | count | op | coeff abs | wavefront | SSIM | NRMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_variant = {row["joint_variant"]: row for row in variant_summary}
    for variant in VARIANT_ORDER:
        row = by_variant.get(variant)
        if row is None:
            lines.append(f"| {variant} | 0 | missing | missing | missing | missing | missing |")
            continue
        lines.append(
            f"| {variant} | {row['count']} | "
            f"{fmt(row['operator_error_calibrated_mean'])} | "
            f"{fmt(row['aligned_coeff_absolute_error_physical_mean'])} | "
            f"{fmt(row['aligned_wavefront_error_physical_mean'])} | "
            f"{fmt(row['ssim_mean'])} | {fmt(row['nrmse_mean'])} |"
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rcp-dir", type=Path, action="append", default=None)
    args = parser.parse_args()
    rcp_dirs = args.rcp_dir if args.rcp_dir else DEFAULT_RCP_DIRS
    rows = load_rows(rcp_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "comparison_by_case_long.csv")
    variant_summary = summary_by_variant(rows)
    write_csv(variant_summary, args.output_dir / "summary_by_variant.csv")
    write_csv(best_by_case(rows), args.output_dir / "best_variant_by_image_rms.csv")
    write_summary_md(args.output_dir, rows, variant_summary)
    print(f"[done] rows={len(rows)} variants={len(variant_summary)} out={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
