#!/usr/bin/env python3
"""Build RCP panels comparing ratio-target control to best amplitude-direction runs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_lambda0_vs_10000_rcp_pairs import (  # noqa: E402
    build_case,
    field_weighted_wavefront_rms,
    make_contact_sheets,
    make_pair_panel,
    parse_float,
    parse_vector,
    rms_label,
    safe_name,
)


IMAGES = ("Iksung_beads", "dendrites", "dendrites_dense")
RMS_VALUES = (0.06, 0.20, 0.40)
VARIANTS = ("amp_direction", "amp_direction_detach_norm")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
    return rows


def filter_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if row.get("image") not in IMAGES:
            continue
        if row.get("direction") != "signed_balanced":
            continue
        if round(float(row["target_wavefront_rms"]), 6) not in {round(v, 6) for v in RMS_VALUES}:
            continue
        output.append(row)
    return output


def control_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
    lookup: dict[tuple[str, float], dict[str, Any]] = {}
    for row in filter_case_rows(rows):
        if not math.isclose(parse_float(row, "lambda"), 1000.0, abs_tol=1e-6):
            continue
        lookup[(row["image"], round(float(row["target_wavefront_rms"]), 6))] = row
    return lookup


def aligned_over_gt_rms(row: dict[str, Any]) -> float:
    gt = parse_vector(row["seidel_gt"])
    aligned = parse_vector(row["aligned_seidel_physical"])
    return field_weighted_wavefront_rms(aligned) / max(field_weighted_wavefront_rms(gt), 1e-12)


def select_best_new(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in filter_case_rows(rows):
        variant = row.get("seidel_parameterization") or row.get("method")
        if variant not in VARIANTS:
            continue
        key = (variant, row["image"], round(float(row["target_wavefront_rms"]), 6))
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: (item[0][2], item[0][1], item[0][0])):
        in_band = [row for row in values if 0.8 <= aligned_over_gt_rms(row) <= 1.2]
        if in_band:
            best = min(in_band, key=lambda row: parse_float(row, "operator_error_calibrated"))
            reason = "lowest_op_with_aligned_rms_0p8_to_1p2"
        else:
            best = min(
                values,
                key=lambda row: (
                    abs(aligned_over_gt_rms(row) - 1.0),
                    parse_float(row, "operator_error_calibrated"),
                ),
            )
            reason = "closest_aligned_rms_to_1"
        out = dict(best)
        out["selected_variant"] = key[0]
        out["selection_reason"] = reason
        selected.append(out)
    return selected


def write_manifest(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (out_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Amplitude-Direction Best RCP Pairs",
        "",
        f"Control evaluator CSV: `{args.control_csv}`",
        f"New evaluator CSV: `{args.new_csv}`",
        "",
        "Top row: existing direct ratio-target alpha=1, lambda=1000.",
        "Bottom row: best amplitude-direction lambda for the same image/RMS/variant.",
        "",
        "Selection rule: lowest operator_error_calibrated among cases with recovered/GT RMS in [0.8, 1.2]; otherwise closest recovered/GT RMS to 1.0.",
        "",
        f"Generated RCP files: {len(rows)}",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['path']}` | variant={row['variant']} | lambda={row['lambda_bottom']:.0f} | {row['selection_reason']}"
        )
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-csv", type=Path, required=True)
    parser.add_argument("--new-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cocoa_like_2d_mechanism"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()

    controls = control_lookup(load_rows(args.control_csv))
    selected = select_best_new(load_rows(args.new_csv))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for new_row in selected:
        image = str(new_row["image"])
        target = round(float(new_row["target_wavefront_rms"]), 6)
        variant = str(new_row["selected_variant"])
        top_row = controls.get((image, target))
        if top_row is None:
            missing.append(f"{image} rms={target:.2f} variant={variant}")
            continue
        top_case = build_case(top_row, args.output_root)
        bottom_case = build_case(new_row, args.output_root)
        label = rms_label(float(target))
        lam = parse_float(new_row, "lambda")
        out_path = (
            args.out_dir
            / label
            / safe_name(image)
            / safe_name(variant)
            / f"control_vs_{safe_name(variant)}_lambda{int(lam)}__{safe_name(image)}__{label}__RCP_vertical.png"
        )
        manifest = make_pair_panel(
            top=top_case,
            bottom=bottom_case,
            out_path=out_path,
            top_label="control ratio target=1x | lambda=1000",
            bottom_label=f"{variant} | lambda_a={lam:g}",
            title_prefix="Amplitude-direction controlled RCP",
        )
        manifest["variant"] = variant
        manifest["selection_reason"] = new_row["selection_reason"]
        manifest["new_run_root"] = new_row.get("run_root", "")
        manifest_rows.append(manifest)

    write_manifest(manifest_rows, args.out_dir)
    write_readme(manifest_rows, args.out_dir, args)
    if not args.no_contact_sheets:
        make_contact_sheets(manifest_rows, args.out_dir)
    if missing:
        (args.out_dir / "missing_cases.txt").write_text("\n".join(missing) + "\n")
    print(f"[done] wrote {len(manifest_rows)} best-pair RCP files to {args.out_dir}")
    if missing:
        print(f"[warn] missing {len(missing)} controls; see {args.out_dir / 'missing_cases.txt'}")


if __name__ == "__main__":
    main()
