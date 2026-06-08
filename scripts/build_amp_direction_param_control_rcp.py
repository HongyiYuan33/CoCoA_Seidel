#!/usr/bin/env python3
"""Build controlled RCP panels comparing two Seidel parameter settings."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_lambda0_vs_10000_rcp_pairs import (  # noqa: E402
    build_case,
    coeff_ylim,
    collect_display_ranges,
    display_path,
    draw_coeff_card,
    draw_image_panel,
    parse_float,
    parse_vector,
    safe_name,
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def method_name(row: dict[str, Any]) -> str:
    value = str(row.get("seidel_parameterization") or "")
    if value:
        return value
    candidate = str(row.get("candidate_id") or "")
    if "amp_direction_detach_norm" in candidate or "detach" in candidate:
        return "amp_direction_detach_norm"
    if "amp_direction" in candidate:
        return "amp_direction"
    return "direct"


def seed_value(row: dict[str, Any]) -> int:
    return int(float(row.get("seed", 0)))


def abs_share(row: dict[str, Any], coeff_index: int) -> float:
    coeff = abs(parse_vector(row["aligned_seidel_physical"]))
    total = float(coeff.sum())
    return float(coeff[coeff_index] / total) if total > 0 else math.nan


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        indexed[(method_name(row), seed_value(row))] = row
    return indexed


def make_pair_panel(
    *,
    old_row: dict[str, Any],
    new_row: dict[str, Any],
    output_root: Path,
    out_path: Path,
    old_label: str,
    new_label: str,
    title_prefix: str,
) -> dict[str, Any]:
    old_case = build_case(old_row, output_root)
    new_case = build_case(new_row, output_root)
    cases = [old_case, new_case]
    ranges = collect_display_ranges(cases)
    ylimits = coeff_ylim(cases)

    method = method_name(old_row)
    seed = seed_value(old_row)
    image = str(old_row["image"])
    direction = str(old_row["direction"])
    target = parse_float(old_row, "target_wavefront_rms")

    old_w222 = abs_share(old_row, 2)
    new_w222 = abs_share(new_row, 2)
    old_op = parse_float(old_row, "operator_error_calibrated")
    new_op = parse_float(new_row, "operator_error_calibrated")
    old_rms = old_case["aligned_rms"] / max(old_case["gt_rms"], 1e-12)
    new_rms = new_case["aligned_rms"] / max(new_case["gt_rms"], 1e-12)

    old_card_label = f"{old_label} | seed={seed} | W222={old_w222:.3f} | op={old_op:.3f} | RMS={old_rms:.3f}x"
    new_card_label = f"{new_label} | seed={seed} | W222={new_w222:.3f} | op={new_op:.3f} | RMS={new_rms:.3f}x"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20.0, 9.4), dpi=150)
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.16, 1.0],
        height_ratios=[1, 1],
        left=0.027,
        right=0.985,
        top=0.92,
        bottom=0.055,
        wspace=0.045,
        hspace=0.22,
    )
    title = f"{title_prefix} | {method} | seed {seed} | {image} | {direction} | 6D | GT RMS {target:.2f}"
    fig.suptitle(title, fontsize=15.5, fontweight="bold", y=0.978)

    draw_image_panel(fig, outer[0, 0], old_case, ranges)
    draw_coeff_card(fig, outer[0, 1], old_case, ylimits, lambda_label=old_card_label)
    draw_image_panel(fig, outer[1, 0], new_case, ranges)
    draw_coeff_card(fig, outer[1, 1], new_case, ylimits, lambda_label=new_card_label)

    fig.text(0.013, 0.705, old_label, rotation=90, ha="center", va="center", fontsize=11, fontweight="bold")
    fig.text(0.013, 0.285, new_label, rotation=90, ha="center", va="center", fontsize=11, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "method": method,
        "seed": seed,
        "image": image,
        "direction": direction,
        "target_wavefront_rms": target,
        "old_operator_error_calibrated": old_op,
        "new_operator_error_calibrated": new_op,
        "old_w222_abs_share": old_w222,
        "new_w222_abs_share": new_w222,
        "old_aligned_over_gt_rms": old_rms,
        "new_aligned_over_gt_rms": new_rms,
        "old_ssim": parse_float(old_row, "ssim_recon_gain_vs_gt"),
        "new_ssim": parse_float(new_row, "ssim_recon_gain_vs_gt"),
        "path": display_path(out_path),
    }


def make_contact_sheet(manifest_rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not manifest_rows:
        return
    resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    images = []
    for row in sorted(manifest_rows, key=lambda r: (str(r["method"]), int(r["seed"]))):
        path = PROJECT_ROOT / str(row["path"])
        im = Image.open(path).convert("RGB")
        target_width = 850
        target_height = int(round(im.height * target_width / im.width))
        images.append(im.resize((target_width, target_height), resample_filter))
    cols = 2
    gap = 24
    rows_n = int(math.ceil(len(images) / cols))
    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)
    canvas = Image.new("RGB", (cols * cell_w + (cols - 1) * gap, rows_n * cell_h + (rows_n - 1) * gap), "white")
    for idx, im in enumerate(images):
        canvas.paste(im, ((idx % cols) * (cell_w + gap), (idx // cols) * (cell_h + gap)))
    overview = out_dir / "00_contact_sheet"
    overview.mkdir(parents=True, exist_ok=True)
    canvas.save(overview / "param_control_rcp_contact_sheet.png")


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
        "# Seidel Parameter-Control RCP",
        "",
        f"Old CSV: `{args.old_csv}`",
        f"New CSV: `{args.new_csv}`",
        "",
        "Each PNG controls image, direction, RMS, lambda, alpha, parameterization, and seed.",
        "Top row uses old/default parameters; bottom row uses the new parameter set.",
        "",
        "Files:",
    ]
    for row in rows:
        lines.append(
            f"- `{row['path']}` | old_op={row['old_operator_error_calibrated']:.4f}, "
            f"new_op={row['new_operator_error_calibrated']:.4f}"
        )
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-csv", type=Path, required=True)
    parser.add_argument("--new-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cocoa_like_2d_mechanism"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--old-label", default="previous params")
    parser.add_argument("--new-label", default="new params")
    parser.add_argument("--no-contact-sheet", action="store_true")
    args = parser.parse_args()

    old = index_rows(load_rows(args.old_csv))
    new = index_rows(load_rows(args.new_csv))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    methods = sorted({method for method, _seed in old} | {method for method, _seed in new})
    for method in methods:
        seeds = sorted({seed for row_method, seed in old if row_method == method} | {seed for row_method, seed in new if row_method == method})
        for seed in seeds:
            key = (method, seed)
            if key not in old or key not in new:
                missing.append(f"{method} seed={seed}")
                continue
            out_path = args.out_dir / safe_name(method) / f"{safe_name(method)}__seed{seed}__param_control_RCP_vertical.png"
            manifest_rows.append(
                make_pair_panel(
                    old_row=old[key],
                    new_row=new[key],
                    output_root=args.output_root,
                    out_path=out_path,
                    old_label=args.old_label,
                    new_label=args.new_label,
                    title_prefix="Controlled RCP: previous params vs new params",
                )
            )
    write_manifest(manifest_rows, args.out_dir)
    write_readme(manifest_rows, args.out_dir, args)
    if not args.no_contact_sheet:
        make_contact_sheet(manifest_rows, args.out_dir)
    if missing:
        (args.out_dir / "missing_cases.txt").write_text("\n".join(missing) + "\n")
    print(f"[done] wrote {len(manifest_rows)} parameter-control RCP panels to {args.out_dir}")
    if missing:
        print(f"[warn] missing {len(missing)} pairs; see {args.out_dir / 'missing_cases.txt'}")


if __name__ == "__main__":
    main()
