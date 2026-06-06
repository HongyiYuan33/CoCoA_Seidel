#!/usr/bin/env python3
"""Build RCP panels comparing amplitude-direction seed sensitivity cases."""

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


METHODS = ("amp_direction", "amp_direction_detach_norm")


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def method_name(row: dict[str, Any]) -> str:
    value = str(row.get("seidel_parameterization") or "")
    if value:
        return value
    candidate = str(row.get("candidate_id") or "")
    return "amp_direction_detach_norm" if "detach" in candidate else "amp_direction"


def seed_value(row: dict[str, Any]) -> int:
    return int(float(row.get("seed", 0)))


def abs_share(row: dict[str, Any], coeff_index: int, key: str = "aligned_seidel_physical") -> float:
    coeff = abs(parse_vector(row[key]))
    total = float(coeff.sum())
    return float(coeff[coeff_index] / total) if total > 0 else math.nan


def make_seed_panel(
    *,
    rows: list[dict[str, Any]],
    output_root: Path,
    out_path: Path,
    title_prefix: str,
) -> dict[str, Any]:
    rows = sorted(rows, key=seed_value)
    cases = [build_case(row, output_root) for row in rows]
    ranges = collect_display_ranges(cases)
    ylimits = coeff_ylim(cases)
    first = rows[0]
    method = method_name(first)
    target = parse_float(first, "target_wavefront_rms")
    image = str(first["image"])
    direction = str(first["direction"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20.0, 17.0), dpi=150)
    outer = fig.add_gridspec(
        len(cases),
        2,
        width_ratios=[1.16, 1.0],
        height_ratios=[1] * len(cases),
        left=0.027,
        right=0.985,
        top=0.935,
        bottom=0.04,
        wspace=0.045,
        hspace=0.22,
    )
    title = f"{title_prefix} | {method} | {image} | {direction} | 6D | GT RMS {target:.2f}"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.982)

    top = 0.935
    bottom = 0.04
    usable = top - bottom
    for idx, (row, case) in enumerate(zip(rows, cases)):
        seed = seed_value(row)
        op = parse_float(row, "operator_error_calibrated")
        w222 = abs_share(row, 2)
        aligned_ratio = case["aligned_rms"] / max(case["gt_rms"], 1e-12)
        label = f"seed={seed} | W222 share={w222:.3f} | op={op:.3f} | RMS={aligned_ratio:.3f}x"
        draw_image_panel(fig, outer[idx, 0], case, ranges)
        draw_coeff_card(fig, outer[idx, 1], case, ylimits, lambda_label=label)
        fig.text(
            0.013,
            bottom + usable * (len(cases) - idx - 0.5) / len(cases),
            f"seed {seed}",
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    fig.savefig(out_path)
    plt.close(fig)

    output = {
        "method": method,
        "image": image,
        "direction": direction,
        "target_wavefront_rms": target,
        "path": display_path(out_path),
    }
    for row, case in zip(rows, cases):
        prefix = f"seed{seed_value(row)}"
        output[f"{prefix}_operator_error_calibrated"] = parse_float(row, "operator_error_calibrated")
        output[f"{prefix}_w222_abs_share"] = abs_share(row, 2)
        output[f"{prefix}_aligned_over_gt_rms"] = case["aligned_rms"] / max(case["gt_rms"], 1e-12)
        output[f"{prefix}_ssim"] = parse_float(row, "ssim_recon_gain_vs_gt")
    return output


def make_contact_sheet(manifest_rows: list[dict[str, Any]], out_dir: Path) -> None:
    images = []
    resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    for row in manifest_rows:
        path = PROJECT_ROOT / str(row["path"])
        im = Image.open(path).convert("RGB")
        target_width = 900
        target_height = int(round(im.height * target_width / im.width))
        images.append(im.resize((target_width, target_height), resample_filter))
    if not images:
        return
    gap = 24
    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)
    canvas = Image.new("RGB", (cell_w, len(images) * cell_h + (len(images) - 1) * gap), "white")
    for idx, im in enumerate(images):
        canvas.paste(im, (0, idx * (cell_h + gap)))
    overview_dir = out_dir / "00_contact_sheet"
    overview_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(overview_dir / "amp_direction_seed_rcp_contact_sheet.png")


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
        "# Amp-Direction Seed RCP Comparison",
        "",
        f"Evaluator CSV: `{args.csv}`",
        "",
        "Each PNG stacks seed 0-3 vertically for one parameterization.",
        "Rows share image display ranges and coefficient y-limits inside each PNG.",
        "",
        "Files:",
    ]
    for row in rows:
        lines.append(f"- `{row['path']}`")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cocoa_like_2d_mechanism"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-contact-sheet", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = [row for row in rows if method_name(row) == method]
        if not method_rows:
            continue
        image = safe_name(str(method_rows[0]["image"]))
        target = str(float(method_rows[0]["target_wavefront_rms"])).replace(".", "p")
        out_path = args.out_dir / f"{safe_name(method)}__{image}__rms{target}__seed_RCP_vertical.png"
        manifest_rows.append(
            make_seed_panel(
                rows=method_rows,
                output_root=args.output_root,
                out_path=out_path,
                title_prefix="RCP seed comparison",
            )
        )
    write_manifest(manifest_rows, args.out_dir)
    write_readme(manifest_rows, args.out_dir, args)
    if not args.no_contact_sheet:
        make_contact_sheet(manifest_rows, args.out_dir)
    print(f"[done] wrote {len(manifest_rows)} seed RCP panels to {args.out_dir}")


if __name__ == "__main__":
    main()
