#!/usr/bin/env python3
"""Build RCP panels comparing direct no-RMS recovery to RMS-measure priors."""

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
import numpy as np
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
    rms_label,
    safe_name,
)


ROW_SPECS = [
    ("no_rms_direct", "no_rms_direct", "no RMS | direct coefficient recovery"),
    ("wavefront_prior", "wavefront", "wavefront RMS prior | amp_direction"),
    ("coefficient_prior", "coefficient", "coefficient RMS prior | amp_direction"),
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def seed_value(row: dict[str, Any]) -> int:
    return int(float(row.get("seed", 0)))


def coeff_rms(coeffs: np.ndarray) -> float:
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)[:6]
    return float(np.sqrt(np.mean(coeffs * coeffs)))


def coeff_abs_cv(coeffs: np.ndarray) -> float:
    values = np.abs(np.asarray(coeffs, dtype=np.float64).reshape(-1)[:6])
    mean = float(np.mean(values))
    if mean <= 1e-12:
        return math.nan
    return float(np.std(values, ddof=0) / mean)


def w222_abs_share(coeffs: np.ndarray) -> float:
    values = np.abs(np.asarray(coeffs, dtype=np.float64).reshape(-1)[:6])
    denom = float(np.sum(values))
    if denom <= 1e-12:
        return math.nan
    return float(values[2] / denom)


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, float, int, str], dict[str, Any]]:
    indexed: dict[tuple[str, str, float, int, str], dict[str, Any]] = {}
    for row in rows:
        measure = str(row.get("seidel_rms_prior_measure") or "wavefront")
        if measure not in {"wavefront", "coefficient"}:
            continue
        key = (
            str(row["image"]),
            str(row["direction"]),
            round(float(row["target_wavefront_rms"]), 6),
            seed_value(row),
            measure,
        )
        indexed[key] = row
    return indexed


def index_direct_no_rms_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, float, int], dict[str, Any]]:
    indexed: dict[tuple[str, str, float, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["image"]),
            str(row["direction"]),
            round(float(row["target_wavefront_rms"]), 6),
            seed_value(row),
        )
        indexed[key] = row
    return indexed


def label_for_case(row: dict[str, Any], case: dict[str, Any], base_label: str) -> str:
    aligned = parse_vector(row["aligned_seidel_physical"])
    gt = parse_vector(row["seidel_gt"])
    wf_ratio = case["aligned_rms"] / max(case["gt_rms"], 1e-12)
    coeff_ratio = coeff_rms(aligned) / max(coeff_rms(gt), 1e-12)
    lam = parse_float(row, "lambda", parse_float(row, "seidel_rms_floor_weight", 0.0))
    return (
        f"{base_label} | seed={seed_value(row)} | lambda={lam:.0f} | "
        f"wf={wf_ratio:.3f}x | coeff={coeff_ratio:.3f}x | "
        f"W222={w222_abs_share(aligned):.3f} | CV={coeff_abs_cv(aligned):.3f}"
    )


def case_metrics(row: dict[str, Any], case: dict[str, Any], prefix: str) -> dict[str, Any]:
    aligned = parse_vector(row["aligned_seidel_physical"])
    gt = parse_vector(row["seidel_gt"])
    return {
        f"{prefix}_operator_error_calibrated": parse_float(row, "operator_error_calibrated"),
        f"{prefix}_aligned_wavefront_over_gt": case["aligned_rms"] / max(case["gt_rms"], 1e-12),
        f"{prefix}_aligned_coeff_over_gt": coeff_rms(aligned) / max(coeff_rms(gt), 1e-12),
        f"{prefix}_w222_abs_share": w222_abs_share(aligned),
        f"{prefix}_abs_coeff_cv": coeff_abs_cv(aligned),
        f"{prefix}_ssim": parse_float(row, "ssim_recon_gain_vs_gt"),
        f"{prefix}_nrmse": parse_float(row, "nrmse_recon_gain_vs_gt"),
    }


def make_four_row_panel(
    *,
    no_rms_direct_row: dict[str, Any],
    wavefront_prior_row: dict[str, Any],
    coefficient_prior_row: dict[str, Any],
    output_root: Path,
    out_path: Path,
    title_prefix: str,
) -> dict[str, Any]:
    rows = {
        "no_rms_direct": no_rms_direct_row,
        "wavefront_prior": wavefront_prior_row,
        "coefficient_prior": coefficient_prior_row,
    }
    cases = {name: build_case(row, output_root) for name, row in rows.items()}
    case_list = [cases[name] for name, _measure, _label in ROW_SPECS]
    ranges = collect_display_ranges(case_list)
    ylimits = coeff_ylim(case_list)

    ref = wavefront_prior_row
    image = str(ref["image"])
    direction = str(ref["direction"])
    target = parse_float(ref, "target_wavefront_rms")
    seed = seed_value(ref)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20.0, 13.8), dpi=150)
    outer = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.16, 1.0],
        height_ratios=[1, 1, 1],
        left=0.027,
        right=0.985,
        top=0.94,
        bottom=0.045,
        wspace=0.045,
        hspace=0.24,
    )
    title = (
        f"{title_prefix} | seed {seed} | {image} | {direction} | "
        f"6D | GT wavefront RMS {target:.2f}"
    )
    fig.suptitle(title, fontsize=15.5, fontweight="bold", y=0.982)

    side_y = [0.78, 0.50, 0.22]
    for row_idx, (key, _measure, label) in enumerate(ROW_SPECS):
        draw_image_panel(fig, outer[row_idx, 0], cases[key], ranges)
        draw_coeff_card(fig, outer[row_idx, 1], cases[key], ylimits, lambda_label=label_for_case(rows[key], cases[key], label))
        fig.text(0.013, side_y[row_idx], label, rotation=90, ha="center", va="center", fontsize=10.3, fontweight="bold")

    fig.savefig(out_path)
    plt.close(fig)

    manifest = {
        "image": image,
        "direction": direction,
        "target_wavefront_rms": target,
        "seed": seed,
        "path": display_path(out_path),
    }
    for key, _measure, _label in ROW_SPECS:
        manifest.update(case_metrics(rows[key], cases[key], key))
    return manifest


def make_contact_sheets(manifest_rows: list[dict[str, Any]], out_dir: Path) -> None:
    by_rms: dict[str, list[dict[str, Any]]] = {}
    for row in manifest_rows:
        by_rms.setdefault(rms_label(float(row["target_wavefront_rms"])), []).append(row)

    overview_dir = out_dir / "00_contact_sheets_by_rms"
    overview_dir.mkdir(parents=True, exist_ok=True)
    resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    for label, rows in sorted(by_rms.items()):
        thumbs = []
        for row in sorted(rows, key=lambda item: int(item["seed"])):
            path = PROJECT_ROOT / str(row["path"])
            im = Image.open(path).convert("RGB")
            target_width = 900
            target_height = int(round(im.height * target_width / im.width))
            thumbs.append(im.resize((target_width, target_height), resample_filter))
        if not thumbs:
            continue
        cols = 2
        gap = 24
        rows_n = int(math.ceil(len(thumbs) / cols))
        cell_w = max(im.width for im in thumbs)
        cell_h = max(im.height for im in thumbs)
        canvas = Image.new(
            "RGB",
            (cols * cell_w + (cols - 1) * gap, rows_n * cell_h + (rows_n - 1) * gap),
            "white",
        )
        for idx, im in enumerate(thumbs):
            x = (idx % cols) * (cell_w + gap)
            y = (idx // cols) * (cell_h + gap)
            canvas.paste(im, (x, y))
        canvas.save(overview_dir / f"{label}_contact_sheet.png")


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
        "# No-RMS vs RMS-Measure Prior RCP",
        "",
        f"No-RMS evaluator CSV: `{args.no_rms_csv}`",
        f"RMS-prior evaluator CSV: `{args.prior_csv}`",
        "",
        "Each PNG controls image, direction, GT RMS level, lambda/alpha where applicable, parameterization, and seed.",
        "",
        "Rows:",
        "- no RMS | direct coefficient recovery",
        "- wavefront RMS prior | amp_direction",
        "- coefficient RMS prior | amp_direction",
        "",
        "Folder layout:",
        "- `rms*/dendrites/seed*/direct_no_RMS_vs_wavefront_and_coefficient_RMS_prior__...png`",
        "- `00_contact_sheets_by_rms/rms*_contact_sheet.png`",
        "",
        f"Generated RCP files: {len(rows)}",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['path']}`")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rms-csv", type=Path, required=True)
    parser.add_argument("--prior-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cocoa_like_2d_mechanism"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title-prefix", default="Controlled RCP: no RMS vs wavefront/coefficient RMS prior")
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()

    no_rms = index_direct_no_rms_rows(load_rows(args.no_rms_csv))
    prior = index_rows(load_rows(args.prior_csv))
    pair_keys = sorted(
        {
            key for key in no_rms
        }
        | {
            (image, direction, target, seed)
            for image, direction, target, seed, _measure in set(prior)
        },
        key=lambda item: (item[2], item[0], item[1], item[3]),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for image, direction, target, seed in pair_keys:
        needed = {
            "no_rms_direct": no_rms.get((image, direction, target, seed)),
            "wavefront_prior": prior.get((image, direction, target, seed, "wavefront")),
            "coefficient_prior": prior.get((image, direction, target, seed, "coefficient")),
        }
        if any(row is None for row in needed.values()):
            missing.append(f"{image} {direction} rms={target:.2f} seed={seed}")
            continue
        label = rms_label(target)
        out_path = (
            args.out_dir
            / label
            / safe_name(image)
            / f"seed{seed}"
            / (
                "direct_no_RMS_vs_wavefront_and_coefficient_RMS_prior__"
                f"{safe_name(image)}__{safe_name(direction)}__{label}__seed{seed}__RCP_vertical.png"
            )
        )
        manifest_rows.append(
            make_four_row_panel(
                no_rms_direct_row=needed["no_rms_direct"],
                wavefront_prior_row=needed["wavefront_prior"],
                coefficient_prior_row=needed["coefficient_prior"],
                output_root=args.output_root,
                out_path=out_path,
                title_prefix=args.title_prefix,
            )
        )

    write_manifest(manifest_rows, args.out_dir)
    write_readme(manifest_rows, args.out_dir, args)
    if not args.no_contact_sheets:
        make_contact_sheets(manifest_rows, args.out_dir)
    if missing:
        (args.out_dir / "missing_cases.txt").write_text("\n".join(missing) + "\n")
    print(f"[done] wrote {len(manifest_rows)} no-RMS/RMS-measure RCP panels to {args.out_dir}")
    if missing:
        print(f"[warn] missing {len(missing)} complete groups; see {args.out_dir / 'missing_cases.txt'}")


if __name__ == "__main__":
    main()
