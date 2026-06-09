"""Build Adam-vs-SGD Seidel optimizer RCP comparison panels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_capacity4d_rcp_stats import (  # noqa: E402
    COEFF_LABELS,
    as_array,
    field_weighted_wavefront_rms,
    load_tensors,
    normalize01,
    parse_float,
    parse_vector,
    percentile01,
    short_float,
    tag_float,
)


DEFAULT_ADAM_RUN = "capacity4d_dirrms_tunedprior_size256_four_images_20260607__baseline"
DEFAULT_SGD_RUN = "seidelopt_sgd4d_tunedprior_size256_four_images_pre400_joint1000_20260608"
IMAGE_ORDER = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]
DIRECTION_ORDER = ["cocoa_signed", "signed_balanced"]
RMS_ORDER = [0.06, 0.20, 0.40]


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


def key_for(row: dict[str, Any]) -> tuple[str, str, float]:
    return (
        str(row["image"]),
        str(row["direction"]),
        round(float(row["target_wavefront_rms"]), 6),
    )


def key_sort(key: tuple[str, str, float]) -> tuple[int, int, int, str, str, float]:
    image, direction, rms = key
    return (
        IMAGE_ORDER.index(image) if image in IMAGE_ORDER else 999,
        DIRECTION_ORDER.index(direction) if direction in DIRECTION_ORDER else 999,
        RMS_ORDER.index(round(rms, 2)) if round(rms, 2) in RMS_ORDER else 999,
        image,
        direction,
        rms,
    )


def keyed(rows: list[dict[str, Any]], label: str) -> dict[tuple[str, str, float], dict[str, Any]]:
    out: dict[tuple[str, str, float], dict[str, Any]] = {}
    for row in rows:
        key = key_for(row)
        if key in out:
            raise ValueError(f"Duplicate {label} row for {key}")
        out[key] = row
    return out


def case_dir(output_root: Path, run: str, row: dict[str, Any]) -> Path:
    return output_root / run / "stage1" / f"{row['image']}__{row['candidate_id']}" / "joint"


def load_case(output_root: Path, run: str, row: dict[str, Any]) -> dict[str, Any]:
    cdir = case_dir(output_root, run, row)
    metrics = json.loads((cdir / "metrics.json").read_text())
    tensors = load_tensors(cdir / "tensors.pt")
    gt = as_array(tensors["sharp_gt"])
    meas = as_array(tensors["measurement_gt"])
    recon = as_array(tensors["sharp_recon"])
    pred = as_array(tensors["measurement_pred"])
    gain = float(metrics.get("best_gain_recon_to_gt", 1.0))
    return {
        "metrics": metrics,
        "gt": gt,
        "meas": meas,
        "recon": recon,
        "pred": pred,
        "recon_gain": recon * gain,
        "error": np.abs(recon * gain - gt),
        "seidel_gt": parse_vector(row["seidel_gt"]),
        "seidel_raw": parse_vector(row["seidel_final"]),
        "seidel_aligned": parse_vector(row["aligned_seidel_physical"]),
    }


def wrapped(text: str, width: int = 82) -> str:
    return "\n".join(wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def draw_variant(
    fig: plt.Figure,
    gridspec: Any,
    *,
    label: str,
    row: dict[str, Any],
    case: dict[str, Any],
) -> None:
    left = gridspec[0].subgridspec(2, 3, wspace=0.08, hspace=0.16)
    image_items = [
        ("Sharp GT", case["gt"], "gray", normalize01),
        ("Measurement", case["meas"], "gray", normalize01),
        (f"{label} recon raw", case["recon"], "gray", normalize01),
        (f"{label} recon pct", case["recon"], "gray", percentile01),
        ("Predicted measurement", case["pred"], "gray", normalize01),
        ("Gain-aligned abs error", case["error"], "magma", percentile01),
    ]
    for idx, (title, arr, cmap, norm_fn) in enumerate(image_items):
        ax = fig.add_subplot(left[idx // 3, idx % 3])
        ax.imshow(norm_fn(arr), cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=8.5, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    right = gridspec[1].subgridspec(1, 2, width_ratios=[0.76, 1.2], wspace=0.22)
    ax_text = fig.add_subplot(right[0, 0])
    ax_text.axis("off")
    gt_rms = field_weighted_wavefront_rms(case["seidel_gt"])
    raw_rms = field_weighted_wavefront_rms(case["seidel_raw"])
    aligned_rms = field_weighted_wavefront_rms(case["seidel_aligned"])
    lines = [
        f"{label}",
        f"op_cal={short_float(parse_float(row, 'operator_error_calibrated'))}",
        f"SSIM={short_float(parse_float(row, 'ssim_recon_gain_vs_gt'))}",
        f"NRMSE={short_float(parse_float(row, 'nrmse_recon_gain_vs_gt'))}",
        f"aligned_WF={short_float(parse_float(row, 'aligned_wavefront_error_physical'))}",
        f"coeff_rel={short_float(parse_float(row, 'aligned_coeff_relative_error_physical'))}",
        f"RMS GT={short_float(gt_rms)}",
        f"rec_aligned={short_float(aligned_rms)}",
        f"rec_raw={short_float(raw_rms)}",
        f"best_phys={row.get('best_physical_transform', '?')}",
        wrapped(str(row.get("candidate_id", "")), width=42),
    ]
    ax_text.text(0.0, 0.98, "\n".join(lines), ha="left", va="top", fontsize=8.4)

    ax_bar = fig.add_subplot(right[0, 1])
    x = np.arange(len(COEFF_LABELS), dtype=np.float64)
    width = 0.34
    ax_bar.bar(x - width / 2, case["seidel_gt"], width, label="GT", color="#55b8b0")
    ax_bar.bar(x + width / 2, case["seidel_aligned"], width, label="aligned recovered", color="#ef7d55")
    ax_bar.scatter(x, case["seidel_raw"], marker="x", color="black", s=24, label="raw", zorder=4)
    ax_bar.axhline(0.0, color="0.55", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(COEFF_LABELS, fontsize=8)
    ax_bar.set_ylabel("coefficient", fontsize=8)
    ax_bar.set_title(f"{label} Seidel coeffs", fontsize=9)
    ax_bar.grid(axis="y", alpha=0.22)
    ax_bar.legend(loc="upper right", fontsize=6.8, frameon=False)
    ax_bar.tick_params(axis="y", labelsize=7.5)


def make_pair_panel(
    *,
    output_root: Path,
    adam_run: str,
    sgd_run: str,
    adam_row: dict[str, Any],
    sgd_row: dict[str, Any],
    comparison_row: dict[str, Any],
    output_dir: Path,
    rank: int,
) -> dict[str, Any]:
    image = str(comparison_row["image"])
    direction = str(comparison_row["direction"])
    rms = float(comparison_row["target_wavefront_rms"])
    candidate = str(sgd_row["candidate_id"])
    adam_case = load_case(output_root, adam_run, adam_row)
    sgd_case = load_case(output_root, sgd_run, sgd_row)

    out_subdir = output_dir / "rcp_pairs" / image / direction / f"rms{tag_float(rms)}"
    out_subdir.mkdir(parents=True, exist_ok=True)
    op_delta = float(comparison_row["delta_operator_error_calibrated"])
    out_path = out_subdir / (
        f"rank{rank:03d}__opdelta{tag_float(op_delta)}__"
        f"{image}__{candidate}__adam_vs_sgd_RCP.png"
    )

    fig = plt.figure(figsize=(21.5, 11.2), dpi=145)
    outer = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.34, 1.0, 1.0],
        width_ratios=[1.08, 1.0],
        left=0.024,
        right=0.988,
        top=0.965,
        bottom=0.055,
        hspace=0.18,
        wspace=0.07,
    )
    ax_header = fig.add_subplot(outer[0, :])
    ax_header.axis("off")
    header = (
        f"Adam vs Seidel-SGD RCP | {image} | {direction} | target RMS={rms:g} | "
        f"delta op={short_float(float(comparison_row['delta_operator_error_calibrated']))} | "
        f"delta SSIM={short_float(float(comparison_row['delta_ssim_recon_gain_vs_gt']))} | "
        f"delta NRMSE={short_float(float(comparison_row['delta_nrmse_recon_gain_vs_gt']))} | "
        f"delta aligned WF={short_float(float(comparison_row['delta_aligned_wavefront_error_physical']))}"
    )
    ax_header.text(0.0, 0.92, header, ha="left", va="top", fontsize=13, fontweight="bold")
    ax_header.text(
        0.0,
        0.46,
        "Lower is better for operator, NRMSE, aligned wavefront and coefficient errors; higher is better for SSIM.",
        ha="left",
        va="top",
        fontsize=9,
    )

    draw_variant(fig, (outer[1, 0], outer[1, 1]), label="Adam baseline", row=adam_row, case=adam_case)
    draw_variant(fig, (outer[2, 0], outer[2, 1]), label="Seidel SGD", row=sgd_row, case=sgd_case)
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "rank": rank,
        "image": image,
        "direction": direction,
        "target_rms": rms,
        "candidate_id": candidate,
        "adam_operator_error_calibrated": comparison_row["adam_operator_error_calibrated"],
        "sgd_operator_error_calibrated": comparison_row["sgd_operator_error_calibrated"],
        "delta_operator_error_calibrated": comparison_row["delta_operator_error_calibrated"],
        "delta_ssim_recon_gain_vs_gt": comparison_row["delta_ssim_recon_gain_vs_gt"],
        "delta_nrmse_recon_gain_vs_gt": comparison_row["delta_nrmse_recon_gain_vs_gt"],
        "delta_aligned_wavefront_error_physical": comparison_row["delta_aligned_wavefront_error_physical"],
        "path": str(out_path.resolve().relative_to(PROJECT_ROOT)),
    }


def make_overview(rows: list[dict[str, Any]], out_path: Path, *, target_width: int = 1700) -> None:
    if not rows:
        return
    resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    images = []
    for row in rows:
        image = Image.open(PROJECT_ROOT / str(row["path"])).convert("RGB")
        height = int(round(image.height * target_width / image.width))
        images.append(image.resize((target_width, height), resample_filter))
    gap = 26
    total_height = sum(image.height for image in images) + gap * max(0, len(images) - 1)
    canvas = Image.new("RGB", (target_width, total_height), "white")
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def selected_overview_rows(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["image"], row["direction"], round(float(row["target_rms"]), 6)): row
        for row in manifest_rows
    }
    preferred = [
        ("Test_figure_1", "cocoa_signed", 0.06),
        ("Test_figure_1", "cocoa_signed", 0.20),
        ("Test_figure_1", "cocoa_signed", 0.40),
        ("dendrites_dense", "signed_balanced", 0.06),
        ("dendrites_dense", "signed_balanced", 0.20),
        ("dendrites_dense", "signed_balanced", 0.40),
    ]
    rows = [by_key[key] for key in preferred if key in by_key]
    if len(rows) >= 6:
        return rows
    return sorted(
        manifest_rows,
        key=lambda row: abs(float(row["delta_operator_error_calibrated"])),
        reverse=True,
    )[:6]


def write_readme(output_dir: Path, manifest_rows: list[dict[str, Any]], overview_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Adam vs Seidel-SGD RCP Comparison",
        "",
        f"Generated pair panels: {len(manifest_rows)}",
        "",
        "Each pair panel has Adam baseline on the first row and Seidel-SGD on the second row.",
        "Deltas are SGD - Adam. Lower is better for operator, NRMSE, aligned wavefront and coefficient errors; higher is better for SSIM.",
        "",
        "Key outputs:",
        "- `manifest.csv`: all 24 pair panels with metric deltas",
        "- `RCP_optimizer_comparison_overview.png`: selected 6-case overview",
        "- `rcp_pairs/<image>/<direction>/<rms>/`: full pair panels",
        "",
        "Overview cases:",
    ]
    for row in overview_rows:
        lines.append(
            f"- {row['image']} / {row['direction']} / RMS={float(row['target_rms']):g}: "
            f"delta_op={short_float(float(row['delta_operator_error_calibrated']))}"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cocoa_like_2d_mechanism"))
    parser.add_argument("--adam-run", default=DEFAULT_ADAM_RUN)
    parser.add_argument("--sgd-run", default=DEFAULT_SGD_RUN)
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=None,
        help="comparison_by_case.csv from compare_seidel_optimizer_sweep.py",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    comparison_csv = args.comparison_csv or args.output_root / f"{args.sgd_run}_adam_vs_sgd" / "comparison_by_case.csv"
    output_dir = args.output_dir or args.output_root / f"{args.sgd_run}_adam_vs_sgd_rcp_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = read_csv(comparison_csv)
    adam_rows = read_csv(args.output_root / args.adam_run / "stage1_operator_eval_dim256" / "seidel_physical_operator_metrics.csv")
    sgd_rows = read_csv(args.output_root / args.sgd_run / "stage1_operator_eval_dim256" / "seidel_physical_operator_metrics.csv")
    adam_by_key = keyed(adam_rows, "adam")
    sgd_by_key = keyed(sgd_rows, "sgd")
    comparison_by_key = keyed(comparison_rows, "comparison")

    manifest_rows: list[dict[str, Any]] = []
    for rank, key in enumerate(sorted(comparison_by_key, key=key_sort), start=1):
        manifest_rows.append(
            make_pair_panel(
                output_root=args.output_root,
                adam_run=args.adam_run,
                sgd_run=args.sgd_run,
                adam_row=adam_by_key[key],
                sgd_row=sgd_by_key[key],
                comparison_row=comparison_by_key[key],
                output_dir=output_dir,
                rank=rank,
            )
        )
    write_csv(manifest_rows, output_dir / "manifest.csv")
    overview_rows = selected_overview_rows(manifest_rows)
    make_overview(overview_rows, output_dir / "RCP_optimizer_comparison_overview.png")
    write_readme(output_dir, manifest_rows, overview_rows)
    print(f"[done] wrote {len(manifest_rows)} RCP pair panels")
    print(f"[done] overview={output_dir / 'RCP_optimizer_comparison_overview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
