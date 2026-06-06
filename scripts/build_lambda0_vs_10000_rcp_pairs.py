#!/usr/bin/env python3
"""Build controlled lambda=0 vs lambda=10000 RCP pair panels."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import (  # noqa: E402
    field_weighted_wavefront_rms,
)


COEFF_LABELS = ["W040", "W131", "W222", "W220", "W311", "Wd"]
IMAGE_ORDER = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]
DIRECTION_ORDER = ["cocoa_signed", "signed_balanced"]


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, int, float, bool)):
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
        raise ValueError("Missing Seidel vector")
    arr = np.asarray([float(x) for x in parsed], dtype=np.float64)
    if arr.shape != (6,):
        raise ValueError(f"Expected 6D vector, got {arr.shape}")
    return arr


def parse_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value in (None, ""):
        return float(default)
    return float(value)


def short_float(value: float, digits: int = 4) -> str:
    if math.isnan(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def rms_label(value: float) -> str:
    return f"rms{value:.2f}".replace(".", "p")


def safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def wrap_line(text: str, width: int = 98) -> str:
    return "\n".join(wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def short_run_name(value: Any) -> str:
    text = str(value)
    marker = "20260605_"
    if marker in text:
        return text.split(marker, 1)[1]
    return text


def load_tensors(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def scale01(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)


def finite_percentile(values: list[np.ndarray], lo: float, hi: float) -> tuple[float, float]:
    flat = np.concatenate([np.ravel(v.astype(np.float32)) for v in values])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    v0, v1 = np.percentile(flat, [lo, hi])
    if float(v1) <= float(v0):
        return float(np.min(flat)), float(np.max(flat) + 1e-6)
    return float(v0), float(v1)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_case(row: dict[str, Any], output_root: Path) -> dict[str, Any]:
    metrics_path = output_root / str(row["metrics_path"])
    tensors_path = metrics_path.parent / "tensors.pt"
    metrics = json.loads(metrics_path.read_text())
    tensors = load_tensors(tensors_path)

    gt = as_array(tensors["sharp_gt"])
    meas = as_array(tensors["measurement_gt"])
    recon = as_array(tensors["sharp_recon"])
    pred = as_array(tensors["measurement_pred"])
    gain = float(metrics.get("best_gain_recon_to_gt", 1.0))
    err = np.abs(gain * recon - gt)

    seidel_gt = parse_vector(row["seidel_gt"])
    seidel_raw = parse_vector(row["seidel_final"])
    seidel_aligned = parse_vector(row["aligned_seidel_physical"])

    return {
        "row": row,
        "metrics": metrics,
        "gt": gt,
        "measurement": meas,
        "recon": recon,
        "pred": pred,
        "err": err,
        "seidel_gt": seidel_gt,
        "seidel_raw": seidel_raw,
        "seidel_aligned": seidel_aligned,
        "gt_rms": field_weighted_wavefront_rms(seidel_gt),
        "raw_rms": field_weighted_wavefront_rms(seidel_raw),
        "aligned_rms": field_weighted_wavefront_rms(seidel_aligned),
        "metrics_path": metrics_path,
        "tensors_path": tensors_path,
    }


def collect_display_ranges(cases: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    gt_recon_values = [case["gt"] for case in cases] + [np.clip(case["recon"], 0.0, 1.0) for case in cases]
    meas_values = [case["measurement"] for case in cases] + [case["pred"] for case in cases]
    err_values = [case["err"] for case in cases]
    return {
        "sharp": (0.0, 1.0),
        "measurement": finite_percentile(meas_values, 0.5, 99.7),
        "error": finite_percentile(err_values, 0.5, 99.7),
        "recon": finite_percentile(gt_recon_values, 0.0, 100.0),
    }


def coeff_ylim(cases: list[dict[str, Any]]) -> tuple[float, float]:
    values = []
    for case in cases:
        values.extend(case["seidel_gt"].tolist())
        values.extend(case["seidel_raw"].tolist())
        values.extend(case["seidel_aligned"].tolist())
    arr = np.asarray(values, dtype=np.float64)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return -0.1, 0.1
    pad = max(0.035, 0.12 * (hi - lo))
    return lo - pad, hi + pad


def draw_image_panel(
    fig: plt.Figure,
    spec: Any,
    case: dict[str, Any],
    ranges: dict[str, tuple[float, float]],
) -> None:
    sub = spec.subgridspec(1, 5, wspace=0.055)
    panels = [
        ("GT", case["gt"], "gray", ranges["sharp"]),
        ("Measurement", case["measurement"], "gray", ranges["measurement"]),
        ("Joint recon", np.clip(case["recon"], 0.0, 1.0), "gray", ranges["recon"]),
        ("Pred meas", case["pred"], "gray", ranges["measurement"]),
        ("Gain abs err", case["err"], "magma", ranges["error"]),
    ]
    for idx, (title, arr, cmap, (vmin, vmax)) in enumerate(panels):
        ax = fig.add_subplot(sub[0, idx])
        ax.imshow(scale01(arr, vmin, vmax), cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=8, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])


def draw_coeff_card(
    fig: plt.Figure,
    spec: Any,
    case: dict[str, Any],
    ylimits: tuple[float, float],
    *,
    lambda_label: str,
) -> None:
    row = case["row"]
    metrics = case["metrics"]
    sub = spec.subgridspec(3, 1, height_ratios=[0.62, 1.8, 0.62], hspace=0.18)

    op_cal = parse_float(row, "operator_error_calibrated")
    phys = parse_float(row, "operator_error_phys_equiv")
    coord = parse_float(row, "operator_error_coord_diagnostic")
    ssim = parse_float(row, "ssim_recon_gain_vs_gt")
    nrmse = parse_float(row, "nrmse_recon_gain_vs_gt")
    ratio = parse_float(row, "wavefront_recovered_over_gt_rms")
    target = parse_float(row, "target_wavefront_rms")

    ax_text = fig.add_subplot(sub[0, 0])
    ax_text.axis("off")
    title = (
        f"{lambda_label} | {row['image']} | {row['direction']} | "
        f"GT RMS {target:.2f}"
    )
    lines = [
        f"op_cal={short_float(op_cal)} | phys={short_float(phys)} | coord={short_float(coord)}",
        (
            f"RMS waves: GT={short_float(case['gt_rms'])} | raw={short_float(case['raw_rms'])} | "
            f"aligned={short_float(case['aligned_rms'])} | raw/GT={short_float(ratio, 3)}"
        ),
        (
            f"best_phys={row.get('best_physical_transform', '?')} | "
            f"SSIM={short_float(ssim)} | NRMSE={short_float(nrmse)} | "
            f"gain={short_float(float(metrics.get('best_gain_recon_to_gt', math.nan)), 3)}"
        ),
        wrap_line(
            f"id={row.get('candidate_id', '')} | run={short_run_name(row.get('run_root', ''))}",
            width=96,
        ),
    ]
    ax_text.text(0.0, 0.98, title, ha="left", va="top", fontsize=10, fontweight="bold")
    ax_text.text(0.0, 0.70, "\n".join(lines), ha="left", va="top", fontsize=7.3)

    ax_bar = fig.add_subplot(sub[1, 0])
    x = np.arange(len(COEFF_LABELS), dtype=np.float64)
    width = 0.34
    ax_bar.bar(x - width / 2, case["seidel_gt"], width, label="GT", color="#55b8b0")
    ax_bar.bar(x + width / 2, case["seidel_aligned"], width, label="aligned recovered", color="#ef7d55")
    ax_bar.scatter(x, case["seidel_raw"], marker="x", color="black", s=23, label="raw recovered", zorder=4)
    ax_bar.axhline(0.0, color="0.50", linewidth=0.8)
    ax_bar.set_ylim(*ylimits)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(COEFF_LABELS, fontsize=7)
    ax_bar.set_ylabel("Coeff", fontsize=7)
    ax_bar.grid(axis="y", alpha=0.22)
    ax_bar.legend(loc="upper right", ncol=3, fontsize=6.5, frameon=False)
    ax_bar.tick_params(axis="y", labelsize=7)

    ax_table = fig.add_subplot(sub[2, 0])
    ax_table.axis("off")
    values = [
        [short_float(v, 3) for v in case["seidel_gt"]],
        [short_float(v, 3) for v in case["seidel_raw"]],
        [short_float(v, 3) for v in case["seidel_aligned"]],
    ]
    table = ax_table.table(
        cellText=values,
        rowLabels=["GT", "raw", "aligned"],
        colLabels=COEFF_LABELS,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    table.scale(1.0, 1.03)
    for cell in table.get_celld().values():
        cell.set_linewidth(0.25)
        cell.set_edgecolor("0.75")


def make_pair_panel(
    *,
    top: dict[str, Any],
    bottom: dict[str, Any],
    out_path: Path,
    top_label: str,
    bottom_label: str,
) -> dict[str, Any]:
    cases = [top, bottom]
    ranges = collect_display_ranges(cases)
    ylimits = coeff_ylim(cases)
    row = top["row"]
    target = parse_float(row, "target_wavefront_rms")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20.0, 9.2), dpi=150)
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.16, 1.0],
        height_ratios=[1, 1],
        left=0.025,
        right=0.985,
        top=0.92,
        bottom=0.055,
        wspace=0.045,
        hspace=0.22,
    )
    title = (
        f"Controlled RCP comparison: lambda=0 vs lambda=10000 | "
        f"{row['image']} | {row['direction']} | 6D | GT RMS {target:.2f}"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.978)

    draw_image_panel(fig, outer[0, 0], top, ranges)
    draw_coeff_card(fig, outer[0, 1], top, ylimits, lambda_label=top_label)
    draw_image_panel(fig, outer[1, 0], bottom, ranges)
    draw_coeff_card(fig, outer[1, 1], bottom, ylimits, lambda_label=bottom_label)

    fig.text(0.012, 0.705, top_label, rotation=90, ha="center", va="center", fontsize=12, fontweight="bold")
    fig.text(0.012, 0.285, bottom_label, rotation=90, ha="center", va="center", fontsize=12, fontweight="bold")

    fig.savefig(out_path)
    plt.close(fig)

    return {
        "image": row["image"],
        "direction": row["direction"],
        "target_wavefront_rms": target,
        "lambda_top": parse_float(top["row"], "lambda"),
        "lambda_bottom": parse_float(bottom["row"], "lambda"),
        "top_operator_error_calibrated": parse_float(top["row"], "operator_error_calibrated"),
        "bottom_operator_error_calibrated": parse_float(bottom["row"], "operator_error_calibrated"),
        "top_recovered_over_gt_rms": parse_float(top["row"], "wavefront_recovered_over_gt_rms"),
        "bottom_recovered_over_gt_rms": parse_float(bottom["row"], "wavefront_recovered_over_gt_rms"),
        "top_ssim": parse_float(top["row"], "ssim_recon_gain_vs_gt"),
        "bottom_ssim": parse_float(bottom["row"], "ssim_recon_gain_vs_gt"),
        "path": display_path(out_path),
    }


def load_eval_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def sorted_keys(rows: list[dict[str, Any]]) -> list[tuple[str, str, float]]:
    def image_rank(image: str) -> tuple[int, str]:
        return (IMAGE_ORDER.index(image), image) if image in IMAGE_ORDER else (len(IMAGE_ORDER), image)

    def direction_rank(direction: str) -> tuple[int, str]:
        return (
            DIRECTION_ORDER.index(direction),
            direction,
        ) if direction in DIRECTION_ORDER else (len(DIRECTION_ORDER), direction)

    keys = {
        (str(row["image"]), str(row["direction"]), float(row["target_wavefront_rms"]))
        for row in rows
    }
    return sorted(keys, key=lambda item: (item[2], image_rank(item[0]), direction_rank(item[1])))


def make_contact_sheets(manifest_rows: list[dict[str, Any]], out_dir: Path) -> None:
    by_rms: dict[str, list[dict[str, Any]]] = {}
    for row in manifest_rows:
        by_rms.setdefault(rms_label(float(row["target_wavefront_rms"])), []).append(row)

    overview_dir = out_dir / "00_contact_sheets_by_rms"
    overview_dir.mkdir(parents=True, exist_ok=True)
    resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    for label, rows in sorted(by_rms.items()):
        images = []
        for row in rows:
            path = PROJECT_ROOT / str(row["path"])
            im = Image.open(path).convert("RGB")
            target_width = 900
            target_height = int(round(im.height * target_width / im.width))
            images.append(im.resize((target_width, target_height), resample_filter))
        if not images:
            continue
        cols = 2
        gap = 24
        rows_n = int(math.ceil(len(images) / cols))
        cell_w = max(im.width for im in images)
        cell_h = max(im.height for im in images)
        canvas = Image.new("RGB", (cols * cell_w + (cols - 1) * gap, rows_n * cell_h + (rows_n - 1) * gap), "white")
        for idx, im in enumerate(images):
            x = (idx % cols) * (cell_w + gap)
            y = (idx // cols) * (cell_h + gap)
            canvas.paste(im, (x, y))
        canvas.save(overview_dir / f"{label}_contact_sheet.png")


def write_manifest(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(rows: list[dict[str, Any]], out_dir: Path, csv_path: Path) -> None:
    lines = [
        "# Lambda 0 vs 10000 controlled RCP pairs",
        "",
        f"Input evaluator CSV: `{csv_path}`",
        "",
        "Folder layout:",
        "- `rms*/<image>/lambda0_vs_lambda10000__<image>__<direction>__<rms>__RCP_vertical.png`",
        "- top row: lambda=0",
        "- bottom row: lambda=10000",
        "",
        "The paired rows use shared image display ranges and shared Seidel coefficient y-limits.",
        "The left image panels are GT, measurement, joint recon, predicted measurement, and gain abs error.",
        "",
        f"Generated paired RCP files: {len(rows)}",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['path']}`")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/cocoa_like_2d_mechanism"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-lambda", type=float, default=0.0)
    parser.add_argument("--bottom-lambda", type=float, default=10000.0)
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()

    eval_rows = load_eval_rows(args.csv)
    row_by_key: dict[tuple[str, str, float, float], dict[str, Any]] = {}
    for row in eval_rows:
        key = (
            str(row["image"]),
            str(row["direction"]),
            float(row["target_wavefront_rms"]),
            float(row["lambda"]),
        )
        row_by_key[key] = row

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for image, direction, target in sorted_keys(eval_rows):
        top_row = row_by_key.get((image, direction, target, args.top_lambda))
        bottom_row = row_by_key.get((image, direction, target, args.bottom_lambda))
        if top_row is None or bottom_row is None:
            missing.append(f"{image} {direction} {target:.2f}")
            continue

        top_case = build_case(top_row, args.output_root)
        bottom_case = build_case(bottom_row, args.output_root)
        label = rms_label(target)
        out_path = (
            args.out_dir
            / label
            / safe_name(image)
            / f"lambda0_vs_lambda10000__{safe_name(image)}__{safe_name(direction)}__{label}__RCP_vertical.png"
        )
        manifest_rows.append(
            make_pair_panel(
                top=top_case,
                bottom=bottom_case,
                out_path=out_path,
                top_label="lambda=0",
                bottom_label="lambda=10000",
            )
        )

    write_manifest(manifest_rows, args.out_dir)
    write_readme(manifest_rows, args.out_dir, args.csv)
    if not args.no_contact_sheets:
        make_contact_sheets(manifest_rows, args.out_dir)
    if missing:
        (args.out_dir / "missing_cases.txt").write_text("\n".join(missing) + "\n")
    print(f"[done] wrote {len(manifest_rows)} paired RCP files to {args.out_dir}")
    if missing:
        print(f"[warn] missing {len(missing)} cases; see {args.out_dir / 'missing_cases.txt'}")


if __name__ == "__main__":
    main()
