#!/usr/bin/env python3
"""Build controlled RCP comparisons for floor, fixed-ratio, and ratio-schedule runs."""

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
IMAGE_ORDER = ["Iksung_beads", "dendrites", "dendrites_dense", "Test_figure_1"]
METHOD_ORDER = ["floor_alpha0p8_lambda10000", "fixed_alpha1_lambda1000", "alpha_schedule_lambda1000"]


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
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def rms_label(value: float) -> str:
    return f"rms{value:.2f}".replace(".", "p")


def safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def wrap_line(text: str, width: int = 100) -> str:
    return "\n".join(wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def short_run_name(value: Any) -> str:
    text = str(value)
    for marker in ["20260606_", "20260605_"]:
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


def build_case(row: dict[str, Any], output_root: Path, method: str, label: str) -> dict[str, Any]:
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
    gt_rms = field_weighted_wavefront_rms(seidel_gt)
    raw_rms = field_weighted_wavefront_rms(seidel_raw)
    aligned_rms = field_weighted_wavefront_rms(seidel_aligned)

    return {
        "method": method,
        "label": label,
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
        "gt_rms": gt_rms,
        "raw_rms": raw_rms,
        "aligned_rms": aligned_rms,
        "raw_ratio": raw_rms / gt_rms if gt_rms > 0 else math.nan,
        "aligned_ratio": aligned_rms / gt_rms if gt_rms > 0 else math.nan,
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


def target_ratio_for_case(case: dict[str, Any]) -> float:
    row = case["row"]
    method = case["method"]
    target = parse_float(row, "target_wavefront_rms")
    alpha = parse_float(row, "alpha")
    if method == "floor_alpha0p8_lambda10000":
        return alpha
    return alpha if math.isfinite(alpha) else math.nan


def draw_coeff_card(
    fig: plt.Figure,
    spec: Any,
    case: dict[str, Any],
    ylimits: tuple[float, float],
) -> None:
    row = case["row"]
    metrics = case["metrics"]
    sub = spec.subgridspec(3, 1, height_ratios=[0.66, 1.7, 0.60], hspace=0.18)

    op_cal = parse_float(row, "operator_error_calibrated")
    phys = parse_float(row, "operator_error_phys_equiv")
    coord = parse_float(row, "operator_error_coord_diagnostic")
    ssim = parse_float(row, "ssim_recon_gain_vs_gt")
    nrmse = parse_float(row, "nrmse_recon_gain_vs_gt")
    target = parse_float(row, "target_wavefront_rms")
    prior_loss = parse_float(row, "final_seidel_rms_floor_loss")
    target_ratio = target_ratio_for_case(case)
    target_abs = target * target_ratio if math.isfinite(target_ratio) else math.nan
    raw_sign = parse_float(row, "canonical_sign_match_rate_raw")
    phys_sign = parse_float(row, "canonical_sign_match_rate_physical")
    gauge_sign = parse_float(row, "canonical_sign_match_rate_gauge")
    gauge_transform = row.get("canonical_transform_gauge", "?")

    ax_text = fig.add_subplot(sub[0, 0])
    ax_text.axis("off")
    title = f"{case['label']} | {row['image']} | {row['direction']} | GT RMS {target:.2f}"
    lines = [
        f"op_cal={short_float(op_cal)} | phys={short_float(phys)} | coord={short_float(coord)}",
        (
            f"RMS waves: GT={short_float(case['gt_rms'])} | aligned={short_float(case['aligned_rms'])} "
            f"({short_float(case['aligned_ratio'], 3)}x GT) | raw={short_float(case['raw_rms'])} "
            f"({short_float(case['raw_ratio'], 3)}x GT)"
        ),
        (
            f"target={short_float(target_ratio, 3)}x GT ({short_float(target_abs)}) | "
            f"prior_loss={short_float(prior_loss, 3)} | SSIM={short_float(ssim)} | "
            f"NRMSE={short_float(nrmse)} | gain={short_float(float(metrics.get('best_gain_recon_to_gt', math.nan)), 3)}"
        ),
        (
            f"sign raw={short_float(raw_sign, 2)} | phys={short_float(phys_sign, 2)} | "
            f"gauge={short_float(gauge_sign, 2)} | gauge_g={gauge_transform}"
        ),
        wrap_line(
            f"id={row.get('candidate_id', '')} | run={short_run_name(row.get('run_root', ''))}",
            width=104,
        ),
    ]
    ax_text.text(0.0, 0.98, title, ha="left", va="top", fontsize=9.6, fontweight="bold")
    ax_text.text(0.0, 0.70, "\n".join(lines), ha="left", va="top", fontsize=7.0)

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
    table.set_fontsize(6.1)
    table.scale(1.0, 0.95)
    for cell in table.get_celld().values():
        cell.set_linewidth(0.25)
        cell.set_edgecolor("0.75")


def make_comparison_panel(
    *,
    cases: list[dict[str, Any]],
    out_path: Path,
) -> dict[str, Any]:
    ranges = collect_display_ranges(cases)
    ylimits = coeff_ylim(cases)
    row = cases[0]["row"]
    target = parse_float(row, "target_wavefront_rms")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20.0, 13.2), dpi=150)
    outer = fig.add_gridspec(
        len(cases),
        2,
        width_ratios=[1.16, 1.0],
        height_ratios=[1] * len(cases),
        left=0.027,
        right=0.985,
        top=0.925,
        bottom=0.045,
        wspace=0.045,
        hspace=0.21,
    )
    title = (
        f"Controlled RCP comparison: floor vs fixed ratio vs alpha schedule | "
        f"{row['image']} | {row['direction']} | 6D | GT RMS {target:.2f}"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.982)

    for idx, case in enumerate(cases):
        draw_image_panel(fig, outer[idx, 0], case, ranges)
        draw_coeff_card(fig, outer[idx, 1], case, ylimits)
        fig.text(
            0.013,
            0.80 - idx * 0.295,
            case["label"],
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )

    fig.savefig(out_path)
    plt.close(fig)

    output = {
        "image": row["image"],
        "direction": row["direction"],
        "target_wavefront_rms": target,
        "path": display_path(out_path),
    }
    for case in cases:
        prefix = case["method"]
        output[f"{prefix}_target_ratio"] = target_ratio_for_case(case)
        output[f"{prefix}_aligned_over_gt_rms"] = case["aligned_ratio"]
        output[f"{prefix}_operator_error_calibrated"] = parse_float(case["row"], "operator_error_calibrated")
        output[f"{prefix}_ssim"] = parse_float(case["row"], "ssim_recon_gain_vs_gt")
        output[f"{prefix}_prior_loss"] = parse_float(case["row"], "final_seidel_rms_floor_loss")
    return output


def load_eval_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def index_rows(
    rows: list[dict[str, Any]],
    *,
    direction: str,
    lambda_value: float,
    images: set[str],
    rms_values: set[float],
) -> dict[tuple[str, str, float], dict[str, Any]]:
    indexed: dict[tuple[str, str, float], dict[str, Any]] = {}
    for row in rows:
        image = str(row["image"])
        row_direction = str(row["direction"])
        rms = round(float(row["target_wavefront_rms"]), 6)
        lam = float(row["lambda"])
        if image not in images or row_direction != direction or rms not in rms_values:
            continue
        if not math.isclose(lam, lambda_value, rel_tol=0.0, abs_tol=1e-6):
            continue
        indexed[(image, row_direction, rms)] = row
    return indexed


def sorted_keys(images: list[str], direction: str, rms_values: list[float]) -> list[tuple[str, str, float]]:
    def image_rank(image: str) -> tuple[int, str]:
        return (IMAGE_ORDER.index(image), image) if image in IMAGE_ORDER else (len(IMAGE_ORDER), image)

    return sorted(
        [(image, direction, round(float(rms), 6)) for rms in rms_values for image in images],
        key=lambda item: (item[2], image_rank(item[0])),
    )


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
            target_width = 880
            target_height = int(round(im.height * target_width / im.width))
            images.append(im.resize((target_width, target_height), resample_filter))
        if not images:
            continue
        cols = 1
        gap = 22
        cell_w = max(im.width for im in images)
        cell_h = max(im.height for im in images)
        canvas = Image.new("RGB", (cols * cell_w, len(images) * cell_h + (len(images) - 1) * gap), "white")
        for idx, im in enumerate(images):
            canvas.paste(im, (0, idx * (cell_h + gap)))
        canvas.save(overview_dir / f"{label}_contact_sheet.png")


def write_manifest(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Ratio target alpha-schedule RCP comparisons",
        "",
        "Each PNG uses shared display ranges and shared Seidel coefficient y-limits across the three rows.",
        "",
        "Rows:",
        "- top: floor alpha=0.8, lambda=10000",
        "- middle: ratio target alpha=1, lambda=1000",
        "- bottom: ratio target alpha schedule, lambda=1000",
        "",
        f"Floor CSV: `{args.floor_csv}`",
        f"Fixed CSV: `{args.fixed_csv}`",
        f"Schedule CSV: `{args.schedule_csv}`",
        "",
        "Folder layout:",
        "- `rms*/<image>/floor_vs_fixed_vs_schedule__<image>__signed_balanced__<rms>__RCP_vertical.png`",
        "- `00_contact_sheets_by_rms/<rms>_contact_sheet.png`",
        "",
        f"Generated comparison RCP files: {len(rows)}",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['path']}`")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor-csv", type=Path, required=True)
    parser.add_argument("--fixed-csv", type=Path, required=True)
    parser.add_argument("--schedule-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cocoa_like_2d_mechanism"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--direction", default="signed_balanced")
    parser.add_argument("--images", nargs="+", default=["Iksung_beads", "dendrites", "dendrites_dense"])
    parser.add_argument("--rms-values", nargs="+", type=float, default=[0.06, 0.20, 0.40])
    parser.add_argument("--floor-lambda", type=float, default=10000.0)
    parser.add_argument("--fixed-lambda", type=float, default=1000.0)
    parser.add_argument("--schedule-lambda", type=float, default=1000.0)
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()

    images = set(args.images)
    rms_values = {round(float(v), 6) for v in args.rms_values}
    floor = index_rows(
        load_eval_rows(args.floor_csv),
        direction=args.direction,
        lambda_value=args.floor_lambda,
        images=images,
        rms_values=rms_values,
    )
    fixed = index_rows(
        load_eval_rows(args.fixed_csv),
        direction=args.direction,
        lambda_value=args.fixed_lambda,
        images=images,
        rms_values=rms_values,
    )
    schedule = index_rows(
        load_eval_rows(args.schedule_csv),
        direction=args.direction,
        lambda_value=args.schedule_lambda,
        images=images,
        rms_values=rms_values,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for image, direction, rms in sorted_keys(args.images, args.direction, args.rms_values):
        key = (image, direction, rms)
        if key not in floor or key not in fixed or key not in schedule:
            missing.append(f"{image} {direction} {rms:.2f}")
            continue
        cases = [
            build_case(floor[key], args.output_root, "floor_alpha0p8_lambda10000", "floor alpha=0.8 | lambda=10000"),
            build_case(fixed[key], args.output_root, "fixed_alpha1_lambda1000", "ratio target=1x | lambda=1000"),
            build_case(schedule[key], args.output_root, "alpha_schedule_lambda1000", "schedule target | lambda=1000"),
        ]
        label = rms_label(rms)
        out_path = (
            args.out_dir
            / label
            / safe_name(image)
            / f"floor_vs_fixed_vs_schedule__{safe_name(image)}__{safe_name(direction)}__{label}__RCP_vertical.png"
        )
        manifest_rows.append(make_comparison_panel(cases=cases, out_path=out_path))

    write_manifest(manifest_rows, args.out_dir)
    write_readme(manifest_rows, args.out_dir, args)
    if not args.no_contact_sheets:
        make_contact_sheets(manifest_rows, args.out_dir)
    if missing:
        (args.out_dir / "missing_cases.txt").write_text("\n".join(missing) + "\n")
    print(f"[done] wrote {len(manifest_rows)} comparison RCP files to {args.out_dir}")
    if missing:
        print(f"[warn] missing {len(missing)} cases; see {args.out_dir / 'missing_cases.txt'}")


if __name__ == "__main__":
    main()
