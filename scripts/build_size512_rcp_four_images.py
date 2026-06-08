"""Build size512 Ranked CoeffSim Panels for the four standard images.

RCP means the combined reconstruction panel plus Seidel coefficient similarity
card. This script is intentionally post-hoc: it reads completed run tensors and
physical-operator evaluator CSVs, then writes PNG panels and a compact manifest.
"""

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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import (  # noqa: E402
    field_weighted_wavefront_rms,
)


DEFAULT_CASES = [
    {
        "rank": 1,
        "image": "Test_figure_1",
        "family": "Fourier",
        "candidate": "oct8_ang30",
        "run_dir": "size512_best_fourier_tunedprior_noskip6x128_pre400scalar5_Test_figure_1_20260604__oct8_ang30",
        "eval_dir": "size512_best_fourier_tunedprior_noskip6x128_pre400scalar5_Test_figure_1_20260604_operator_eval_dim512",
        "file_label": "oct8_ang30",
    },
    {
        "rank": 2,
        "image": "Iksung_beads",
        "family": "Fourier",
        "candidate": "oct8_ang30",
        "run_dir": "size512_best_fourier_tunedprior_noskip6x128_pre400scalar5_Iksung_beads_20260604__oct8_ang30",
        "eval_dir": "size512_best_fourier_tunedprior_noskip6x128_pre400scalar5_Iksung_beads_20260604_operator_eval_dim512",
        "file_label": "oct8_ang30",
    },
    {
        "rank": 3,
        "image": "dendrites",
        "family": "Pretrain",
        "candidate": "pre400__scalar7p5",
        "run_dir": "size512_best_pretrain_tunedprior_noskip6x128_dendrites_20260604__pre400__scalar7p5",
        "eval_dir": "size512_best_pretrain_tunedprior_noskip6x128_dendrites_20260604_operator_eval_dim512",
        "file_label": "pre400__scalar7p5",
    },
    {
        "rank": 4,
        "image": "dendrites_dense",
        "family": "Fourier",
        "candidate": "oct8_ang30",
        "run_dir": "size512_best_fourier_tunedprior_noskip6x128_pre400scalar5_dendrites_dense_20260604__oct8_ang30",
        "eval_dir": "size512_best_fourier_tunedprior_noskip6x128_pre400scalar5_dendrites_dense_20260604_operator_eval_dim512",
        "file_label": "oct8_ang30",
    },
]

COEFF_LABELS = ["W040", "W131", "W222", "W220", "W311", "Wd"]


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple, int, float, bool)):
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
        raise ValueError("Missing vector value")
    arr = np.asarray([float(item) for item in parsed], dtype=np.float64)
    if arr.shape != (6,):
        raise ValueError(f"Expected a 6D Seidel vector, got shape {arr.shape}")
    return arr


def parse_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value in (None, ""):
        return float(default)
    return float(value)


def load_eval_row(eval_csv: Path, candidate: str) -> dict[str, Any]:
    rows = list(csv.DictReader(eval_csv.open()))
    if not rows:
        raise ValueError(f"No rows in {eval_csv}")
    matches = [
        row
        for row in rows
        if row.get("candidate_id") == candidate or row.get("profile") == candidate
    ]
    if len(matches) == 1:
        return matches[0]
    if len(rows) == 1:
        return rows[0]
    raise ValueError(f"Could not find exactly one candidate={candidate!r} in {eval_csv}")


def load_tensors(tensors_path: Path) -> dict[str, Any]:
    try:
        return torch.load(tensors_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(tensors_path, map_location="cpu")


def as_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def percentile01(arr: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if float(hi) <= float(lo):
        return normalize01(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def short_float(value: float, digits: int = 4) -> str:
    if math.isnan(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def wrapped(text: str, width: int = 98) -> str:
    return "\n".join(wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def make_panel(
    *,
    case: dict[str, Any],
    output_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    run_dir = output_root / case["run_dir"]
    metrics_path = run_dir / "joint" / "metrics.json"
    tensors_path = run_dir / "joint" / "tensors.pt"
    eval_csv = output_root / case["eval_dir"] / "seidel_physical_operator_metrics.csv"

    metrics = json.loads(metrics_path.read_text())
    eval_row = load_eval_row(eval_csv, str(case["candidate"]))
    tensors = load_tensors(tensors_path)

    gt = as_array(tensors["sharp_gt"])
    meas = as_array(tensors["measurement_gt"])
    recon = as_array(tensors["sharp_recon"])
    pred = as_array(tensors["measurement_pred"])
    gain = float(metrics.get("best_gain_recon_to_gt", 1.0))
    recon_gain = recon * gain
    err = np.abs(recon_gain - gt)

    seidel_gt = parse_vector(eval_row["seidel_gt"])
    seidel_raw = parse_vector(eval_row["seidel_final"])
    seidel_aligned = parse_vector(eval_row["aligned_seidel_physical"])
    gt_rms = field_weighted_wavefront_rms(seidel_gt)
    raw_rms = field_weighted_wavefront_rms(seidel_raw)
    aligned_rms = field_weighted_wavefront_rms(seidel_aligned)

    op_cal = parse_float(eval_row, "operator_error_calibrated")
    phys = parse_float(eval_row, "operator_error_phys_equiv")
    coord = parse_float(eval_row, "operator_error_coord_diagnostic")
    aligned_wf = parse_float(eval_row, "aligned_wavefront_error_physical")
    ssim = parse_float(eval_row, "ssim_recon_gain_vs_gt")
    nrmse = parse_float(eval_row, "nrmse_recon_gain_vs_gt")
    raw_sign = parse_float(eval_row, "canonical_sign_match_rate_raw")
    phys_sign = parse_float(eval_row, "canonical_sign_match_rate_physical")
    gauge_sign = parse_float(eval_row, "canonical_sign_match_rate_gauge")
    gauge_transform = eval_row.get("canonical_transform_gauge", "?")

    rank = int(case["rank"])
    file_name = (
        f"rank{rank:02d}__{case['image']}__{case['file_label']}__"
        "size512_RCP.png"
    )
    out_path = output_dir / file_name

    fig = plt.figure(figsize=(20.0, 6.0), dpi=150)
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 1.0],
        left=0.025,
        right=0.985,
        top=0.92,
        bottom=0.075,
        wspace=0.06,
    )
    left = outer[0, 0].subgridspec(2, 3, wspace=0.08, hspace=0.16)
    image_items = [
        ("Sharp GT", gt, "gray", normalize01),
        ("Measurement", meas, "gray", normalize01),
        ("Recon raw clipped", recon, "gray", normalize01),
        ("Recon percentile", recon, "gray", percentile01),
        ("Predicted measurement", pred, "gray", normalize01),
        ("Gain-aligned abs error", err, "magma", percentile01),
    ]
    for idx, (title, arr, cmap, norm_fn) in enumerate(image_items):
        ax = fig.add_subplot(left[idx // 3, idx % 3])
        ax.imshow(norm_fn(arr), cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.text(
        0.27,
        0.972,
        (
            f"SSIM_gain={short_float(ssim)}  NRMSE_gain={short_float(nrmse)}  "
            f"HF raw={short_float(float(metrics.get('recon_raw_hf_ratio', math.nan)))}  "
            f"HF meas={short_float(float(metrics.get('measurement_hf_ratio', math.nan)))}"
        ),
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    right = outer[0, 1].subgridspec(3, 1, height_ratios=[0.68, 2.1, 0.68], hspace=0.22)
    ax_text = fig.add_subplot(right[0, 0])
    ax_text.axis("off")
    title = (
        f"RCP size512 rank {rank:02d} | {case['image']} | "
        f"{case['family']} {case['candidate']}"
    )
    lines = [
        title,
        f"op_cal={short_float(op_cal)} | phys={short_float(phys)} | coord={short_float(coord)}",
        (
            f"RMS waves: GT={short_float(gt_rms)} | "
            f"rec_aligned={short_float(aligned_rms)} | rec_raw={short_float(raw_rms)}"
        ),
        (
            f"best_phys={eval_row.get('best_physical_transform', '?')} | "
            f"aligned_WFrel={short_float(aligned_wf)} | "
            f"SSIM={short_float(ssim)} | NRMSE={short_float(nrmse)}"
        ),
        (
            f"sign raw={short_float(raw_sign, 2)} | phys={short_float(phys_sign, 2)} | "
            f"gauge={short_float(gauge_sign, 2)} | gauge_g={gauge_transform}"
        ),
        wrapped(f"run={case['run_dir']}", width=92),
    ]
    ax_text.text(0.0, 0.98, title, ha="left", va="top", fontsize=11, fontweight="bold")
    ax_text.text(0.0, 0.74, "\n".join(lines[1:]), ha="left", va="top", fontsize=8.8)

    ax_bar = fig.add_subplot(right[1, 0])
    x = np.arange(len(COEFF_LABELS), dtype=np.float64)
    width = 0.34
    ax_bar.bar(x - width / 2, seidel_gt, width, label="GT", color="#55b8b0")
    ax_bar.bar(x + width / 2, seidel_aligned, width, label="aligned recovered", color="#ef7d55")
    ax_bar.scatter(x, seidel_raw, marker="x", color="black", s=28, label="raw recovered", zorder=4)
    ax_bar.axhline(0.0, color="0.55", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(COEFF_LABELS, fontsize=8)
    ax_bar.set_ylabel("coefficient", fontsize=8)
    ax_bar.set_title("Seidel coefficients: GT vs raw/aligned recovered", fontsize=9)
    ax_bar.grid(axis="y", alpha=0.22)
    ax_bar.legend(loc="upper right", ncol=3, fontsize=7, frameon=False)
    ax_bar.tick_params(axis="y", labelsize=8)

    ax_table = fig.add_subplot(right[2, 0])
    ax_table.axis("off")
    table_values = [
        [short_float(v, 3) for v in seidel_gt],
        [short_float(v, 3) for v in seidel_raw],
        [short_float(v, 3) for v in seidel_aligned],
    ]
    table = ax_table.table(
        cellText=table_values,
        rowLabels=["GT", "raw", "aligned"],
        colLabels=COEFF_LABELS,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.15)
    for cell in table.get_celld().values():
        cell.set_linewidth(0.25)
        cell.set_edgecolor("0.75")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    rel_path = display_path(out_path)
    return {
        "rank": rank,
        "image": case["image"],
        "candidate": case["candidate"],
        "family": case["family"],
        "op_cal": op_cal,
        "phys_equiv": phys,
        "coord_diag": coord,
        "ssim": ssim,
        "nrmse": nrmse,
        "gt_rms": gt_rms,
        "recovered_rms_aligned": aligned_rms,
        "recovered_rms_raw": raw_rms,
        "aligned_wavefront_error_relative": aligned_wf,
        "best_physical_transform": eval_row.get("best_physical_transform", ""),
        "path": rel_path,
    }


def write_manifest(rows: list[dict[str, Any]], output_dir: Path) -> None:
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(rows: list[dict[str, Any]], output_dir: Path, overview_name: str) -> None:
    lines = [
        "# Size512 RCP for four images",
        "",
        "RCP includes field-weighted Seidel RMS in waves:",
        "- `gt_rms`: RMS of `seidel_gt`",
        "- `recovered_rms_aligned`: RMS of `aligned_seidel_physical` and the main `rec_aligned` value shown on the card",
        "- `recovered_rms_raw`: RMS of raw `seidel_final` before physical-equivalence alignment",
        "",
    ]
    lines.extend(str(row["path"]) for row in rows)
    lines.extend(["", f"Overview: {display_path(output_dir / overview_name)}", ""])
    (output_dir / "README.md").write_text("\n".join(lines))


def make_overview(rows: list[dict[str, Any]], output_dir: Path, overview_name: str) -> Path:
    images = [Image.open(PROJECT_ROOT / row["path"]).convert("RGB") for row in rows]
    target_width = 1600
    resized = []
    resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    for image in images:
        height = int(round(image.height * target_width / image.width))
        resized.append(image.resize((target_width, height), resample_filter))
    gap = 28
    total_height = sum(image.height for image in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (target_width, total_height), "white")
    y = 0
    for image in resized:
        canvas.paste(image, (0, y))
        y += image.height + gap
    out_path = output_dir / overview_name
    canvas.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/cocoa_like_2d_mechanism"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cocoa_like_2d_mechanism/size512_rcp_four_images_20260604"),
    )
    parser.add_argument("--overview-name", default="four_images_size512_RCP_overview.png")
    args = parser.parse_args()

    output_root = args.output_root
    output_dir = args.output_dir
    rows = [
        make_panel(case=case, output_root=output_root, output_dir=output_dir)
        for case in DEFAULT_CASES
    ]
    write_manifest(rows, output_dir)
    write_readme(rows, output_dir, args.overview_name)
    overview = make_overview(rows, output_dir, args.overview_name)
    print(f"[done] wrote {overview}")


if __name__ == "__main__":
    main()
