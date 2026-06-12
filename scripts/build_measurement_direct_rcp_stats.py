"""Build RCP-style panels for measurement-direct pretrain contrast runs.

These runs use the input image itself as the measurement.  There is no
synthetic Seidel ground truth, so this script deliberately skips physical
operator/alignment metrics and focuses on measurement fit plus the raw
equivalent Seidel coefficients recovered by the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
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


DEFAULT_PREFIX = "desktop3_measurement_direct_scalar5_4d_size256_pre400_joint1000_20260612"
DEFAULT_OUTPUT_DIR = (
    "outputs/cocoa_like_2d_mechanism/"
    "desktop3_measurement_direct_scalar5_4d_size256_pre400_joint1000_20260612_rcp_stats"
)
COEFF_LABELS = ["W040", "W131", "W222", "W220", "W311", "Wd"]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
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


def load_settings_manifest(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Settings manifest must be a JSON list: {path}")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Settings manifest entry {idx} is not an object")
        row = dict(item)
        if "method" not in row:
            raise ValueError(f"Settings manifest entry {idx} is missing method")
        row.setdefault("label", row["method"])
        row.setdefault("family", "measurement_direct")
        out.append(row)
    return out


def load_tensors(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def parse_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_vector(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    else:
        text = str(value).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = []
        arr = np.asarray(parsed, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size < 6:
        arr = np.pad(arr, (0, 6 - arr.size), constant_values=0.0)
    return arr[:6]


def tag_text(text: str) -> str:
    safe = []
    for char in str(text):
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "case"


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


def clipped01(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)


def optimal_gain(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).ravel()
    est = np.asarray(estimate, dtype=np.float64).ravel()
    denom = float(np.dot(est, est))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(ref, est) / denom)


def short_float(value: float, digits: int = 4) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def run_dir_for(output_root: Path, prefix: str, method: str) -> Path:
    return output_root / f"{prefix}__{method}"


def case_dir_for(output_root: Path, prefix: str, row: dict[str, Any]) -> Path:
    return (
        run_dir_for(output_root, prefix, str(row["pretrain_method"]))
        / "stage1"
        / f"{row['image']}__{row['candidate_id']}"
        / "joint"
    )


def load_rows(output_root: Path, prefix: str, settings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for setting in settings:
        method = str(setting["method"])
        run_dir = run_dir_for(output_root, prefix, method)
        for row_idx, row in enumerate(read_csv(run_dir / "stage1_metrics.csv")):
            merged = dict(row)
            merged["pretrain_method"] = method
            merged["method_label"] = str(setting.get("label", method))
            merged["family"] = str(setting.get("family", "measurement_direct"))
            merged["_source_row_index"] = row_idx
            for key, value in setting.items():
                merged.setdefault(key, value)
            rows.append(merged)
    rows.sort(key=lambda row: (str(row.get("image", "")), str(row.get("pretrain_method", ""))))
    return rows


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        seidel = parse_vector(row.get("seidel_final", "[]"))
        row["seidel_wavefront_rms_f"] = field_weighted_wavefront_rms(seidel)
        row["ssim_meas_pred_vs_meas_f"] = parse_float(row, "ssim_meas_pred_vs_meas")
        row["nrmse_meas_pred_vs_meas_f"] = parse_float(row, "nrmse_meas_pred_vs_meas")
        row["ssim_object_vs_input_f"] = parse_float(row, "ssim_recon_gain_vs_gt")
        row["nrmse_object_vs_input_f"] = parse_float(row, "nrmse_recon_gain_vs_gt")
        row["final_loss_f"] = parse_float(row, "final_loss")
        row["final_ssim_loss_f"] = parse_float(row, "final_ssim_loss")
        row["final_rsd_loss_f"] = parse_float(row, "final_rsd_loss")
        row["pretrain_final_loss_f"] = parse_float(row, "pretrain_final_loss")
        row["pretrain_render_ssim_vs_target_f"] = parse_float(row, "pretrain_render_ssim_vs_target")
        row["pretrain_render_nrmse_vs_target_f"] = parse_float(row, "pretrain_render_nrmse_vs_target")
        out.append(row)
    return out


def grouped_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    metrics = [
        "ssim_meas_pred_vs_meas_f",
        "nrmse_meas_pred_vs_meas_f",
        "ssim_object_vs_input_f",
        "nrmse_object_vs_input_f",
        "seidel_wavefront_rms_f",
        "final_loss_f",
        "pretrain_final_loss_f",
    ]
    out: list[dict[str, Any]] = []
    for value, group in sorted(groups.items()):
        record: dict[str, Any] = {key: value, "count": len(group)}
        for metric in metrics:
            vals = [float(row[metric]) for row in group if math.isfinite(float(row[metric]))]
            record[f"{metric}_mean"] = float(np.mean(vals)) if vals else math.nan
            record[f"{metric}_median"] = float(np.median(vals)) if vals else math.nan
        out.append(record)
    return out


def comparison_by_case(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "image": row.get("image"),
                "candidate_id": row.get("candidate_id"),
                "pretrain_method": row.get("pretrain_method"),
                "method_label": row.get("method_label"),
                "ssim_meas_pred_vs_meas": row["ssim_meas_pred_vs_meas_f"],
                "nrmse_meas_pred_vs_meas": row["nrmse_meas_pred_vs_meas_f"],
                "ssim_object_vs_input": row["ssim_object_vs_input_f"],
                "nrmse_object_vs_input": row["nrmse_object_vs_input_f"],
                "seidel_wavefront_rms": row["seidel_wavefront_rms_f"],
                "final_loss": row["final_loss_f"],
                "pretrain_final_loss": row["pretrain_final_loss_f"],
            }
        )
    return out


def wrapped(text: str, width: int = 52) -> str:
    return "\n".join(wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False))


def render_case_rcp(
    *,
    output_root: Path,
    prefix: str,
    row: dict[str, Any],
    output_dir: Path,
    rank: int,
) -> Path:
    case_dir = case_dir_for(output_root, prefix, row)
    tensors = load_tensors(case_dir / "tensors.pt")
    measurement = as_array(tensors["measurement_gt"])
    pretrain_target = as_array(tensors.get("pretrain_target", measurement))
    pretrain_render = as_array(tensors.get("pretrain_render", measurement))
    pretrain_abs_error = as_array(tensors.get("pretrain_abs_error", np.abs(pretrain_render - pretrain_target)))
    recon = as_array(tensors["sharp_recon"])
    pred = as_array(tensors["measurement_pred"])
    pred_error = np.abs(pred - measurement)
    gain = optimal_gain(measurement, recon)
    object_error = np.abs(measurement - gain * recon)
    seidel = parse_vector(tensors.get("seidel_final", row.get("seidel_final", "[]")))

    fig = plt.figure(figsize=(18, 10.5))
    gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 1.18])
    panels = [
        (measurement, "Input measurement", "gray", 0.0, 1.0),
        (clipped01(pretrain_target), "Pretrain target clipped", "gray", 0.0, 1.0),
        (clipped01(pretrain_render), "Pretrain render clipped", "gray", 0.0, 1.0),
        (percentile01(pretrain_abs_error), "Pretrain abs error", "magma", 0.0, 1.0),
        (clipped01(recon), "Recovered object raw clipped", "gray", 0.0, 1.0),
        (percentile01(recon), "Recovered object percentile", "gray", 0.0, 1.0),
        (clipped01(pred), "Predicted measurement clipped", "gray", 0.0, 1.0),
        (percentile01(pred_error), "Predicted-measurement error", "magma", 0.0, 1.0),
        (percentile01(object_error), "Gain-aligned object-input error", "magma", 0.0, 1.0),
    ]
    for idx, (arr, title, cmap, vmin, vmax) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    text_ax = fig.add_subplot(gs[0, 3])
    text_ax.axis("off")
    label = row.get("method_label", row.get("pretrain_method", ""))
    info = (
        f"Measurement-direct RCP rank {rank:03d}\n"
        f"{row.get('image')} | {row.get('pretrain_method')}\n\n"
        f"{wrapped(label)}\n\n"
        f"meas SSIM={short_float(row['ssim_meas_pred_vs_meas_f'])}\n"
        f"meas NRMSE={short_float(row['nrmse_meas_pred_vs_meas_f'])}\n"
        f"object-input SSIM={short_float(row['ssim_object_vs_input_f'])}\n"
        f"object-input NRMSE={short_float(row['nrmse_object_vs_input_f'])}\n"
        f"raw Seidel RMS={short_float(row['seidel_wavefront_rms_f'])}\n"
        f"final loss={short_float(row['final_loss_f'])}\n"
        f"pretrain loss={short_float(row['pretrain_final_loss_f'])}\n\n"
        "No synthetic measurement or GT Seidel was used."
    )
    text_ax.text(0.0, 1.0, info, va="top", ha="left", fontsize=10)

    coeff_ax = fig.add_subplot(gs[1, 3])
    colors = ["#5ab4ac" if value >= 0 else "#f46d43" for value in seidel]
    coeff_ax.axhline(0.0, color="#555555", linewidth=0.8)
    coeff_ax.bar(np.arange(len(COEFF_LABELS)), seidel, color=colors)
    coeff_ax.set_xticks(np.arange(len(COEFF_LABELS)), COEFF_LABELS, rotation=30, ha="right")
    coeff_ax.set_title("Raw recovered equivalent Seidel")
    coeff_ax.grid(axis="y", alpha=0.2)

    loss_ax = fig.add_subplot(gs[2, 3])
    pre = list(tensors.get("pretrain_history", []))
    joint = list(tensors.get("loss_history", []))
    if pre:
        loss_ax.plot(np.arange(len(pre)), pre, label="pretrain")
    if joint:
        offset = len(pre)
        loss_ax.plot(np.arange(len(joint)) + offset, joint, label="joint")
    loss_ax.set_yscale("log")
    loss_ax.set_title("Loss history")
    loss_ax.set_xlabel("step")
    loss_ax.grid(alpha=0.2)
    loss_ax.legend(fontsize=8)

    fig.suptitle(
        f"{row.get('image')} | {row.get('method_label')} | measurement-direct",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir = output_dir / "rcp_all" / str(row.get("pretrain_method")) / str(row.get("image"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rank{rank:03d}__{tag_text(row.get('image'))}__{tag_text(row.get('pretrain_method'))}__measurement_direct_rcp.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_overview(rcp_paths: list[Path], output_dir: Path) -> None:
    if not rcp_paths:
        return
    thumbs = []
    for path in rcp_paths:
        im = Image.open(path).convert("RGB")
        target_w = 560
        target_h = int(im.height * target_w / im.width)
        thumbs.append(im.resize((target_w, target_h)))
    cols = min(3, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    margin = 18
    cell_w = max(im.width for im in thumbs)
    cell_h = max(im.height for im in thumbs)
    canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * margin, rows * cell_h + (rows + 1) * margin), "white")
    for idx, im in enumerate(thumbs):
        x = margin + (idx % cols) * (cell_w + margin)
        y = margin + (idx // cols) * (cell_h + margin)
        canvas.paste(im, (x, y))
    canvas.save(output_dir / "RCP_measurement_direct_overview.png")


def make_metric_figure(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    labels = [f"{row['image']}\n{row['pretrain_method']}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(rows) * 1.2), 7), sharex=True)
    axes[0].bar(x, [row["ssim_meas_pred_vs_meas_f"] for row in rows], color="#5ab4ac")
    axes[0].set_ylabel("measurement SSIM")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(x, [row["nrmse_meas_pred_vs_meas_f"] for row in rows], color="#f46d43")
    axes[1].set_ylabel("measurement NRMSE")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    fig.suptitle("Measurement-direct fit quality")
    fig.tight_layout()
    (output_dir / "stats").mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "stats" / "measurement_fit_by_case.png", dpi=160)
    plt.close(fig)


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# Measurement-direct summary",
        "",
        "Input images were used directly as measurements. No synthetic measurement or GT Seidel was used.",
        "",
        "| image | method | meas SSIM | meas NRMSE | object-input SSIM | object-input NRMSE | raw Seidel RMS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['image']} | {row['pretrain_method']} | "
            f"{short_float(row['ssim_meas_pred_vs_meas_f'])} | "
            f"{short_float(row['nrmse_meas_pred_vs_meas_f'])} | "
            f"{short_float(row['ssim_object_vs_input_f'])} | "
            f"{short_float(row['nrmse_object_vs_input_f'])} | "
            f"{short_float(row['seidel_wavefront_rms_f'])} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--settings-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "cocoa_like_2d_mechanism",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings_manifest(args.settings_manifest)
    rows = enrich_rows(load_rows(args.output_root, args.prefix, settings))
    rows.sort(key=lambda row: (row["nrmse_meas_pred_vs_meas_f"], -row["ssim_meas_pred_vs_meas_f"]))

    comparison = comparison_by_case(rows)
    write_csv(comparison, stats_dir / "comparison_by_case.csv")
    write_csv(rows, stats_dir / "combined_measurement_direct_metrics.csv")
    write_csv(grouped_summary(rows, "pretrain_method"), stats_dir / "summary_by_method.csv")
    write_csv(grouped_summary(rows, "image"), stats_dir / "summary_by_image.csv")

    manifest: list[dict[str, Any]] = []
    rcp_paths: list[Path] = []
    for rank, row in enumerate(rows, start=1):
        path = render_case_rcp(output_root=args.output_root, prefix=args.prefix, row=row, output_dir=output_dir, rank=rank)
        rcp_paths.append(path)
        manifest.append(
            {
                "rank": rank,
                "image": row["image"],
                "pretrain_method": row["pretrain_method"],
                "ssim_meas_pred_vs_meas": row["ssim_meas_pred_vs_meas_f"],
                "nrmse_meas_pred_vs_meas": row["nrmse_meas_pred_vs_meas_f"],
                "rcp_path": str(path),
            }
        )
    write_csv(manifest, output_dir / "manifest.csv")
    make_overview(rcp_paths, output_dir)
    make_metric_figure(rows, output_dir)
    write_summary(rows, output_dir)

    print(f"[measurement-direct-rcp-done] rows={len(rows)} out={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
