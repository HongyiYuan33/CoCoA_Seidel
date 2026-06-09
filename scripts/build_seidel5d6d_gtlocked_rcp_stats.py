"""Build 4D/5D/6D GT-locked Seidel comparison stats and RCP panels."""

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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

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


DEFAULT_PREFIX = "seidel5d6d_gtlocked_tunedadam256_four_images_pre400_joint1000_20260609"
DEFAULT_BASELINE_RUN = "capacity4d_dirrms_tunedprior_size256_four_images_20260607__baseline"
CONVENTIONS = ("classical4d", "classical5d", "classical6d")
IMAGE_ORDER = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]
DIRECTION_ORDER = ["cocoa_signed", "signed_balanced"]
RMS_ORDER = [0.06, 0.20, 0.40]
LOWER_BETTER = {
    "operator_error_calibrated",
    "operator_error_phys_equiv",
    "operator_error_coord_diagnostic",
    "nrmse_recon_gain_vs_gt",
    "aligned_wavefront_error_physical",
    "aligned_coeff_relative_error_physical",
}


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
                seen.add(key)
                fieldnames.append(key)
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


def run_name(prefix: str, convention: str, baseline_run: str) -> str:
    if convention == "classical4d":
        return baseline_run
    return f"{prefix}__{convention}"


def eval_dir_name(convention: str) -> str:
    return "stage1_operator_eval_dim256"


def load_eval_rows(output_root: Path, prefix: str, baseline_run: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for convention in CONVENTIONS:
        run = run_name(prefix, convention, baseline_run)
        path = output_root / run / eval_dir_name(convention) / "seidel_physical_operator_metrics.csv"
        for row in read_csv(path):
            enriched = dict(row)
            enriched["comparison_convention"] = convention
            enriched["comparison_run"] = run
            enriched["target_rms_f"] = float(row["target_wavefront_rms"])
            rows.append(enriched)
    rows.sort(key=lambda row: (key_sort(key_for(row)), CONVENTIONS.index(row["comparison_convention"])))
    return rows


def keyed(rows: list[dict[str, Any]], label: str) -> dict[tuple[str, str, float], dict[str, Any]]:
    out: dict[tuple[str, str, float], dict[str, Any]] = {}
    for row in rows:
        key = key_for(row)
        if key in out:
            raise ValueError(f"Duplicate {label} row for {key}")
        out[key] = row
    return out


def mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else math.nan


def grouped_summary(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for group_key, group in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item.update(
            {
                "num_cases": len(group),
                "operator_error_calibrated_mean": mean([parse_float(r, "operator_error_calibrated") for r in group]),
                "ssim_mean": mean([parse_float(r, "ssim_recon_gain_vs_gt") for r in group]),
                "nrmse_mean": mean([parse_float(r, "nrmse_recon_gain_vs_gt") for r in group]),
                "aligned_wavefront_error_mean": mean([parse_float(r, "aligned_wavefront_error_physical") for r in group]),
                "aligned_coeff_relative_error_mean": mean([parse_float(r, "aligned_coeff_relative_error_physical") for r in group]),
            }
        )
        out.append(item)
    return out


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_conv = {conv: keyed([r for r in rows if r["comparison_convention"] == conv], conv) for conv in CONVENTIONS}
    out: list[dict[str, Any]] = []
    metrics = [
        "operator_error_calibrated",
        "operator_error_phys_equiv",
        "operator_error_coord_diagnostic",
        "ssim_recon_gain_vs_gt",
        "nrmse_recon_gain_vs_gt",
        "aligned_wavefront_error_physical",
        "aligned_coeff_relative_error_physical",
    ]
    for key in sorted(by_conv["classical4d"], key=key_sort):
        base = by_conv["classical4d"][key]
        row: dict[str, Any] = {
            "image": key[0],
            "direction": key[1],
            "target_wavefront_rms": key[2],
            "candidate_id": base["candidate_id"],
        }
        for conv in CONVENTIONS:
            current = by_conv[conv][key]
            row[f"{conv}_actual_wavefront_rms"] = current.get("actual_wavefront_rms")
            row[f"{conv}_seidel_gt"] = current.get("seidel_gt")
            for metric in metrics:
                row[f"{conv}_{metric}"] = current.get(metric, "")
                if conv != "classical4d":
                    delta_key = f"delta_{conv}_minus_4d_{metric}"
                    row[delta_key] = parse_float(current, metric) - parse_float(base, metric)
        out.append(row)
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
    aligned_key = "aligned_seidel_physical" if row.get("aligned_seidel_physical") else "seidel_final"
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
        "seidel_aligned": parse_vector(row[aligned_key]),
    }


def wrapped(text: str, width: int = 78) -> str:
    return "\n".join(wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def draw_variant(fig: plt.Figure, gridspec: Any, *, label: str, row: dict[str, Any], case: dict[str, Any]) -> None:
    left = gridspec[0].subgridspec(2, 3, wspace=0.08, hspace=0.16)
    images = [
        ("Sharp GT", case["gt"], "gray", normalize01),
        ("Measurement", case["meas"], "gray", normalize01),
        (f"{label} recon raw", case["recon"], "gray", normalize01),
        (f"{label} recon pct", case["recon"], "gray", percentile01),
        ("Predicted measurement", case["pred"], "gray", normalize01),
        ("Gain-aligned abs error", case["error"], "magma", percentile01),
    ]
    for idx, (title, arr, cmap, norm_fn) in enumerate(images):
        ax = fig.add_subplot(left[idx // 3, idx % 3])
        ax.imshow(norm_fn(arr), cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=8.2, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    right = gridspec[1].subgridspec(1, 2, width_ratios=[0.8, 1.25], wspace=0.22)
    ax_text = fig.add_subplot(right[0, 0])
    ax_text.axis("off")
    gt_rms = field_weighted_wavefront_rms(case["seidel_gt"])
    aligned_rms = field_weighted_wavefront_rms(case["seidel_aligned"])
    lines = [
        label,
        f"op_cal={short_float(parse_float(row, 'operator_error_calibrated'))}",
        f"SSIM={short_float(parse_float(row, 'ssim_recon_gain_vs_gt'))}",
        f"NRMSE={short_float(parse_float(row, 'nrmse_recon_gain_vs_gt'))}",
        f"aligned_WF={short_float(parse_float(row, 'aligned_wavefront_error_physical'))}",
        f"coeff_rel={short_float(parse_float(row, 'aligned_coeff_relative_error_physical'))}",
        f"target label={short_float(parse_float(row, 'target_wavefront_rms'))}",
        f"actual GT RMS={short_float(gt_rms)}",
        f"aligned rec RMS={short_float(aligned_rms)}",
        f"best_phys={row.get('best_physical_transform', '?')}",
        wrapped(str(row.get("candidate_id", "")), width=38),
    ]
    ax_text.text(0.0, 0.98, "\n".join(lines), ha="left", va="top", fontsize=8.2)

    ax_bar = fig.add_subplot(right[0, 1])
    x = np.arange(len(COEFF_LABELS), dtype=np.float64)
    width = 0.34
    ax_bar.bar(x - width / 2, case["seidel_gt"], width, label="GT", color="#55b8b0")
    ax_bar.bar(x + width / 2, case["seidel_aligned"], width, label="aligned recovered", color="#ef7d55")
    ax_bar.scatter(x, case["seidel_raw"], marker="x", color="black", s=22, label="raw", zorder=4)
    ax_bar.axhline(0.0, color="0.55", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(COEFF_LABELS, fontsize=7.8)
    ax_bar.set_ylabel("coefficient", fontsize=8)
    ax_bar.set_title(f"{label} Seidel coeffs", fontsize=8.8)
    ax_bar.grid(axis="y", alpha=0.22)
    ax_bar.legend(loc="upper right", fontsize=6.6, frameon=False)
    ax_bar.tick_params(axis="y", labelsize=7.4)


def make_rcp_panel(
    *,
    output_root: Path,
    prefix: str,
    baseline_run: str,
    rows_by_convention: dict[str, dict[tuple[str, str, float], dict[str, Any]]],
    key: tuple[str, str, float],
    out_dir: Path,
    rank: int,
) -> dict[str, Any]:
    image, direction, rms = key
    cases: dict[str, dict[str, Any]] = {}
    for conv in CONVENTIONS:
        row = rows_by_convention[conv][key]
        cases[conv] = load_case(output_root, run_name(prefix, conv, baseline_run), row)

    out_subdir = out_dir / "rcp_all" / image / direction / f"rms{tag_float(rms)}"
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"rank{rank:03d}__{image}__{direction}__rms{tag_float(rms)}__4d5d6d_RCP.png"

    fig = plt.figure(figsize=(22, 15.2), dpi=140)
    outer = fig.add_gridspec(
        4,
        2,
        height_ratios=[0.23, 1.0, 1.0, 1.0],
        width_ratios=[1.08, 1.0],
        left=0.024,
        right=0.988,
        top=0.975,
        bottom=0.045,
        hspace=0.17,
        wspace=0.07,
    )
    ax_header = fig.add_subplot(outer[0, :])
    ax_header.axis("off")
    base = rows_by_convention["classical4d"][key]
    row5 = rows_by_convention["classical5d"][key]
    row6 = rows_by_convention["classical6d"][key]
    header = (
        f"GT-locked 4D/5D/6D RCP | {image} | {direction} | target RMS label={rms:g} | "
        f"op 4D={short_float(parse_float(base, 'operator_error_calibrated'))}, "
        f"5D={short_float(parse_float(row5, 'operator_error_calibrated'))}, "
        f"6D={short_float(parse_float(row6, 'operator_error_calibrated'))}"
    )
    ax_header.text(0.0, 0.92, header, ha="left", va="top", fontsize=13, fontweight="bold")
    ax_header.text(
        0.0,
        0.38,
        "5D/6D GT front four are copied exactly from the 4D baseline; added W311/Wd are not followed by RMS rescaling.",
        ha="left",
        va="top",
        fontsize=9,
    )

    labels = {
        "classical4d": "4D TunedAdam baseline",
        "classical5d": "5D GT-locked",
        "classical6d": "6D GT-locked",
    }
    for idx, conv in enumerate(CONVENTIONS, start=1):
        draw_variant(
            fig,
            (outer[idx, 0], outer[idx, 1]),
            label=labels[conv],
            row=rows_by_convention[conv][key],
            case=cases[conv],
        )
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "rank": rank,
        "image": image,
        "direction": direction,
        "target_rms": rms,
        "candidate_id": base["candidate_id"],
        "classical4d_operator_error_calibrated": base.get("operator_error_calibrated", ""),
        "classical5d_operator_error_calibrated": row5.get("operator_error_calibrated", ""),
        "classical6d_operator_error_calibrated": row6.get("operator_error_calibrated", ""),
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
    canvas = Image.new("RGB", (target_width, sum(img.height for img in images) + gap * (len(images) - 1)), "white")
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_metric_plot(rows: list[dict[str, Any]], out_dir: Path, metric: str, title: str) -> None:
    summary = grouped_summary(rows, ["comparison_convention", "direction", "target_wavefront_rms"])
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    width = 0.23
    positions = np.arange(len(DIRECTION_ORDER) * len(RMS_ORDER), dtype=np.float64)
    labels = [f"{direction}\n{rms:g}" for direction in DIRECTION_ORDER for rms in RMS_ORDER]
    for ci, conv in enumerate(CONVENTIONS):
        vals = []
        for direction in DIRECTION_ORDER:
            for rms in RMS_ORDER:
                match = [
                    row
                    for row in summary
                    if row["comparison_convention"] == conv
                    and row["direction"] == direction
                    and abs(float(row["target_wavefront_rms"]) - rms) < 1e-6
                ]
                vals.append(float(match[0][f"{metric}_mean"]) if match else math.nan)
        ax.bar(positions + (ci - 1) * width, vals, width, label=conv)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric}_by_direction_rms.png", dpi=170)
    plt.close(fig)


def write_summary(out_dir: Path, summary_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> None:
    overall = [row for row in summary_rows if set(row) >= {"comparison_convention"}]
    best = min(overall, key=lambda row: float(row["operator_error_calibrated_mean"])) if overall else None
    lines = [
        "# GT-Locked 4D/5D/6D TunedAdam-256 Summary",
        "",
        f"RCP panels: {len(manifest_rows)}",
        "",
        "| convention | cases | mean operator | mean SSIM | mean NRMSE | mean aligned WF |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['comparison_convention']} | {row['num_cases']} | "
            f"{short_float(row['operator_error_calibrated_mean'])} | "
            f"{short_float(row['ssim_mean'])} | "
            f"{short_float(row['nrmse_mean'])} | "
            f"{short_float(row['aligned_wavefront_error_mean'])} |"
        )
    if best:
        lines += [
            "",
            f"Best mean operator error: `{best['comparison_convention']}` "
            f"({short_float(best['operator_error_calibrated_mean'])}).",
        ]
    lines += [
        "",
        "Key outputs:",
        "- `combined_4d5d6d_metrics.csv`",
        "- `comparison_by_case.csv`",
        "- `summary_by_convention.csv`",
        "- `summary_by_convention_direction_rms.csv`",
        "- `summary_by_image.csv`",
        "- `manifest.csv` and `RCP_gtlocked_4d5d6d_overview.png`",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cocoa_like_2d_mechanism"))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--baseline-run", default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-rcp", action="store_true")
    args = parser.parse_args(argv)

    output_root = args.output_root
    out_dir = args.output_dir or output_root / f"{args.prefix}_rcp_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_rows(output_root, args.prefix, args.baseline_run)
    write_csv(rows, stats_dir / "combined_4d5d6d_metrics.csv")
    comp_rows = comparison_rows(rows)
    write_csv(comp_rows, stats_dir / "comparison_by_case.csv")
    summary_by_convention = grouped_summary(rows, ["comparison_convention"])
    write_csv(summary_by_convention, stats_dir / "summary_by_convention.csv")
    write_csv(grouped_summary(rows, ["comparison_convention", "direction", "target_wavefront_rms"]), stats_dir / "summary_by_convention_direction_rms.csv")
    write_csv(grouped_summary(rows, ["comparison_convention", "image"]), stats_dir / "summary_by_image.csv")
    make_metric_plot(rows, stats_dir, "operator_error_calibrated", "Mean operator error by direction/RMS")
    make_metric_plot(rows, stats_dir, "ssim", "Mean SSIM by direction/RMS")
    make_metric_plot(rows, stats_dir, "nrmse", "Mean NRMSE by direction/RMS")

    manifest_rows: list[dict[str, Any]] = []
    if not args.skip_rcp:
        by_conv = {conv: keyed([row for row in rows if row["comparison_convention"] == conv], conv) for conv in CONVENTIONS}
        keys = sorted(by_conv["classical4d"], key=key_sort)
        for rank, key in enumerate(keys, start=1):
            manifest_rows.append(
                make_rcp_panel(
                    output_root=output_root,
                    prefix=args.prefix,
                    baseline_run=args.baseline_run,
                    rows_by_convention=by_conv,
                    key=key,
                    out_dir=out_dir,
                    rank=rank,
                )
            )
        write_csv(manifest_rows, out_dir / "manifest.csv")
        overview = [
            row
            for row in manifest_rows
            if (row["image"], row["direction"], round(float(row["target_rms"]), 6))
            in {
                ("Test_figure_1", "cocoa_signed", 0.06),
                ("Test_figure_1", "cocoa_signed", 0.2),
                ("Test_figure_1", "cocoa_signed", 0.4),
                ("dendrites_dense", "signed_balanced", 0.06),
                ("dendrites_dense", "signed_balanced", 0.2),
                ("dendrites_dense", "signed_balanced", 0.4),
            }
        ]
        make_overview(overview or manifest_rows[:6], out_dir / "RCP_gtlocked_4d5d6d_overview.png")
    write_summary(out_dir, summary_by_convention, manifest_rows)
    print(f"[done] wrote stats to {stats_dir}")
    print(f"[done] wrote RCP output to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
