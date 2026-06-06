#!/usr/bin/env python3
"""Plot lambda=0 vs lambda=10000 Seidel RMS floor comparison."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fnum(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def mean(values: Iterable[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def fmt(value: float, ndigits: int = 3) -> str:
    return "nan" if math.isnan(value) else f"{value:.{ndigits}f}"


def load_rows(csv_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    numeric_fields = [
        "lambda",
        "target_wavefront_rms",
        "wavefront_gt_rms",
        "wavefront_recovered_rms",
        "wavefront_recovered_over_gt_rms",
        "operator_error_calibrated",
        "ssim_recon_gain_vs_gt",
        "nrmse_recon_gain_vs_gt",
    ]
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, object] = dict(row)
            for field in numeric_fields:
                parsed[field] = fnum(row.get(field))
            rows.append(parsed)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = load_rows(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    images_order = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]
    images = [x for x in images_order if any(r["image"] == x for r in rows)]
    for x in sorted({str(r["image"]) for r in rows}):
        if x not in images:
            images.append(x)

    directions = ["cocoa_signed", "signed_balanced"]
    directions = [x for x in directions if any(r["direction"] == x for r in rows)]
    rms_levels = sorted(
        {
            float(r["target_wavefront_rms"])
            for r in rows
            if not math.isnan(float(r["target_wavefront_rms"]))
        }
    )
    lambdas = [0.0, 10000.0]
    colors = {0.0: "#555555", 10000.0: "#d62728"}
    labels = {0.0: "lambda 0", 10000.0: "lambda 10000"}
    markers = {0.0: "o", 10000.0: "s"}

    def select(
        image: str | None = None,
        direction: str | None = None,
        lam: float | None = None,
        rms: float | None = None,
    ) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for row in rows:
            if image is not None and row["image"] != image:
                continue
            if direction is not None and row["direction"] != direction:
                continue
            if lam is not None and abs(float(row["lambda"]) - lam) > 1e-6:
                continue
            if rms is not None and abs(float(row["target_wavefront_rms"]) - rms) > 1e-9:
                continue
            out.append(row)
        return out

    def series(image: str, direction: str, lam: float, field: str) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for rms in rms_levels:
            vals = [float(r[field]) for r in select(image=image, direction=direction, lam=lam, rms=rms)]
            xs.append(rms)
            ys.append(mean(vals))
        return xs, ys

    def setup_grid(title: str, ylabel: str, ylim: tuple[float, float] | None = None):
        fig, axes = plt.subplots(
            len(images),
            len(directions),
            figsize=(15.5, 3.3 * len(images)),
            sharex=True,
        )
        axes = np.asarray(axes)
        if axes.ndim == 1:
            if len(images) == 1:
                axes = axes[None, :]
            else:
                axes = axes[:, None]
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.995)
        for i, image in enumerate(images):
            for j, direction in enumerate(directions):
                ax = axes[i, j]
                ax.grid(True, alpha=0.25)
                ax.set_title(f"{image} | {direction}", fontsize=10)
                ax.set_ylabel(ylabel)
                if ylim:
                    ax.set_ylim(*ylim)
                ax.set_xticks(rms_levels)
                ax.set_xticklabels([f"{x:.2f}" for x in rms_levels])
        return fig, axes

    # 1. Recovered/GT ratio.
    fig, axes = setup_grid(
        "Recovered / GT Seidel wavefront RMS: lambda 0 vs 10000",
        "recovered / GT RMS",
        ylim=(-0.03, 1.12),
    )
    for i, image in enumerate(images):
        for j, direction in enumerate(directions):
            ax = axes[i, j]
            ax.axhline(
                1.0,
                color="#111111",
                lw=1.0,
                alpha=0.55,
                label="GT ratio 1.0" if i == 0 and j == 0 else None,
            )
            ax.axhline(
                0.8,
                color="#777777",
                lw=1.0,
                alpha=0.45,
                ls="--",
                label="target 0.8" if i == 0 and j == 0 else None,
            )
            for lam in lambdas:
                xs, ys = series(image, direction, lam, "wavefront_recovered_over_gt_rms")
                ax.plot(
                    xs,
                    ys,
                    color=colors[lam],
                    marker=markers[lam],
                    lw=2.0,
                    ms=5,
                    label=labels[lam] if i == 0 and j == 0 else None,
                )
                for x, y in zip(xs, ys):
                    if not math.isnan(y):
                        ax.text(x, y + 0.025, f"{y:.2f}", fontsize=7, ha="center", color=colors[lam])
    for ax in axes[-1, :]:
        ax.set_xlabel("target / GT wavefront RMS")
    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out_dir / "01_recovered_over_gt_rms_by_image_direction.png", dpi=200)
    plt.close(fig)

    # 2. Absolute recovered RMS against GT RMS.
    fig, axes = setup_grid("Recovered Seidel wavefront RMS vs GT RMS", "recovered wavefront RMS")
    max_rms = max(rms_levels) * 1.08
    for i, image in enumerate(images):
        for j, direction in enumerate(directions):
            ax = axes[i, j]
            ax.plot(
                [0, max_rms],
                [0, max_rms],
                color="#111111",
                lw=1.0,
                alpha=0.55,
                label="y = GT" if i == 0 and j == 0 else None,
            )
            ax.plot(
                [0, max_rms],
                [0, 0.8 * max_rms],
                color="#777777",
                lw=1.0,
                alpha=0.45,
                ls="--",
                label="0.8 x GT" if i == 0 and j == 0 else None,
            )
            for lam in lambdas:
                xs: list[float] = []
                ys: list[float] = []
                for rms in rms_levels:
                    matches = select(image=image, direction=direction, lam=lam, rms=rms)
                    xs.append(mean([float(r["wavefront_gt_rms"]) for r in matches]))
                    ys.append(mean([float(r["wavefront_recovered_rms"]) for r in matches]))
                ax.plot(
                    xs,
                    ys,
                    color=colors[lam],
                    marker=markers[lam],
                    lw=2.0,
                    ms=5,
                    label=labels[lam] if i == 0 and j == 0 else None,
                )
            ax.set_xlim(0, max_rms)
            ax.set_ylim(0, max_rms)
    for ax in axes[-1, :]:
        ax.set_xlabel("GT wavefront RMS")
    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out_dir / "02_recovered_rms_vs_gt_rms_by_image_direction.png", dpi=200)
    plt.close(fig)

    # 3. Operator error.
    fig, axes = setup_grid("Operator error calibrated: lambda 0 vs 10000", "operator_error_calibrated")
    for i, image in enumerate(images):
        for j, direction in enumerate(directions):
            ax = axes[i, j]
            for lam in lambdas:
                xs, ys = series(image, direction, lam, "operator_error_calibrated")
                ax.plot(
                    xs,
                    ys,
                    color=colors[lam],
                    marker=markers[lam],
                    lw=2.0,
                    ms=5,
                    label=labels[lam] if i == 0 and j == 0 else None,
                )
                for x, y in zip(xs, ys):
                    if not math.isnan(y):
                        ax.text(x, y + 0.006, f"{y:.3f}", fontsize=7, ha="center", color=colors[lam])
    for ax in axes[-1, :]:
        ax.set_xlabel("target / GT wavefront RMS")
    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out_dir / "03_operator_error_calibrated_by_image_direction.png", dpi=200)
    plt.close(fig)

    # 4. Object SSIM.
    if any(not math.isnan(float(r["ssim_recon_gain_vs_gt"])) for r in rows):
        fig, axes = setup_grid("Object SSIM: lambda 0 vs 10000", "SSIM recon gain vs GT", ylim=(0.0, 1.02))
        for i, image in enumerate(images):
            for j, direction in enumerate(directions):
                ax = axes[i, j]
                for lam in lambdas:
                    xs, ys = series(image, direction, lam, "ssim_recon_gain_vs_gt")
                    ax.plot(
                        xs,
                        ys,
                        color=colors[lam],
                        marker=markers[lam],
                        lw=2.0,
                        ms=5,
                        label=labels[lam] if i == 0 and j == 0 else None,
                    )
        for ax in axes[-1, :]:
            ax.set_xlabel("target / GT wavefront RMS")
        fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(args.out_dir / "04_object_ssim_by_image_direction.png", dpi=200)
        plt.close(fig)

    # 5/6. Delta heatmaps.
    combos = [(image, direction) for image in images for direction in directions]
    heat_specs = [
        (
            "wavefront_recovered_over_gt_rms",
            "Delta recovered/GT RMS ratio: lambda10000 - lambda0",
            "05_delta_recovered_over_gt_ratio_heatmap.png",
        ),
        (
            "operator_error_calibrated",
            "Delta operator_error_calibrated: lambda10000 - lambda0",
            "06_delta_operator_error_heatmap.png",
        ),
    ]
    for field, title, filename in heat_specs:
        data = np.full((len(combos), len(rms_levels)), np.nan)
        for row_idx, (image, direction) in enumerate(combos):
            for col_idx, rms in enumerate(rms_levels):
                val_10000 = mean(
                    [float(r[field]) for r in select(image=image, direction=direction, lam=10000.0, rms=rms)]
                )
                val_0 = mean([float(r[field]) for r in select(image=image, direction=direction, lam=0.0, rms=rms)])
                data[row_idx, col_idx] = val_10000 - val_0
        vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
        fig, ax = plt.subplots(figsize=(11.5, 5.8))
        imh = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xticks(range(len(rms_levels)))
        ax.set_xticklabels([f"{x:.2f}" for x in rms_levels])
        ax.set_yticks(range(len(combos)))
        ax.set_yticklabels([f"{image} | {direction}" for image, direction in combos], fontsize=8)
        ax.set_xlabel("target / GT wavefront RMS")
        for row_idx in range(data.shape[0]):
            for col_idx in range(data.shape[1]):
                value = float(data[row_idx, col_idx])
                if not math.isnan(value):
                    ax.text(col_idx, row_idx, f"{value:+.3f}", ha="center", va="center", fontsize=7, color="black")
        fig.colorbar(imh, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        fig.savefig(args.out_dir / filename, dpi=200)
        plt.close(fig)

    summary_rows: list[dict[str, object]] = []
    for lam in lambdas:
        for direction in directions:
            for rms in rms_levels:
                matches = select(direction=direction, lam=lam, rms=rms)
                summary_rows.append(
                    {
                        "lambda": lam,
                        "direction": direction,
                        "target_wavefront_rms": rms,
                        "mean_recovered_over_gt": mean(
                            [float(r["wavefront_recovered_over_gt_rms"]) for r in matches]
                        ),
                        "mean_recovered_rms": mean([float(r["wavefront_recovered_rms"]) for r in matches]),
                        "mean_gt_rms": mean([float(r["wavefront_gt_rms"]) for r in matches]),
                        "mean_operator_error_calibrated": mean(
                            [float(r["operator_error_calibrated"]) for r in matches]
                        ),
                        "mean_ssim": mean([float(r["ssim_recon_gain_vs_gt"]) for r in matches]),
                        "mean_nrmse": mean([float(r["nrmse_recon_gain_vs_gt"]) for r in matches]),
                        "n": len(matches),
                    }
                )
    with (args.out_dir / "summary_by_lambda_direction_rms.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    overall: dict[float, dict[str, float]] = {}
    for lam in lambdas:
        matches = select(lam=lam)
        overall[lam] = {
            "ratio": mean([float(r["wavefront_recovered_over_gt_rms"]) for r in matches]),
            "op": mean([float(r["operator_error_calibrated"]) for r in matches]),
            "ssim": mean([float(r["ssim_recon_gain_vs_gt"]) for r in matches]),
            "nrmse": mean([float(r["nrmse_recon_gain_vs_gt"]) for r in matches]),
        }

    match_keys = sorted({(str(r["image"]), str(r["direction"]), float(r["target_wavefront_rms"])) for r in rows})
    matched = 0
    ratio_improved = 0
    op_improved = 0
    both_improved = 0
    for image, direction, rms in match_keys:
        rows_0 = select(image=image, direction=direction, lam=0.0, rms=rms)
        rows_10000 = select(image=image, direction=direction, lam=10000.0, rms=rms)
        if not rows_0 or not rows_10000:
            continue
        matched += 1
        ratio_delta = mean([float(r["wavefront_recovered_over_gt_rms"]) for r in rows_10000]) - mean(
            [float(r["wavefront_recovered_over_gt_rms"]) for r in rows_0]
        )
        op_delta = mean([float(r["operator_error_calibrated"]) for r in rows_10000]) - mean(
            [float(r["operator_error_calibrated"]) for r in rows_0]
        )
        if ratio_delta > 0:
            ratio_improved += 1
        if op_delta < 0:
            op_improved += 1
        if ratio_delta > 0 and op_delta < 0:
            both_improved += 1

    report: list[str] = []
    report.append("# Lambda 0 vs 10000 comparison\n\n")
    report.append(f"Input CSV: `{args.csv}`\n\n")
    report.append("## Overall mean\n\n")
    report.append("| lambda | recovered/GT RMS | operator_error_calibrated | SSIM | NRMSE |\n")
    report.append("|---:|---:|---:|---:|---:|\n")
    for lam in lambdas:
        values = overall[lam]
        report.append(
            f"| {lam:.0f} | {fmt(values['ratio'])} | {fmt(values['op'])} | "
            f"{fmt(values['ssim'])} | {fmt(values['nrmse'])} |\n"
        )
    report.append("\n")
    report.append("## Matched case counts\n\n")
    report.append(f"Matched image-direction-RMS cases: {matched}\n\n")
    report.append(f"Recovered/GT RMS higher at lambda10000: {ratio_improved}/{matched}\n\n")
    report.append(f"Operator error lower at lambda10000: {op_improved}/{matched}\n\n")
    report.append(f"Both higher RMS ratio and lower operator error: {both_improved}/{matched}\n\n")
    report.append("## Mean by target RMS and direction\n")
    for direction in directions:
        report.append(f"\n### {direction}\n\n")
        report.append("| RMS | ratio l0 | ratio l10000 | op l0 | op l10000 | SSIM l0 | SSIM l10000 |\n")
        report.append("|---:|---:|---:|---:|---:|---:|---:|\n")
        for rms in rms_levels:
            rows_0 = select(direction=direction, lam=0.0, rms=rms)
            rows_10000 = select(direction=direction, lam=10000.0, rms=rms)
            report.append(
                f"| {rms:.2f} | "
                f"{fmt(mean([float(r['wavefront_recovered_over_gt_rms']) for r in rows_0]))} | "
                f"{fmt(mean([float(r['wavefront_recovered_over_gt_rms']) for r in rows_10000]))} | "
                f"{fmt(mean([float(r['operator_error_calibrated']) for r in rows_0]))} | "
                f"{fmt(mean([float(r['operator_error_calibrated']) for r in rows_10000]))} | "
                f"{fmt(mean([float(r['ssim_recon_gain_vs_gt']) for r in rows_0]))} | "
                f"{fmt(mean([float(r['ssim_recon_gain_vs_gt']) for r in rows_10000]))} |\n"
            )

    (args.out_dir / "comparison_report.md").write_text("".join(report))
    print(f"wrote {args.out_dir}")
    print(f"overall lambda0 ratio={overall[0.0]['ratio']:.4f} op={overall[0.0]['op']:.4f}")
    print(f"overall lambda10000 ratio={overall[10000.0]['ratio']:.4f} op={overall[10000.0]['op']:.4f}")
    print(f"matched={matched} ratio_improved={ratio_improved} op_improved={op_improved} both={both_improved}")


if __name__ == "__main__":
    main()
