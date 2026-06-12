"""Build RCP-style comparison panels and stats for even-flip oracle controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "cocoa_like_2d_mechanism"
OLD_RUN = "seidel_oracle_controls_4D_6D_4imgs_2dirs_rms006_020_040_seed0_noRMS_pre400_joint1000_20260607"
NEW_RUN = "seidel_even_flip_fixed_oracle_controls_4D_6D_4imgs_2dirs_rms006_020_040_seed0_noRMS_pre400_joint1000_2gpu_20260608"
COEFF_NAMES = ["W040", "W131", "W222", "W220", "W311", "Wd"]
EVEN_INDICES = [0, 2, 3, 5]
ODD_INDICES = [1, 4]
# Recovered-coefficient overlays come from the reference oracle run's combined
# evaluator CSV; these are the two oracle modes that actually optimize Seidel.
RECOVERED_MODES = [
    ("joint_no_RMS", "recovered joint no-RMS (aligned)", "#4C78A8"),
    ("object_gt_fixed", "recovered object-GT oracle (aligned)", "#9467BD"),
]
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)


def tag_float(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def parse_vector(value: str | list[float]) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(v) for v in json.loads(value)]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def l2_distance(a: list[float], b: list[float]) -> float:
    return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def build_recovered_lookup(old_root: Path) -> dict[tuple[str, str, str, str], dict[str, list[float]]]:
    path = old_root / "oracle_controls_evaluator_combined.csv"
    lookup: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(dict)
    if not path.is_file():
        print(f"[warn] recovered-coefficient source missing: {path}", file=sys.stderr)
        return lookup
    wanted = {mode for mode, _label, _color in RECOVERED_MODES}
    for row in read_csv(path):
        if row.get("oracle_mode") not in wanted:
            continue
        raw = row.get("aligned_seidel_physical") or row.get("seidel_final")
        if not raw:
            continue
        key = (
            row["seidel_convention"],
            row["image"],
            row["candidate_id"],
            str(int(float(row.get("seed") or 0))),
        )
        lookup[key][row["oracle_mode"]] = parse_vector(raw)
    return lookup


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def old_comparison_path(old_root: Path, row: dict[str, str]) -> Path:
    return (
        old_root
        / row["seidel_convention"]
        / f"seed{int(row.get('seed') or 0)}"
        / f"{row['image']}__{row['candidate_id']}"
        / "seidel_gt_fixed"
        / "seidel_gt_fixed"
        / "comparison.png"
    )


def new_comparison_path(new_root: Path, row: dict[str, str]) -> Path:
    return (
        new_root
        / row["seidel_convention"]
        / f"seed{int(row.get('seed') or 0)}"
        / f"{row['image']}__{row['candidate_id']}"
        / "even_flip_fixed"
        / "comparison.png"
    )


def output_case_path(out_root: Path, row: dict[str, str]) -> Path:
    rms_tag = f"rms{tag_float(float(row['target_wavefront_rms']))}"
    name = (
        f"even_flip_vs_gt_fixed__{row['seidel_convention']}__{row['image']}__"
        f"{row['direction']}__seed{int(row.get('seed') or 0)}__{rms_tag}__RCP_compare.png"
    )
    return out_root / row["seidel_convention"] / row["image"] / row["direction"] / rms_tag / name


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def add_panel_title(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, pad=8)
    ax.axis("off")


def plot_coefficients(
    ax: plt.Axes,
    gt: list[float],
    even: list[float],
    recovered: dict[str, list[float]] | None = None,
) -> None:
    x = np.arange(len(COEFF_NAMES))
    ax.axhline(0, color="#222222", linewidth=0.7)
    if not recovered:
        width = 0.36
        colors_gt = ["#4C9A8B" if idx in EVEN_INDICES else "#5F6FB3" for idx in range(6)]
        colors_even = ["#E36C4D" if idx in EVEN_INDICES else "#5F6FB3" for idx in range(6)]
        ax.bar(x - width / 2, gt, width, color=colors_gt, label="GT / GT-fixed")
        ax.bar(x + width / 2, even, width, color=colors_even, label="even-flip fixed")
        legend_fontsize = 8
    else:
        series: list[tuple[list[float], str, str]] = [
            (gt, "GT / GT-fixed", "#4C9A8B"),
            (even, "even-flip fixed", "#E36C4D"),
        ]
        for mode, label, color in RECOVERED_MODES:
            if mode in recovered:
                series.append((recovered[mode], label, color))
        width = 0.8 / len(series)
        offset0 = -0.4 + width / 2
        for idx, (values, label, color) in enumerate(series):
            ax.bar(x + offset0 + idx * width, values, width, color=color, label=label)
        legend_fontsize = 7
    ax.set_xticks(x, COEFF_NAMES, fontsize=8)
    ax.set_title("Seidel coefficients", fontsize=10)
    ax.set_ylabel("coefficient", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=legend_fontsize, loc="best")


def render_metric_card(
    ax: plt.Axes,
    row: dict[str, str],
    op_row: dict[str, str] | None,
    recovered: dict[str, list[float]] | None = None,
) -> None:
    ax.axis("off")
    lines = [
        "Metric card",
        f"case: {row['seidel_convention']} / {row['image']} / {row['candidate_id']}",
        f"GT-fixed SSIM: {float(row['gt_fixed_ssim']):.4f}",
        f"even-flip SSIM: {float(row['even_flip_ssim']):.4f}",
        f"delta SSIM: {float(row['delta_ssim_even_minus_gtfixed']):+.4f}",
        f"GT-fixed NRMSE: {float(row['gt_fixed_nrmse']):.4f}",
        f"even-flip NRMSE: {float(row['even_flip_nrmse']):.4f}",
        f"delta NRMSE: {float(row['delta_nrmse_even_minus_gtfixed']):+.4f}",
        f"even-flip rel WF error: {float(row['even_flip_relative_wavefront_error']):.4f}",
        f"even-flip coeff L2: {float(row['even_flip_l2_seidel_vs_gt']):.4f}",
    ]
    if op_row:
        lines.extend(
            [
                f"operator calibrated: {float(op_row['operator_error_calibrated']):.3e}",
                f"operator phys-equiv: {float(op_row['operator_error_phys_equiv']):.3e}",
                f"best physical transform: {op_row['best_physical_transform']}",
            ]
        )
    fontsize = 9.0
    linespacing = 1.35
    if recovered is not None:
        fontsize = 8.0
        linespacing = 1.25
        if recovered:
            gt = parse_vector(row["seidel_gt"])
            even = parse_vector(row["even_flip_seidel"])
            for mode, _label, _color in RECOVERED_MODES:
                vec = recovered.get(mode)
                if vec is None:
                    continue
                d_gt = l2_distance(vec, gt)
                d_even = l2_distance(vec, even)
                verdict = "GT" if d_gt <= d_even else "even-flip"
                lines.append(
                    f"rec {mode}: L2gt={d_gt:.3f} L2flip={d_even:.3f} -> {verdict}"
                )
        else:
            lines.append("recovered: n/a")
    ax.text(
        0.02,
        0.96,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=fontsize,
        family="monospace",
        linespacing=linespacing,
    )


def build_operator_lookup(new_root: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    lookup: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for convention in ("classical4d", "classical6d"):
        path = new_root / f"operator_eval_{convention}_dim256" / "seidel_physical_operator_metrics.csv"
        if not path.is_file():
            continue
        for row in read_csv(path):
            key = (
                row["seidel_convention"],
                row["image"],
                row["candidate_id"],
                str(int(float(row.get("seed") or 0))),
            )
            lookup[key] = row
    return lookup


def build_case_rcp(
    row: dict[str, str],
    *,
    old_root: Path,
    new_root: Path,
    out_root: Path,
    operator_lookup: dict[tuple[str, str, str, str], dict[str, str]],
    recovered_lookup: dict[tuple[str, str, str, str], dict[str, list[float]]] | None = None,
) -> Path | None:
    old_png = old_comparison_path(old_root, row)
    new_png = new_comparison_path(new_root, row)
    if not old_png.is_file() or not new_png.is_file():
        return None
    old_img = load_rgb(old_png)
    new_img = load_rgb(new_png)
    key = (
        row["seidel_convention"],
        row["image"],
        row["candidate_id"],
        str(int(row.get("seed") or 0)),
    )
    op_row = operator_lookup.get(key)
    gt = parse_vector(row["seidel_gt"])
    even = parse_vector(row["even_flip_seidel"])
    recovered: dict[str, list[float]] | None = None
    if recovered_lookup is not None:
        recovered = recovered_lookup.get(key, {})
        if not recovered:
            print(f"[warn] no recovered coefficients for {key}", file=sys.stderr)

    fig = plt.figure(figsize=(17, 10.5))
    grid = fig.add_gridspec(3, 2, height_ratios=[0.08, 0.66, 0.26], hspace=0.22, wspace=0.08)
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.55,
        (
            f"GT-fixed vs even-flip fixed Seidel | {row['seidel_convention']} | "
            f"{row['image']} | {row['candidate_id']}"
        ),
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )
    ax_old = fig.add_subplot(grid[1, 0])
    ax_new = fig.add_subplot(grid[1, 1])
    ax_old.imshow(old_img)
    add_panel_title(
        ax_old,
        "Reference: Seidel fixed to GT",
        f"SSIM={float(row['gt_fixed_ssim']):.4f}, NRMSE={float(row['gt_fixed_nrmse']):.4f}",
    )
    ax_new.imshow(new_img)
    add_panel_title(
        ax_new,
        "New: even terms sign-flipped, odd terms GT",
        f"SSIM={float(row['even_flip_ssim']):.4f}, NRMSE={float(row['even_flip_nrmse']):.4f}",
    )
    coeff_ax = fig.add_subplot(grid[2, 0])
    card_ax = fig.add_subplot(grid[2, 1])
    plot_coefficients(coeff_ax, gt, even, recovered=recovered)
    render_metric_card(card_ax, row, op_row, recovered=recovered)
    out_path = output_case_path(out_root, row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_contact_sheet(paths: list[Path], out_path: Path, *, title: str) -> None:
    if not paths:
        return
    thumb_w = 420
    pad = 16
    label_h = 42
    title_h = 54
    cols = 4
    thumbs: list[tuple[Path, Image.Image]] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        scale = thumb_w / img.width
        thumb_h = int(img.height * scale)
        thumbs.append((path, img.resize((thumb_w, thumb_h), RESAMPLE_LANCZOS)))
    row_h = max(img.height for _, img in thumbs[:cols]) + label_h + pad
    rows = math.ceil(len(thumbs) / cols)
    sheet_w = cols * thumb_w + (cols + 1) * pad
    sheet_h = title_h + rows * row_h + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        font_label = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
    draw.text((pad, 16), title, fill=(25, 25, 25), font=font_title)
    for idx, (path, img) in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = pad + col * (thumb_w + pad)
        y = title_h + row * row_h
        sheet.paste(img, (x, y))
        label = path.stem.replace("even_flip_vs_gt_fixed__", "").replace("__RCP_compare", "")
        draw.text((x, y + img.height + 6), label[:66], fill=(35, 35, 35), font=font_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def group_mean(rows: list[dict[str, str]], key: str) -> dict[tuple[str, str, float], float]:
    groups: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["seidel_convention"], row["direction"], float(row["target_wavefront_rms"]))].append(
            float(row[key])
        )
    return {group: float(np.mean(values)) for group, values in groups.items()}


def plot_delta_metric(rows: list[dict[str, str]], out_path: Path, *, key: str, ylabel: str, title: str) -> None:
    means = group_mean(rows, key)
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        ("classical4d", "cocoa_signed"): ("#2C7FB8", "o", "-"),
        ("classical4d", "signed_balanced"): ("#2C7FB8", "s", "--"),
        ("classical6d", "cocoa_signed"): ("#D95F0E", "o", "-"),
        ("classical6d", "signed_balanced"): ("#D95F0E", "s", "--"),
    }
    for (conv, direction), style in styles.items():
        xs = sorted({rms for c, d, rms in means if c == conv and d == direction})
        ys = [means[(conv, direction, x)] for x in xs]
        if xs:
            ax.plot(xs, ys, color=style[0], marker=style[1], linestyle=style[2], label=f"{conv} {direction}")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("target wavefront RMS (waves)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_old_new_metric(rows: list[dict[str, str]], out_path: Path, *, old_key: str, new_key: str, ylabel: str, title: str) -> None:
    old_means = group_mean(rows, old_key)
    new_means = group_mean(rows, new_key)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = {"classical4d": "#2C7FB8", "classical6d": "#D95F0E"}
    markers = {"cocoa_signed": "o", "signed_balanced": "s"}
    for conv in sorted(colors):
        for direction in sorted(markers):
            xs = sorted({rms for c, d, rms in old_means if c == conv and d == direction})
            if not xs:
                continue
            ax.plot(
                xs,
                [old_means[(conv, direction, x)] for x in xs],
                color=colors[conv],
                marker=markers[direction],
                linestyle=":",
                alpha=0.75,
                label=f"{conv} {direction} GT-fixed",
            )
            ax.plot(
                xs,
                [new_means[(conv, direction, x)] for x in xs],
                color=colors[conv],
                marker=markers[direction],
                linestyle="-",
                label=f"{conv} {direction} even-flip",
            )
    ax.set_xlabel("target wavefront RMS (waves)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_operator_error(new_root: Path, out_path: Path) -> None:
    rows: list[dict[str, str]] = []
    for convention in ("classical4d", "classical6d"):
        path = new_root / f"operator_eval_{convention}_dim256" / "seidel_physical_operator_metrics.csv"
        if path.is_file():
            rows.extend(read_csv(path))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color in [
        ("operator_error_calibrated", "calibrated", "#4C78A8"),
        ("operator_error_phys_equiv", "phys-equiv", "#F58518"),
        ("psf_error_calibrated", "PSF calibrated", "#54A24B"),
    ]:
        means = group_mean(rows, key)
        for conv, linestyle in [("classical4d", "-"), ("classical6d", "--")]:
            xs = sorted({rms for c, _d, rms in means if c == conv})
            ys = []
            for x in xs:
                vals = [means[(c, d, rms)] for c, d, rms in means if c == conv and rms == x]
                ys.append(float(np.mean(vals)))
            if xs:
                ax.plot(xs, ys, linestyle=linestyle, marker="o", color=color, label=f"{conv} {label}")
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_xlabel("target wavefront RMS (waves)")
    ax.set_ylabel("operator / PSF error")
    ax.set_title("Even-flip physical-operator evaluator")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_stats(rows: list[dict[str, str]], new_root: Path, out_root: Path) -> None:
    stats_dir = out_root / "stats"
    plot_delta_metric(
        rows,
        stats_dir / "delta_ssim_even_minus_gtfixed_by_rms.png",
        key="delta_ssim_even_minus_gtfixed",
        ylabel="mean delta SSIM (even - GT-fixed)",
        title="Object SSIM change from even-flip fixed Seidel",
    )
    plot_delta_metric(
        rows,
        stats_dir / "delta_nrmse_even_minus_gtfixed_by_rms.png",
        key="delta_nrmse_even_minus_gtfixed",
        ylabel="mean delta NRMSE (even - GT-fixed)",
        title="Object NRMSE change from even-flip fixed Seidel",
    )
    plot_old_new_metric(
        rows,
        stats_dir / "ssim_gtfixed_vs_evenflip_by_rms.png",
        old_key="gt_fixed_ssim",
        new_key="even_flip_ssim",
        ylabel="mean SSIM",
        title="GT-fixed and even-flip object SSIM",
    )
    plot_old_new_metric(
        rows,
        stats_dir / "nrmse_gtfixed_vs_evenflip_by_rms.png",
        old_key="gt_fixed_nrmse",
        new_key="even_flip_nrmse",
        ylabel="mean NRMSE",
        title="GT-fixed and even-flip object NRMSE",
    )
    plot_delta_metric(
        rows,
        stats_dir / "even_flip_relative_wavefront_error_by_rms.png",
        key="even_flip_relative_wavefront_error",
        ylabel="relative wavefront error",
        title="Coefficient-space wavefront residual of even-flip vector",
    )
    plot_delta_metric(
        rows,
        stats_dir / "even_flip_coeff_l2_by_rms.png",
        key="even_flip_l2_seidel_vs_gt",
        ylabel="coefficient L2 vs GT",
        title="Coefficient L2 of even-flip vector vs GT",
    )
    plot_operator_error(new_root, stats_dir / "operator_error_by_rms.png")


def build_contact_sheets(rows: list[dict[str, str]], rcp_paths: dict[tuple[str, str, str, int], Path], out_root: Path) -> None:
    contact_dir = out_root / "00_contact_sheets_by_dimension_rms"
    for convention in sorted({row["seidel_convention"] for row in rows}):
        for rms in sorted({float(row["target_wavefront_rms"]) for row in rows if row["seidel_convention"] == convention}):
            paths = []
            for row in rows:
                if row["seidel_convention"] != convention or abs(float(row["target_wavefront_rms"]) - rms) > 1e-9:
                    continue
                key = (row["seidel_convention"], row["image"], row["candidate_id"], int(row.get("seed") or 0))
                if key in rcp_paths:
                    paths.append(rcp_paths[key])
            paths.sort(key=lambda path: str(path))
            make_contact_sheet(
                paths,
                contact_dir / f"{convention}__rms{tag_float(rms)}__contact_sheet.png",
                title=f"{convention} RMS {rms:.2f}: GT-fixed vs even-flip fixed",
            )


def write_readme(out_root: Path, generated: list[Path], *, include_recovered: bool = False) -> None:
    lines = [
        "# Even-Flip vs GT-Fixed RCP Comparison",
        "",
        "Each per-case RCP places the previous `seidel_gt_fixed` comparison panel on the left and",
        "the new `even_flip_fixed` comparison panel on the right, with Seidel coefficient bars and",
        "operator/object metrics below.",
        "",
        "Key folders:",
        "- `classical4d/` and `classical6d/`: per-case RCP comparison PNGs.",
        "- `00_contact_sheets_by_dimension_rms/`: contact sheets grouped by convention and RMS.",
        "- `stats/`: summary statistics plots.",
        "",
        f"Generated per-case RCPs: {len(generated)}",
        "",
    ]
    if include_recovered:
        insert_at = lines.index("Key folders:")
        lines[insert_at:insert_at] = [
            "The coefficient bar chart additionally overlays the *recovered* Seidel vectors",
            "(`aligned_seidel_physical`) from the reference oracle run's",
            "`oracle_controls_evaluator_combined.csv`:",
            "- `joint_no_RMS` (blue): free joint recovery, nothing fixed.",
            "- `object_gt_fixed` (purple): object fixed to sharp GT, Seidel recovered.",
            "The metric card reports each recovered vector's coefficient L2 to GT and to the",
            "even-flip twin, with a `closer to` verdict per mode.",
            "",
        ]
    write_csv(
        [{"path": str(path.relative_to(out_root))} for path in generated],
        out_root / "manifest.csv",
    )
    (out_root / "README.md").write_text("\n".join(lines))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, default=OUTPUT_BASE / OLD_RUN)
    parser.add_argument("--new-root", type=Path, default=OUTPUT_BASE / NEW_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--include-recovered",
        action="store_true",
        help=(
            "Overlay recovered Seidel coefficients (joint_no_RMS and object_gt_fixed "
            "modes from the old run's combined evaluator CSV) on the coefficient bars; "
            "writes to RCP_even_flip_vs_gt_fixed_with_recovered by default."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    old_root = args.old_root
    new_root = args.new_root
    default_name = (
        "RCP_even_flip_vs_gt_fixed_with_recovered"
        if args.include_recovered
        else "RCP_even_flip_vs_gt_fixed"
    )
    out_root = args.output_dir or (new_root / default_name)
    comparison_csv = new_root / "comparison_vs_seidel_gt_fixed.csv"
    rows = read_csv(comparison_csv)
    operator_lookup = build_operator_lookup(new_root)
    recovered_lookup = build_recovered_lookup(old_root) if args.include_recovered else None
    generated: list[Path] = []
    rcp_paths: dict[tuple[str, str, str, int], Path] = {}
    for row in rows:
        out_path = build_case_rcp(
            row,
            old_root=old_root,
            new_root=new_root,
            out_root=out_root,
            operator_lookup=operator_lookup,
            recovered_lookup=recovered_lookup,
        )
        if out_path is None:
            print(f"[missing] {row['seidel_convention']} {row['image']} {row['candidate_id']}", file=sys.stderr)
            continue
        generated.append(out_path)
        key = (row["seidel_convention"], row["image"], row["candidate_id"], int(row.get("seed") or 0))
        rcp_paths[key] = out_path
    build_contact_sheets(rows, rcp_paths, out_root)
    build_stats(rows, new_root, out_root)
    write_readme(out_root, generated, include_recovered=args.include_recovered)
    print(f"[done] generated_rcps={len(generated)} output={out_root}")


if __name__ == "__main__":
    main()
