"""Compare Adam-vs-SGD Seidel optimizer sweeps.

The script expects two physical-operator evaluator CSVs with matching
``image``, ``direction``, and ``target_wavefront_rms`` keys.  It writes
case-level deltas, aggregate summaries, a short Markdown report, and simple
delta plots.  Lower is better for operator, NRMSE, aligned wavefront, and
aligned coefficient error; higher is better for SSIM.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRICS = [
    ("operator_error_calibrated", False),
    ("aligned_wavefront_error_physical", False),
    ("aligned_coeff_relative_error_physical", False),
    ("ssim_recon_gain_vs_gt", True),
    ("nrmse_recon_gain_vs_gt", False),
]


def parse_float(value: Any, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_csv(path: Path) -> list[dict[str, str]]:
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


def case_key(row: dict[str, Any]) -> tuple[str, str, float]:
    return (
        str(row["image"]),
        str(row["direction"]),
        round(parse_float(row["target_wavefront_rms"]), 6),
    )


def keyed_rows(rows: list[dict[str, str]], label: str) -> dict[tuple[str, str, float], dict[str, str]]:
    out: dict[tuple[str, str, float], dict[str, str]] = {}
    for row in rows:
        key = case_key(row)
        if key in out:
            raise ValueError(f"Duplicate {label} case key: {key}")
        out[key] = row
    return out


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(sum(clean) / len(clean)) if clean else math.nan


def median(values: list[float]) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return math.nan
    mid = len(clean) // 2
    if len(clean) % 2:
        return float(clean[mid])
    return float((clean[mid - 1] + clean[mid]) / 2.0)


def format_float(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def compare_rows(adam_rows: list[dict[str, str]], sgd_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    adam_by_key = keyed_rows(adam_rows, "adam")
    sgd_by_key = keyed_rows(sgd_rows, "sgd")
    common = sorted(
        set(adam_by_key) & set(sgd_by_key),
        key=lambda key: (key[0], key[1], key[2]),
    )
    missing_adam = sorted(set(sgd_by_key) - set(adam_by_key))
    missing_sgd = sorted(set(adam_by_key) - set(sgd_by_key))
    if missing_adam or missing_sgd:
        raise ValueError(
            "CSV case keys do not match: "
            f"missing_adam={len(missing_adam)} missing_sgd={len(missing_sgd)}"
        )

    rows: list[dict[str, Any]] = []
    for key in common:
        adam = adam_by_key[key]
        sgd = sgd_by_key[key]
        out: dict[str, Any] = {
            "image": key[0],
            "direction": key[1],
            "target_wavefront_rms": f"{key[2]:.6g}",
            "adam_candidate_id": adam.get("candidate_id", ""),
            "sgd_candidate_id": sgd.get("candidate_id", ""),
        }
        for metric, higher_is_better in METRICS:
            adam_value = parse_float(adam.get(metric))
            sgd_value = parse_float(sgd.get(metric))
            delta = sgd_value - adam_value
            improved = delta > 0.0 if higher_is_better else delta < 0.0
            out[f"adam_{metric}"] = adam_value
            out[f"sgd_{metric}"] = sgd_value
            out[f"delta_{metric}"] = delta
            out[f"sgd_better_{metric}"] = bool(improved) if math.isfinite(delta) else ""
        rows.append(out)
    return rows


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        summary: dict[str, Any] = {key: value for key, value in zip(keys, key_values)}
        summary["cases"] = len(group)
        for metric, higher_is_better in METRICS:
            deltas = [parse_float(row.get(f"delta_{metric}")) for row in group]
            better = [row.get(f"sgd_better_{metric}") is True for row in group]
            summary[f"mean_delta_{metric}"] = mean(deltas)
            summary[f"median_delta_{metric}"] = median(deltas)
            summary[f"sgd_better_count_{metric}"] = sum(better)
            summary[f"sgd_better_rate_{metric}"] = sum(better) / len(group) if group else math.nan
            summary[f"direction_for_better_{metric}"] = "higher" if higher_is_better else "lower"
        out.append(summary)
    return out


def plot_metric(rows: list[dict[str, Any]], metric: str, higher_is_better: bool, output_dir: Path) -> None:
    labels = [
        f"{row['image']}\n{row['direction']}\nrms={row['target_wavefront_rms']}"
        for row in rows
    ]
    deltas = [parse_float(row.get(f"delta_{metric}")) for row in rows]
    colors = [
        "#2E7D32" if ((value > 0.0) if higher_is_better else (value < 0.0)) else "#B3261E"
        for value in deltas
    ]
    fig_width = max(10.0, 0.55 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_width, 5.4))
    ax.bar(range(len(rows)), deltas, color=colors, width=0.82)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_ylabel(f"SGD - Adam delta ({metric})")
    ax.set_title(f"{metric} delta by case")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    note = "green means SGD better"
    ax.text(0.995, 0.98, note, ha="right", va="top", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / f"delta_{metric}.png", dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict[str, Any]], summary_by_direction_rms: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Seidel optimizer comparison",
        "",
        f"Matched cases: {len(rows)}",
        "",
        "Lower is better for operator, NRMSE, aligned wavefront, and aligned coefficient error. Higher is better for SSIM.",
        "",
        "## Overall deltas",
        "",
    ]
    for metric, higher_is_better in METRICS:
        deltas = [parse_float(row.get(f"delta_{metric}")) for row in rows]
        better = sum(row.get(f"sgd_better_{metric}") is True for row in rows)
        direction = "higher" if higher_is_better else "lower"
        lines.append(
            f"- `{metric}` ({direction} is better): mean SGD-Adam delta "
            f"{format_float(mean(deltas))}; SGD better in {better}/{len(rows)} cases."
        )

    lines += ["", "## Direction / RMS summary", ""]
    for row in summary_by_direction_rms:
        lines.append(
            f"- {row['direction']} rms={row['target_wavefront_rms']}: "
            f"operator mean delta {format_float(parse_float(row['mean_delta_operator_error_calibrated']))}, "
            f"SSIM mean delta {format_float(parse_float(row['mean_delta_ssim_recon_gain_vs_gt']))}, "
            f"NRMSE mean delta {format_float(parse_float(row['mean_delta_nrmse_recon_gain_vs_gt']))}."
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adam-csv", type=Path, required=True)
    parser.add_argument("--sgd-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = compare_rows(read_csv(args.adam_csv), read_csv(args.sgd_csv))
    write_csv(rows, args.output_dir / "comparison_by_case.csv")

    by_direction_rms = summarize(rows, ["direction", "target_wavefront_rms"])
    by_image = summarize(rows, ["image"])
    write_csv(by_direction_rms, args.output_dir / "summary_by_direction_rms.csv")
    write_csv(by_image, args.output_dir / "summary_by_image.csv")
    write_summary(rows, by_direction_rms, args.output_dir / "summary.md")

    for metric, higher_is_better in METRICS:
        plot_metric(rows, metric, higher_is_better, args.output_dir)

    print(f"[done] wrote comparison outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
