"""Build RMS0.40 manifests for RMS0.20-derived pretrain contrast cases.

The manifest contains the original 150 RMS0.20 (image, method) cases mapped to
RMS0.40, plus a separate manifest for cases not already completed at RMS0.40.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs/cocoa_like_2d_mechanism"
RMS020_PREFIX = "pretrain_contrast_sweep4d_size256_three_images_rms020_pre400_joint1000_20260609"
RMS040_TOP_PREFIX = "pretrain_contrast_top10plusbase4d_size256_three_images_rms040_pre400_joint1000_20260609"
DEFAULT_OUT_DIR = OUTPUT_ROOT / "pretrain_contrast_rms040_complete_from_rms020_20260609_logs"
IMAGES = ["Iksung_beads", "dendrites", "dendrites_dense"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def method_id(row: dict[str, Any]) -> str:
    return str(row.get("method") or row.get("pretrain_method") or "")


def read_settings(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    out = []
    for row in rows:
        item = dict(row)
        item["method"] = method_id(item)
        out.append(item)
    return out


def key(row: dict[str, str]) -> tuple[str, str]:
    return row["image"], row["pretrain_method"]


def case_row_from_source(
    row: dict[str, str],
    *,
    case_index: int,
    target_rms: float,
) -> dict[str, Any]:
    rms_token = str(target_rms).replace(".", "p")
    return {
        "case_index": case_index,
        "source_image": row["image"],
        "source_candidate_id": row.get("candidate_id", ""),
        "source_target_rms": row.get("target_rms", ""),
        "image": row["image"],
        "direction": "signed_balanced",
        "target_rms": float(target_rms),
        "candidate_id": f"signed_balanced__rms{rms_token}",
        "pretrain_method": row["pretrain_method"],
        "method_label": row.get("method_label", ""),
        "source_operator_error_calibrated": row.get("operator_error_calibrated", ""),
        "source_ssim": row.get("ssim", ""),
        "source_nrmse": row.get("nrmse", ""),
        "source_aligned_coeff_absolute_error_physical": row.get(
            "aligned_coeff_absolute_error_physical",
            "",
        ),
    }


def validate(
    *,
    full_rows: list[dict[str, Any]],
    remaining_rows: list[dict[str, Any]],
    settings: list[dict[str, Any]],
    target_rms: float,
    expected_full: int,
    expected_remaining: int,
) -> None:
    if len(full_rows) != expected_full:
        raise RuntimeError(f"Expected {expected_full} full rows, got {len(full_rows)}")
    if len(remaining_rows) != expected_remaining:
        raise RuntimeError(
            f"Expected {expected_remaining} remaining rows, got {len(remaining_rows)}"
        )
    if len(settings) != 50:
        raise RuntimeError(f"Expected 50 settings, got {len(settings)}")
    methods = {method_id(row) for row in settings}
    if "scalar5" in methods:
        raise RuntimeError("settings_manifest must not contain scalar5")
    for rows, expected_per_image in [(full_rows, 50), (remaining_rows, 39)]:
        counts = Counter(str(row["image"]) for row in rows)
        for image in IMAGES:
            if counts[image] != expected_per_image:
                raise RuntimeError(
                    f"{image}: expected {expected_per_image}, got {counts[image]}"
                )
    for row in full_rows + remaining_rows:
        if float(row["target_rms"]) != float(target_rms):
            raise RuntimeError(f"Wrong RMS in row: {row}")
        if row["candidate_id"] != "signed_balanced__rms0p4":
            raise RuntimeError(f"Wrong candidate_id in row: {row}")
        if row["pretrain_method"] not in methods:
            raise RuntimeError(f"Unknown method in row: {row['pretrain_method']}")


def reindex(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row, case_index=index) for index, row in enumerate(rows)]


def split_for_fourier(
    remaining_rows: list[dict[str, Any]],
    *,
    fourier_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if fourier_count < 0 or fourier_count > len(remaining_rows):
        raise ValueError("--fourier-count must be within remaining row count")
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in remaining_rows:
        by_image[str(row["image"])].append(row)
    base_take, extra = divmod(fourier_count, len(IMAGES))
    fourier_keys: set[tuple[str, str]] = set()
    for image_index, image in enumerate(IMAGES):
        take = base_take + (1 if image_index < extra else 0)
        rows = by_image[image]
        if take > len(rows):
            raise RuntimeError(f"Cannot take {take} rows for {image}; only {len(rows)} available")
        for row in rows[:take]:
            fourier_keys.add((str(row["image"]), str(row["pretrain_method"])))
    fourier_rows = [
        row
        for row in remaining_rows
        if (str(row["image"]), str(row["pretrain_method"])) in fourier_keys
    ]
    caml_rows = [
        row
        for row in remaining_rows
        if (str(row["image"]), str(row["pretrain_method"])) not in fourier_keys
    ]
    if len(fourier_rows) != fourier_count:
        raise RuntimeError(f"Expected {fourier_count} Fourier rows, got {len(fourier_rows)}")
    return reindex(fourier_rows), reindex(caml_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rms020-comparison",
        type=Path,
        default=OUTPUT_ROOT / f"{RMS020_PREFIX}_rcp_stats" / "stats" / "comparison_by_case.csv",
    )
    parser.add_argument(
        "--rms040-comparison",
        type=Path,
        default=OUTPUT_ROOT / f"{RMS040_TOP_PREFIX}_rcp_stats" / "stats" / "comparison_by_case.csv",
    )
    parser.add_argument(
        "--settings-json",
        type=Path,
        default=OUTPUT_ROOT / f"{RMS020_PREFIX}_rcp_stats" / "settings_manifest.json",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-rms", type=float, default=0.40)
    parser.add_argument("--expected-full", type=int, default=150)
    parser.add_argument("--expected-remaining", type=int, default=117)
    parser.add_argument("--fourier-count", type=int, default=60)
    args = parser.parse_args()

    rms020_rows = read_csv(args.rms020_comparison)
    rms040_rows = read_csv(args.rms040_comparison)
    settings = read_settings(args.settings_json)

    completed_rms040 = {
        key(row)
        for row in rms040_rows
        if row.get("pretrain_method") and row.get("pretrain_method") != "scalar5"
    }

    full_rows = [
        case_row_from_source(row, case_index=index, target_rms=args.target_rms)
        for index, row in enumerate(rms020_rows)
    ]
    remaining_rows = [
        dict(row, case_index=index)
        for index, row in enumerate(full_rows)
        if (row["image"], row["pretrain_method"]) not in completed_rms040
    ]

    validate(
        full_rows=full_rows,
        remaining_rows=remaining_rows,
        settings=settings,
        target_rms=args.target_rms,
        expected_full=args.expected_full,
        expected_remaining=args.expected_remaining,
    )
    fourier_rows, caml_rows = split_for_fourier(
        remaining_rows,
        fourier_count=args.fourier_count,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(remaining_rows, args.out_dir / "remaining_case_manifest.csv")
    (args.out_dir / "remaining_case_manifest.json").write_text(
        json.dumps(remaining_rows, indent=2) + "\n"
    )
    write_csv(full_rows, args.out_dir / "full150_case_manifest.csv")
    (args.out_dir / "full150_case_manifest.json").write_text(
        json.dumps(full_rows, indent=2) + "\n"
    )
    write_csv(settings, args.out_dir / "settings_manifest.csv")
    (args.out_dir / "settings_manifest.json").write_text(json.dumps(settings, indent=2) + "\n")
    write_csv(fourier_rows, args.out_dir / "fourier_case_manifest.csv")
    (args.out_dir / "fourier_case_manifest.json").write_text(
        json.dumps(fourier_rows, indent=2) + "\n"
    )
    write_csv(caml_rows, args.out_dir / "caml_case_manifest.csv")
    (args.out_dir / "caml_case_manifest.json").write_text(json.dumps(caml_rows, indent=2) + "\n")

    print(
        f"[done] full={len(full_rows)} remaining={len(remaining_rows)} "
        f"fourier={len(fourier_rows)} caml={len(caml_rows)} "
        f"settings={len(settings)} out={args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
