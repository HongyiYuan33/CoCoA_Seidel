"""Build the scalar5 single-vs-second-joint manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs/cocoa_like_2d_mechanism"
DEFAULT_PREFIX = "secondjoint_scalar5_4d_size256_three_images_rms020_030_040_pre400_joint1000x2_20260612"
IMAGES = ["Iksung_beads", "dendrites", "dendrites_dense"]
RMS_VALUES = [0.20, 0.30, 0.40]
VARIANTS = [
    ("scalar5_single_joint", "single_joint", 0, "scalar=5 single joint"),
    ("scalar5_second_joint", "second_joint", 1000, "scalar=5 second joint reset-Seidel"),
]


def rms_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def scalar5_settings() -> list[dict[str, Any]]:
    rows = []
    for method, variant, second_iter, label in VARIANTS:
        rows.append(
            {
                "method": method,
                "pretrain_method": method,
                "setting_id": method,
                "family": "scalar5_secondjoint",
                "joint_variant": variant,
                "label": label,
                "method_label": label,
                "pretrain_scalar": 5.0,
                "target_transform": "none",
                "pretrain_target_transform": "none",
                "contrast_alpha": 1.0,
                "pretrain_contrast_alpha": 1.0,
                "percentile_lo": 1.0,
                "percentile_hi": 99.0,
                "gamma": 1.0,
                "pretrain_rsd_weight": 0.0,
                "pretrain_edge_weight": 0.0,
                "pretrain_edge_mode": "sobel",
                "second_joint_iter": second_iter,
            }
        )
    return rows


def case_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_index = 0
    for image in IMAGES:
        for rms in RMS_VALUES:
            candidate = f"signed_balanced__rms{rms_tag(rms)}"
            for method, variant, second_iter, label in VARIANTS:
                rows.append(
                    {
                        "case_index": case_index,
                        "global_case_index": case_index,
                        "image": image,
                        "direction": "signed_balanced",
                        "target_rms": f"{rms:.2f}",
                        "candidate_id": candidate,
                        "pretrain_method": method,
                        "method": method,
                        "method_label": label,
                        "label": label,
                        "family": "scalar5_secondjoint",
                        "joint_variant": variant,
                        "second_joint_iter": second_iter,
                    }
                )
                case_index += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_ROOT / f"{DEFAULT_PREFIX}_logs",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings = scalar5_settings()
    cases = case_rows()
    write_csv(settings, args.out_dir / "settings_manifest.csv")
    write_csv(cases, args.out_dir / "case_manifest.csv")
    (args.out_dir / "settings_manifest.json").write_text(json.dumps(settings, indent=2) + "\n")
    (args.out_dir / "case_manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    (args.out_dir / "run_prefix.txt").write_text(DEFAULT_PREFIX + "\n")
    print(f"[manifest] settings={len(settings)} cases={len(cases)} out={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
