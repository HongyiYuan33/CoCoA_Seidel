"""Build the second-joint manifest using first-joint recon percentile as target."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs/cocoa_like_2d_mechanism"
DEFAULT_PREFIX = "secondjoint_postreconpct_4d_size256_three_images_rms020_030_040_pre400_joint1000x2_20260612"
IMAGES = ["Iksung_beads", "dendrites", "dendrites_dense"]
RMS_VALUES = [0.20, 0.30, 0.40]
METHODS = [
    {
        "method": "scalar5_second_joint_postreconpct_keepobj",
        "family": "secondjoint_postreconpct",
        "joint_variant": "second_joint_postreconpct_keepobj",
        "post_joint_object_init": "inherit",
        "label": "scalar=5 first joint + post-joint recon percentile, keep object",
    },
    {
        "method": "scalar5_second_joint_postreconpct_resetobj",
        "family": "secondjoint_postreconpct",
        "joint_variant": "second_joint_postreconpct_resetobj",
        "post_joint_object_init": "reset_fresh_same_seed",
        "label": "scalar=5 first joint + post-joint recon percentile, reset object",
    },
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


def common_setting(method: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method["method"],
        "pretrain_method": method["method"],
        "setting_id": method["method"],
        "family": method["family"],
        "joint_variant": method["joint_variant"],
        "label": method["label"],
        "method_label": method["label"],
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
        "second_joint_iter": 1000,
        "post_joint_pretrain_iter": 400,
        "post_joint_pretrain_source": "first_joint_object_raw",
        "post_joint_object_init": method["post_joint_object_init"],
        "post_joint_pretrain_scalar": 1.0,
        "post_joint_pretrain_target_transform": "percentile_gamma",
        "post_joint_pretrain_contrast_alpha": 1.0,
        "post_joint_pretrain_percentile_lo": 0.5,
        "post_joint_pretrain_percentile_hi": 99.5,
        "post_joint_pretrain_gamma": 1.0,
    }


def settings() -> list[dict[str, Any]]:
    return [common_setting(method) for method in METHODS]


def case_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_index = 0
    for method in METHODS:
        base = common_setting(method)
        for image in IMAGES:
            for rms in RMS_VALUES:
                candidate = f"signed_balanced__rms{rms_tag(rms)}"
                row = {
                    "case_index": case_index,
                    "global_case_index": case_index,
                    "image": image,
                    "direction": "signed_balanced",
                    "target_rms": f"{rms:.2f}",
                    "candidate_id": candidate,
                    "pretrain_method": method["method"],
                    "method": method["method"],
                    "method_label": method["label"],
                    "label": method["label"],
                }
                row.update({k: v for k, v in base.items() if k not in row})
                rows.append(row)
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
    setting_rows = settings()
    cases = case_rows()
    write_csv(setting_rows, args.out_dir / "settings_manifest.csv")
    write_csv(cases, args.out_dir / "case_manifest.csv")
    for method in METHODS:
        split_cases = [row for row in cases if row["pretrain_method"] == method["method"]]
        split_name = method["joint_variant"].replace("second_joint_postreconpct_", "")
        write_csv(split_cases, args.out_dir / f"case_manifest_{split_name}.csv")
        (args.out_dir / f"case_manifest_{split_name}.json").write_text(
            json.dumps(split_cases, indent=2) + "\n"
        )
    (args.out_dir / "settings_manifest.json").write_text(json.dumps(setting_rows, indent=2) + "\n")
    (args.out_dir / "case_manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    (args.out_dir / "run_prefix.txt").write_text(DEFAULT_PREFIX + "\n")
    print(f"[manifest] settings={len(setting_rows)} cases={len(cases)} out={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
