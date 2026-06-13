"""Build post-joint pretrain 1000-iter second-joint manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs/cocoa_like_2d_mechanism"
DEFAULT_PREFIX = "secondjoint_postpre1000_4d_size256_three_images_rms020_030_040_pre400_joint1000x2_20260613"
IMAGES = ["Iksung_beads", "dendrites", "dendrites_dense"]
RMS_VALUES = [0.20, 0.30, 0.40]
POST_PRETRAIN_ITER = 1000
SECOND_JOINT_ITER = 1000

METHODS = [
    {
        "method": "scalar5_second_joint_postobjraw_scalar5_postpre1000",
        "family": "secondjoint_postobjraw",
        "joint_variant": "postobjraw_scalar5_postpre1000",
        "label": "post-object-raw scalar5, post-pretrain 1000",
        "post_joint_pretrain_source": "first_joint_object_raw_clipped",
        "post_joint_pretrain_scalar": 5.0,
        "post_joint_pretrain_target_transform": "none",
        "post_joint_pretrain_contrast_alpha": 1.0,
        "post_joint_pretrain_percentile_lo": 1.0,
        "post_joint_pretrain_percentile_hi": 99.0,
        "post_joint_pretrain_gamma": 1.0,
        "post_joint_object_init": "inherit",
        "split": "gpu1",
    },
    {
        "method": "scalar5_second_joint_postobjraw_pg_p0p1_p99p9_g1p5_postpre1000",
        "family": "secondjoint_postobjraw",
        "joint_variant": "postobjraw_pg_postpre1000",
        "label": "post-object-raw p0.1/p99.9 gamma1.5, post-pretrain 1000",
        "post_joint_pretrain_source": "first_joint_object_raw_clipped",
        "post_joint_pretrain_scalar": 1.0,
        "post_joint_pretrain_target_transform": "percentile_gamma",
        "post_joint_pretrain_contrast_alpha": 1.0,
        "post_joint_pretrain_percentile_lo": 0.1,
        "post_joint_pretrain_percentile_hi": 99.9,
        "post_joint_pretrain_gamma": 1.5,
        "post_joint_object_init": "inherit",
        "split": "gpu1",
    },
    {
        "method": "scalar5_second_joint_postreconpct_keepobj_postpre1000",
        "family": "secondjoint_postreconpct",
        "joint_variant": "postreconpct_keepobj_postpre1000",
        "label": "post-recon-percentile keep object, post-pretrain 1000",
        "post_joint_pretrain_source": "first_joint_object_raw",
        "post_joint_pretrain_scalar": 1.0,
        "post_joint_pretrain_target_transform": "percentile_gamma",
        "post_joint_pretrain_contrast_alpha": 1.0,
        "post_joint_pretrain_percentile_lo": 0.5,
        "post_joint_pretrain_percentile_hi": 99.5,
        "post_joint_pretrain_gamma": 1.0,
        "post_joint_object_init": "inherit",
        "split": "gpu2",
    },
    {
        "method": "scalar5_second_joint_postreconpct_resetobj_postpre1000",
        "family": "secondjoint_postreconpct",
        "joint_variant": "postreconpct_resetobj_postpre1000",
        "label": "post-recon-percentile reset object, post-pretrain 1000",
        "post_joint_pretrain_source": "first_joint_object_raw",
        "post_joint_pretrain_scalar": 1.0,
        "post_joint_pretrain_target_transform": "percentile_gamma",
        "post_joint_pretrain_contrast_alpha": 1.0,
        "post_joint_pretrain_percentile_lo": 0.5,
        "post_joint_pretrain_percentile_hi": 99.5,
        "post_joint_pretrain_gamma": 1.0,
        "post_joint_object_init": "reset_fresh_same_seed",
        "split": "gpu2",
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
        "second_joint_iter": SECOND_JOINT_ITER,
        "post_joint_pretrain_iter": POST_PRETRAIN_ITER,
        "post_joint_pretrain_source": method["post_joint_pretrain_source"],
        "post_joint_object_init": method["post_joint_object_init"],
        "post_joint_pretrain_scalar": method["post_joint_pretrain_scalar"],
        "post_joint_pretrain_target_transform": method["post_joint_pretrain_target_transform"],
        "post_joint_pretrain_contrast_alpha": method["post_joint_pretrain_contrast_alpha"],
        "post_joint_pretrain_percentile_lo": method["post_joint_pretrain_percentile_lo"],
        "post_joint_pretrain_percentile_hi": method["post_joint_pretrain_percentile_hi"],
        "post_joint_pretrain_gamma": method["post_joint_pretrain_gamma"],
        "postpretrain_iter_tag": "postpre1000",
        "split": method["split"],
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
                row.update({key: value for key, value in base.items() if key not in row})
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
    for split in ["gpu1", "gpu2"]:
        split_cases = [row for row in cases if row["split"] == split]
        write_csv(split_cases, args.out_dir / f"case_manifest_{split}.csv")
        (args.out_dir / f"case_manifest_{split}.json").write_text(
            json.dumps(split_cases, indent=2) + "\n"
        )
    (args.out_dir / "settings_manifest.json").write_text(json.dumps(setting_rows, indent=2) + "\n")
    (args.out_dir / "case_manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    (args.out_dir / "run_prefix.txt").write_text(DEFAULT_PREFIX + "\n")
    print(
        f"[manifest] settings={len(setting_rows)} cases={len(cases)} "
        f"gpu1={sum(1 for row in cases if row['split'] == 'gpu1')} "
        f"gpu2={sum(1 for row in cases if row['split'] == 'gpu2')} "
        f"out={args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
