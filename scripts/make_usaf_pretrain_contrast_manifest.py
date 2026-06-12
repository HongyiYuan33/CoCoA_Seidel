"""Create the USAF-1951 pretrain-contrast case manifest.

The manifest is intentionally tiny: one image, one full-grid setting, and
three target RMS values.  It is used by the generic pretrain-contrast runner.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_PREFIX = "usaf1951_fg_s3_alpha2_rsd1em01_4d_size256_rms020_030_040_pre400_joint1000_20260612"


SETTINGS: dict[str, dict[str, Any]] = {
    "fg_s3_alpha2_rsd1em01": {
        "method": "fg_s3_alpha2_rsd1em01",
        "label": "scalar=3 + alpha=2 + RSD=0.1",
        "family": "fullgrid_linear_contrast",
        "pretrain_scalar": 3.0,
        "target_transform": "linear_contrast",
        "contrast_alpha": 2.0,
        "pretrain_rsd_weight": 0.1,
        "pretrain_edge_weight": 0.0,
        "pretrain_edge_mode": "sobel",
        "percentile_lo": 1.0,
        "percentile_hi": 99.0,
        "gamma": 1.0,
    },
    "scalar5": {
        "method": "scalar5",
        "label": "scalar=5",
        "family": "scalar",
        "pretrain_scalar": 5.0,
        "target_transform": "none",
        "contrast_alpha": 1.0,
        "pretrain_rsd_weight": 0.0,
        "pretrain_edge_weight": 0.0,
        "pretrain_edge_mode": "sobel",
        "percentile_lo": 1.0,
        "percentile_hi": 99.0,
        "gamma": 1.0,
    },
}


def rms_tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--image", default="USAF_1951", help="Single image key, kept for compatibility.")
    parser.add_argument("--images", nargs="+", default=None, help="One or more image keys.")
    parser.add_argument("--direction", default="signed_balanced")
    parser.add_argument("--rms", nargs="+", type=float, default=[0.2, 0.3, 0.4])
    parser.add_argument(
        "--measurement-direct",
        action="store_true",
        help="Write one direct-measurement case per image instead of RMS candidates.",
    )
    parser.add_argument(
        "--setting",
        choices=sorted(SETTINGS),
        default="fg_s3_alpha2_rsd1em01",
        help="Pretrain setting to write into the manifest.",
    )
    args = parser.parse_args()
    setting = SETTINGS[args.setting]

    images = args.images if args.images is not None else [args.image]
    rows: list[dict[str, Any]] = []
    for image in images:
        rms_values = [0.0] if args.measurement_direct else args.rms
        for target_rms in rms_values:
            idx = len(rows)
            direction = "measurement_direct" if args.measurement_direct else args.direction
            candidate_id = "measurement_direct" if args.measurement_direct else f"{args.direction}__rms{rms_tag(target_rms)}"
            rows.append(
                {
                    "case_index": idx,
                    "global_case_index": idx,
                    "image": image,
                    "direction": direction,
                    "candidate_id": candidate_id,
                    "target_rms": f"{target_rms:.12g}",
                    "pretrain_method": setting["method"],
                    "method_label": setting["label"],
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "case_manifest.csv")
    (args.output_dir / "case_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "settings_manifest.json").write_text(json.dumps([setting], indent=2) + "\n")
    write_csv([setting], args.output_dir / "settings_manifest.csv")
    (args.output_dir / "run_prefix.txt").write_text(args.prefix + "\n")
    print(f"[manifest] prefix={args.prefix} cases={len(rows)} out={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
