"""Build scalar3+p1/p99 gamma0.7+RSD0.001 first-pretrain architecture manifests."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_secondjoint_s3_alpha2_rsd0p1_manifest as base  # noqa: E402


OLD_PREFIX = "s3_alpha2_rsd0p1"
NEW_PREFIX = "s3_pg_p1_p99_g0p7_rsd1em03"
OLD_LABEL = "first pretrain scalar3 alpha2 RSD0.1"
NEW_LABEL = "first pretrain scalar3 p1/p99 gamma0.7 RSD0.001"

base.DEFAULT_PREFIX = (
    "secondjoint_s3_pg_p1_p99_g0p7_rsd1em03_4d_size256_three_images_"
    "rms020_030_040_pre400_joint1000x2_20260613"
)
base.FIRST_PRETRAIN = {
    "pretrain_scalar": 3.0,
    "target_transform": "percentile_gamma",
    "pretrain_target_transform": "percentile_gamma",
    "contrast_alpha": 1.0,
    "pretrain_contrast_alpha": 1.0,
    "percentile_lo": 1.0,
    "percentile_hi": 99.0,
    "gamma": 0.7,
    "pretrain_rsd_weight": 0.001,
    "pretrain_edge_weight": 0.0,
    "pretrain_edge_mode": "sobel",
}


def remap_method(method: dict[str, object]) -> dict[str, object]:
    out = dict(method)
    out["method"] = str(out["method"]).replace(OLD_PREFIX, NEW_PREFIX)
    out["family"] = "firstpre_s3_pg_rsd"
    out["label"] = str(out["label"]).replace(OLD_LABEL, NEW_LABEL)
    if out.get("post_joint_pretrain_iter") == 0:
        out["post_joint_pretrain_scalar"] = 3.0
        out["post_joint_pretrain_target_transform"] = "percentile_gamma"
        out["post_joint_pretrain_contrast_alpha"] = 1.0
        out["post_joint_pretrain_percentile_lo"] = 1.0
        out["post_joint_pretrain_percentile_hi"] = 99.0
        out["post_joint_pretrain_gamma"] = 0.7
    out["post_joint_pretrain_rsd_weight"] = 0.0
    return out


base.METHODS = [remap_method(method) for method in base.METHODS]


if __name__ == "__main__":
    raise SystemExit(base.main())
