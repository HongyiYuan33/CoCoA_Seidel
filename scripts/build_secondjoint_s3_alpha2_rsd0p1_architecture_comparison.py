"""Build comparison views for scalar3+alpha2+RSD0.1 first-pretrain variants."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_secondjoint_postpre1000_architecture_comparison as base  # noqa: E402


base.DEFAULT_OUT = (
    base.OUTPUT_ROOT / "secondjoint_s3_alpha2_rsd0p1_architecture_comparison_20260613"
)
base.DEFAULT_RCP_DIRS = [
    base.OUTPUT_ROOT
    / "secondjoint_s3_alpha2_rsd0p1_4d_size256_three_images_rms020_030_040_pre400_joint1000x2_20260613_rcp_stats"
]
base.METHOD_TO_VARIANT = {
    "s3_alpha2_rsd0p1_single_joint": "single_joint",
    "s3_alpha2_rsd0p1_second_joint": "second_joint",
    "s3_alpha2_rsd0p1_postobjraw_scalar5_postpre400": "postobjraw_scalar5_postpre400",
    "s3_alpha2_rsd0p1_postobjraw_scalar5_postpre1000": "postobjraw_scalar5_postpre1000",
    "s3_alpha2_rsd0p1_postobjraw_pg_postpre400": "postobjraw_pg_postpre400",
    "s3_alpha2_rsd0p1_postobjraw_pg_postpre1000": "postobjraw_pg_postpre1000",
    "s3_alpha2_rsd0p1_postreconpct_keepobj_postpre400": "postreconpct_keepobj_postpre400",
    "s3_alpha2_rsd0p1_postreconpct_keepobj_postpre1000": "postreconpct_keepobj_postpre1000",
    "s3_alpha2_rsd0p1_postreconpct_resetobj_postpre400": "postreconpct_resetobj_postpre400",
    "s3_alpha2_rsd0p1_postreconpct_resetobj_postpre1000": "postreconpct_resetobj_postpre1000",
}
base.VARIANT_LABELS = {
    "single_joint": "baseline: s3 alpha2 RSD0.1 single joint",
    "second_joint": "s3 alpha2 RSD0.1 second joint reset Seidel",
    "postobjraw_scalar5_postpre400": "s3 alpha2 RSD0.1 + post object-raw scalar5, post-pretrain 400",
    "postobjraw_scalar5_postpre1000": "s3 alpha2 RSD0.1 + post object-raw scalar5, post-pretrain 1000",
    "postobjraw_pg_postpre400": "s3 alpha2 RSD0.1 + post object-raw p/g, post-pretrain 400",
    "postobjraw_pg_postpre1000": "s3 alpha2 RSD0.1 + post object-raw p/g, post-pretrain 1000",
    "postreconpct_keepobj_postpre400": "s3 alpha2 RSD0.1 + post recon percentile keep, post-pretrain 400",
    "postreconpct_keepobj_postpre1000": "s3 alpha2 RSD0.1 + post recon percentile keep, post-pretrain 1000",
    "postreconpct_resetobj_postpre400": "s3 alpha2 RSD0.1 + post recon percentile reset, post-pretrain 400",
    "postreconpct_resetobj_postpre1000": "s3 alpha2 RSD0.1 + post recon percentile reset, post-pretrain 1000",
}


if __name__ == "__main__":
    raise SystemExit(base.main())
