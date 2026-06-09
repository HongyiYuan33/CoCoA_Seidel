"""Create the 50-setting manifest for the broad pretrain-contrast sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_PREFIX = "pretrain_contrast_sweep4d_size256_three_images_rms020_pre400_joint1000_20260609"
DEFAULT_OUTPUT = (
    Path("outputs/cocoa_like_2d_mechanism")
    / f"{DEFAULT_PREFIX}_settings_manifest.json"
)


def tag_float(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p").replace("+", "")


def tag_sci(value: float) -> str:
    text = f"{float(value):.0e}".replace("-", "m").replace("+", "")
    return text.replace("e", "e")


def canonical(setting: dict[str, Any]) -> tuple[Any, ...]:
    scalar = float(setting.get("pretrain_scalar", 1.0))
    transform = str(setting.get("target_transform", "none"))
    alpha = float(setting.get("contrast_alpha", 1.0))
    rsd = float(setting.get("pretrain_rsd_weight", 0.0))
    pct_lo = float(setting.get("percentile_lo", 1.0))
    pct_hi = float(setting.get("percentile_hi", 99.0))
    gamma = float(setting.get("gamma", 1.0))
    if transform == "linear_contrast" and math.isclose(alpha, 1.0):
        transform = "none"
    if transform == "none":
        alpha = 1.0
        pct_lo = 1.0
        pct_hi = 99.0
        gamma = 1.0
    elif transform == "linear_contrast":
        pct_lo = 1.0
        pct_hi = 99.0
        gamma = 1.0
    return (
        round(scalar, 12),
        transform,
        round(alpha, 12),
        round(pct_lo, 12),
        round(pct_hi, 12),
        round(gamma, 12),
        round(rsd, 14),
    )


def base_setting(method: str, family: str, label: str) -> dict[str, Any]:
    return {
        "method": method,
        "family": family,
        "label": label,
        "pretrain_scalar": 1.0,
        "target_transform": "none",
        "contrast_alpha": 1.0,
        "pretrain_rsd_weight": 0.0,
        "pretrain_edge_weight": 0.0,
        "pretrain_edge_mode": "sobel",
        "percentile_lo": 1.0,
        "percentile_hi": 99.0,
        "gamma": 1.0,
    }


def make_setting(
    method: str,
    family: str,
    label: str,
    *,
    scalar: float = 1.0,
    transform: str = "none",
    alpha: float = 1.0,
    rsd: float = 0.0,
    pct_lo: float = 1.0,
    pct_hi: float = 99.0,
    gamma: float = 1.0,
) -> dict[str, Any]:
    row = base_setting(method, family, label)
    row.update(
        {
            "pretrain_scalar": float(scalar),
            "target_transform": transform,
            "contrast_alpha": float(alpha),
            "pretrain_rsd_weight": float(rsd),
            "percentile_lo": float(pct_lo),
            "percentile_hi": float(pct_hi),
            "gamma": float(gamma),
        }
    )
    return row


def build_settings() -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    omitted = {
        "combo_scalar0p5_alpha0p5",
        "combo_scalar0p5_rsd1em04",
        "combo_alpha0p5_rsd1em04",
    }

    def add(row: dict[str, Any]) -> None:
        if row["method"] in omitted:
            return
        key = canonical(row)
        if key in seen:
            return
        seen.add(key)
        settings.append(row)

    add(make_setting("baseline_scalar1", "baseline", "baseline: scalar=1"))

    for scalar in [0.5, 1.0, 3.0, 7.5, 10.0]:
        add(
            make_setting(
                f"scalar{tag_float(scalar)}",
                "scalar",
                f"scalar={scalar:g}",
                scalar=scalar,
            )
        )

    for rsd in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        add(
            make_setting(
                f"rsd{tag_sci(rsd)}",
                "rsd",
                f"RSD={rsd:g}",
                rsd=rsd,
            )
        )

    for alpha in [0.5, 1.0, 2.0, 4.0, 8.0]:
        add(
            make_setting(
                f"alpha{tag_float(alpha)}",
                "alpha",
                f"contrast alpha={alpha:g}",
                transform="linear_contrast",
                alpha=alpha,
            )
        )

    for pct_lo, pct_hi in [(0.1, 99.9), (1.0, 99.0), (5.0, 95.0)]:
        for gamma in [0.3, 0.7, 1.5]:
            add(
                make_setting(
                    f"pg_p{tag_float(pct_lo)}_p{tag_float(pct_hi)}_g{tag_float(gamma)}",
                    "percentile_gamma",
                    f"p{pct_lo:g}/p{pct_hi:g} gamma={gamma:g}",
                    transform="percentile_gamma",
                    pct_lo=pct_lo,
                    pct_hi=pct_hi,
                    gamma=gamma,
                )
            )

    for scalar in [0.5, 3.0, 7.5]:
        for alpha in [0.5, 2.0, 4.0]:
            add(
                make_setting(
                    f"combo_scalar{tag_float(scalar)}_alpha{tag_float(alpha)}",
                    "scalar_alpha",
                    f"scalar={scalar:g} + alpha={alpha:g}",
                    scalar=scalar,
                    transform="linear_contrast",
                    alpha=alpha,
                )
            )

    for scalar in [0.5, 3.0, 7.5]:
        for rsd in [1e-4, 1e-3, 1e-2]:
            add(
                make_setting(
                    f"combo_scalar{tag_float(scalar)}_rsd{tag_sci(rsd)}",
                    "scalar_rsd",
                    f"scalar={scalar:g} + RSD={rsd:g}",
                    scalar=scalar,
                    rsd=rsd,
                )
            )

    for alpha in [0.5, 2.0, 4.0]:
        for rsd in [1e-4, 1e-3, 1e-2]:
            add(
                make_setting(
                    f"combo_alpha{tag_float(alpha)}_rsd{tag_sci(rsd)}",
                    "alpha_rsd",
                    f"alpha={alpha:g} + RSD={rsd:g}",
                    transform="linear_contrast",
                    alpha=alpha,
                    rsd=rsd,
                )
            )

    for pct_lo, pct_hi, gamma, rsd in [
        (0.1, 99.9, 0.3, 1e-3),
        (1.0, 99.0, 0.7, 1e-3),
        (5.0, 95.0, 1.5, 1e-3),
    ]:
        add(
            make_setting(
                f"combo_pg_p{tag_float(pct_lo)}_p{tag_float(pct_hi)}_g{tag_float(gamma)}_rsd{tag_sci(rsd)}",
                "percentile_gamma_rsd",
                f"p{pct_lo:g}/p{pct_hi:g} gamma={gamma:g} + RSD={rsd:g}",
                transform="percentile_gamma",
                pct_lo=pct_lo,
                pct_hi=pct_hi,
                gamma=gamma,
                rsd=rsd,
            )
        )

    if len(settings) != 50:
        raise RuntimeError(f"Expected exactly 50 unique settings, got {len(settings)}")
    return settings


def write_csv(settings: list[dict[str, Any]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in settings:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(settings)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()
    settings = build_settings()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(settings, indent=2) + "\n")
    if args.csv_output is not None:
        write_csv(settings, args.csv_output)
    print(f"[done] wrote {len(settings)} settings to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
