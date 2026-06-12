"""Build full-factorial pretrain-contrast manifests.

This creates a 420-setting grid (1260 cases) at a configurable target RMS and
splits the cases into fixed work manifests. The split scheme is configurable
via --splits (default: CAML GPU0/GPU1 + Fourier GPU1/GPU2 from the RMS0.20
run); the case order is deterministic so any split is a stable slice.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_PREFIX = "pretrain_contrast_fullgrid4d_size256_three_images_rms020_pre400_joint1000_20260609"
IMAGES = ["Iksung_beads", "dendrites", "dendrites_dense"]
DEFAULT_SPLITS = "caml_gpu0:250,caml_gpu1:250,fourier_gpu1:380,fourier_gpu2:380"


def parse_splits(text: str) -> list[tuple[str, int]]:
    splits: list[tuple[str, int]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        name, _, count_text = token.partition(":")
        name = name.strip()
        count = int(count_text)
        if not name or count <= 0:
            raise ValueError(f"Invalid split token {token!r}; expected name:count with count > 0")
        if any(existing == name for existing, _ in splits):
            raise ValueError(f"Duplicate split name {name!r}")
        splits.append((name, count))
    if not splits:
        raise ValueError(f"No splits parsed from {text!r}")
    return splits


def tag_float(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p").replace("+", "")


def tag_sci(value: float) -> str:
    if float(value) == 0.0:
        return "0"
    return f"{float(value):.0e}".replace("-", "m").replace("+", "")


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
    scalar: float,
    transform: str,
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
    scalars = [0.5, 1.0, 3.0, 7.5, 10.0]
    rsds = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    for scalar in scalars:
        for rsd in rsds:
            if scalar == 1.0 and rsd == 0.0:
                method = "baseline_scalar1"
                label = "baseline: scalar=1"
            else:
                method = f"fg_s{tag_float(scalar)}_none_rsd{tag_sci(rsd)}"
                label = f"scalar={scalar:g} + none + RSD={rsd:g}"
            settings.append(
                make_setting(
                    method,
                    "fullgrid_none",
                    label,
                    scalar=scalar,
                    transform="none",
                    rsd=rsd,
                )
            )

    for scalar in scalars:
        for alpha in [0.5, 2.0, 4.0, 8.0]:
            for rsd in rsds:
                settings.append(
                    make_setting(
                        f"fg_s{tag_float(scalar)}_alpha{tag_float(alpha)}_rsd{tag_sci(rsd)}",
                        "fullgrid_linear_contrast",
                        f"scalar={scalar:g} + alpha={alpha:g} + RSD={rsd:g}",
                        scalar=scalar,
                        transform="linear_contrast",
                        alpha=alpha,
                        rsd=rsd,
                    )
                )

    for scalar in scalars:
        for pct_lo, pct_hi in [(0.1, 99.9), (1.0, 99.0), (5.0, 95.0)]:
            for gamma in [0.3, 0.7, 1.5]:
                for rsd in rsds:
                    settings.append(
                        make_setting(
                            (
                                f"fg_s{tag_float(scalar)}_pg_p{tag_float(pct_lo)}_"
                                f"p{tag_float(pct_hi)}_g{tag_float(gamma)}_rsd{tag_sci(rsd)}"
                            ),
                            "fullgrid_percentile_gamma",
                            (
                                f"scalar={scalar:g} + p{pct_lo:g}/p{pct_hi:g} "
                                f"gamma={gamma:g} + RSD={rsd:g}"
                            ),
                            scalar=scalar,
                            transform="percentile_gamma",
                            pct_lo=pct_lo,
                            pct_hi=pct_hi,
                            gamma=gamma,
                            rsd=rsd,
                        )
                    )

    methods = [str(row["method"]) for row in settings]
    if len(settings) != 420 or len(set(methods)) != 420:
        raise RuntimeError(f"Expected 420 unique settings, got {len(settings)} / {len(set(methods))}")
    return settings


def build_cases(settings: list[dict[str, Any]], *, target_rms: float) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    candidate = f"signed_balanced__rms{str(target_rms).replace('.', 'p')}"
    for setting in settings:
        for image in IMAGES:
            cases.append(
                {
                    "global_case_index": len(cases),
                    "case_index": len(cases),
                    "image": image,
                    "direction": "signed_balanced",
                    "target_rms": float(target_rms),
                    "candidate_id": candidate,
                    "pretrain_method": setting["method"],
                    "method_label": setting["label"],
                    "family": setting["family"],
                }
            )
    if len(cases) != 1260:
        raise RuntimeError(f"Expected 1260 cases, got {len(cases)}")
    return cases


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


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


def write_case_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    local_rows = [dict(row, case_index=index) for index, row in enumerate(rows)]
    write_csv(local_rows, path.with_suffix(".csv"))
    write_json(local_rows, path.with_suffix(".json"))


def write_outputs(
    settings: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    out_dir: Path,
    splits: list[tuple[str, int]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(settings, out_dir / "settings_manifest.csv")
    write_json(settings, out_dir / "settings_manifest.json")
    write_csv(cases, out_dir / "case_manifest_full1260.csv")
    write_json(cases, out_dir / "case_manifest_full1260.json")

    cursor = 0
    for split_name, count in splits:
        rows = cases[cursor : cursor + count]
        cursor += count
        write_case_manifest(rows, out_dir / f"case_manifest_{split_name}")
    if cursor != len(cases):
        raise RuntimeError(f"Split consumed {cursor} cases, expected {len(cases)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--target-rms", type=float, default=0.20)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("outputs/cocoa_like_2d_mechanism") / f"{args.prefix}_logs"
    splits = parse_splits(args.splits)

    settings = build_settings()
    cases = build_cases(settings, target_rms=args.target_rms)
    write_outputs(settings, cases, out_dir, splits)
    print(
        f"[done] settings={len(settings)} cases={len(cases)} "
        f"splits={','.join(f'{name}:{count}' for name, count in splits)} out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
