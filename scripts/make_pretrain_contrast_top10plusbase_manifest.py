"""Build the RMS0.40 top10+baseline+scalar5 pretrain-contrast manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "outputs/cocoa_like_2d_mechanism"
    / "pretrain_contrast_sweep4d_size256_three_images_rms020_pre400_joint1000_20260609_rcp_stats"
)
DEFAULT_COMPARISON_CSV = DEFAULT_SOURCE_DIR / "stats" / "comparison_by_case.csv"
DEFAULT_SETTINGS_JSON = DEFAULT_SOURCE_DIR / "settings_manifest.json"
DEFAULT_PREFIX = "pretrain_contrast_top10plusbase4d_size256_three_images_rms040_pre400_joint1000_20260609"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "outputs/cocoa_like_2d_mechanism"
    / f"{DEFAULT_PREFIX}_logs"
)
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
        for row in rows:
            writer.writerow(row)


def method_id(row: dict[str, Any]) -> str:
    return str(row.get("method") or row.get("pretrain_method") or "")


def scalar5_setting() -> dict[str, Any]:
    return {
        "method": "scalar5",
        "family": "scalar",
        "label": "scalar=5",
        "pretrain_scalar": 5.0,
        "target_transform": "none",
        "contrast_alpha": 1.0,
        "pretrain_rsd_weight": 0.0,
        "pretrain_edge_weight": 0.0,
        "pretrain_edge_mode": "sobel",
        "percentile_lo": 1.0,
        "percentile_hi": 99.0,
        "gamma": 1.0,
    }


def load_settings(path: Path) -> dict[str, dict[str, Any]]:
    settings = json.loads(path.read_text())
    by_method = {method_id(row): dict(row) for row in settings}
    by_method["scalar5"] = scalar5_setting()
    return by_method


def top_methods_for_image(
    rows: list[dict[str, str]],
    image: str,
    *,
    top_k: int,
) -> list[dict[str, str]]:
    image_rows = [row for row in rows if row.get("image") == image]
    image_rows.sort(key=lambda row: float(row["operator_error_calibrated"]))
    return image_rows[:top_k]


def build_case_manifest(
    comparison_rows: list[dict[str, str]],
    settings_by_method: dict[str, dict[str, Any]],
    *,
    target_rms: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    selected_settings: list[dict[str, Any]] = []
    selected_seen: set[str] = set()

    def remember_setting(name: str) -> dict[str, Any]:
        if name not in settings_by_method:
            raise KeyError(f"Missing setting for method {name!r}")
        setting = dict(settings_by_method[name])
        setting["method"] = method_id(setting)
        if setting["method"] not in selected_seen:
            selected_settings.append(setting)
            selected_seen.add(setting["method"])
        return setting

    for image in IMAGES:
        chosen = list(top_methods_for_image(comparison_rows, image, top_k=top_k))
        existing = {row["pretrain_method"] for row in chosen}
        for extra in ["baseline_scalar1", "scalar5"]:
            if extra not in existing:
                chosen.append(
                    {
                        "image": image,
                        "direction": "signed_balanced",
                        "candidate_id": "signed_balanced__rms0p2",
                        "target_rms": "0.2",
                        "pretrain_method": extra,
                        "method_label": settings_by_method[extra]["label"],
                        "operator_error_calibrated": "",
                        "ssim": "",
                        "nrmse": "",
                    }
                )
        if len(chosen) != top_k + 2:
            raise RuntimeError(f"Expected {top_k + 2} rows for {image}, got {len(chosen)}")
        for rank, source_row in enumerate(chosen, start=1):
            name = source_row["pretrain_method"]
            setting = remember_setting(name)
            case_rows.append(
                {
                    "case_index": len(case_rows),
                    "source_rank_rms020": rank if rank <= top_k else "",
                    "image": image,
                    "direction": "signed_balanced",
                    "target_rms": float(target_rms),
                    "candidate_id": f"signed_balanced__rms{str(target_rms).replace('.', 'p')}",
                    "pretrain_method": name,
                    "method_label": setting["label"],
                    "source_operator_error_calibrated": source_row.get("operator_error_calibrated", ""),
                    "source_ssim": source_row.get("ssim", ""),
                    "source_nrmse": source_row.get("nrmse", ""),
                    "family": setting.get("family", ""),
                }
            )
    return case_rows, selected_settings


def validate(case_rows: list[dict[str, Any]], *, target_rms: float, top_k: int) -> None:
    if len(case_rows) != len(IMAGES) * (top_k + 2):
        raise RuntimeError(f"Expected {len(IMAGES) * (top_k + 2)} cases, got {len(case_rows)}")
    for image in IMAGES:
        rows = [row for row in case_rows if row["image"] == image]
        methods = {row["pretrain_method"] for row in rows}
        if len(rows) != top_k + 2:
            raise RuntimeError(f"{image}: expected {top_k + 2} rows, got {len(rows)}")
        if "baseline_scalar1" not in methods or "scalar5" not in methods:
            raise RuntimeError(f"{image}: missing baseline_scalar1 or scalar5")
    bad_rms = [row for row in case_rows if float(row["target_rms"]) != float(target_rms)]
    if bad_rms:
        raise RuntimeError(f"Found rows with wrong target RMS: {bad_rms[:3]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--settings-json", type=Path, default=DEFAULT_SETTINGS_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-rms", type=float, default=0.40)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    settings_by_method = load_settings(args.settings_json)
    case_rows, selected_settings = build_case_manifest(
        read_csv(args.comparison_csv),
        settings_by_method,
        target_rms=args.target_rms,
        top_k=args.top_k,
    )
    validate(case_rows, target_rms=args.target_rms, top_k=args.top_k)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(case_rows, args.out_dir / "case_manifest.csv")
    (args.out_dir / "case_manifest.json").write_text(json.dumps(case_rows, indent=2) + "\n")
    write_csv(selected_settings, args.out_dir / "settings_manifest.csv")
    (args.out_dir / "settings_manifest.json").write_text(json.dumps(selected_settings, indent=2) + "\n")
    print(f"[done] cases={len(case_rows)} settings={len(selected_settings)} out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
