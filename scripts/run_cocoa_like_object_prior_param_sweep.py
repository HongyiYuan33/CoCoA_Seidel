"""Overnight object-prior parameter sweep for the CoCoA-like 2-D runner."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_cocoa_like_2d_mechanism as cocoa  # noqa: E402
from run_cocoa_like_seidel_accuracy_sweep import field_weighted_wavefront_rms, norm01  # noqa: E402


IMAGES = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]
SEIDEL_PROFILES = {
    "cocoa_signed__rms0p06": [
        0.1388305276632309,
        -0.046276841312646866,
        0.023138420656323433,
        0.03702147305011749,
        0.0,
        0.03702147305011749,
    ],
    "cocoa_signed__rms0p10": [
        0.23138420283794403,
        -0.07712806761264801,
        0.038564033806324005,
        0.06170245632529259,
        0.0,
        0.06170245632529259,
    ],
    "signed_balanced__rms0p06": [
        0.13096117973327637,
        -0.04365372657775879,
        0.03819701075553894,
        0.03274029493331909,
        0.0,
        0.04365372657775879,
    ],
}
SOFTPLUS_MAX_VALS = [10.0, 20.0, 40.0, 80.0]
SOFTPLUS_RSD_WEIGHTS = [0.0, 1e-4, 5e-4, 1e-3, 2e-3]
SOFTPLUS_BETAS = [0.5, 1.0, 2.0, 5.0]
SIGMOID_MAX_VALS = [20.0, 40.0, 80.0]
SIGMOID_RSD_WEIGHTS = [0.0, 5e-4, 1e-3]
STAGE1 = {"size": 128, "pretrain_iter": 200, "num_iter": 500}
STAGE2 = {"size": 256, "pretrain_iter": 400, "num_iter": 1000}
STAGE3 = {"size": 256, "pretrain_iter": 400, "num_iter": 1000}


@dataclass(frozen=True)
class SeidelProfile:
    profile_id: str
    seidel: np.ndarray
    wavefront_rms: float


@dataclass(frozen=True)
class ParamConfig:
    param_id: str
    output_mode: str
    max_val: float
    rsd_weight: float
    nerf_beta: float
    is_control: bool


def tag_float(value: float) -> str:
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 0.01:
        return f"{value:.0e}".replace("e-0", "em").replace("e-", "em").replace("e+0", "ep").replace("e+", "ep")
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def make_profiles(profile_ids: list[str]) -> list[SeidelProfile]:
    profiles = []
    for profile_id in profile_ids:
        seidel = np.asarray(SEIDEL_PROFILES[profile_id], dtype=np.float32)
        profiles.append(
            SeidelProfile(
                profile_id=profile_id,
                seidel=seidel,
                wavefront_rms=field_weighted_wavefront_rms(seidel),
            )
        )
    return profiles


def make_param_configs(
    *,
    max_vals: list[float],
    rsd_weights: list[float],
    nerf_betas: list[float],
    include_sigmoid_controls: bool,
) -> list[ParamConfig]:
    configs = []
    for max_val in max_vals:
        for rsd_weight in rsd_weights:
            for beta in nerf_betas:
                configs.append(
                    ParamConfig(
                        param_id=(
                            f"softplus__max{tag_float(max_val)}__"
                            f"rsd{tag_float(rsd_weight)}__beta{tag_float(beta)}"
                        ),
                        output_mode="softplus",
                        max_val=float(max_val),
                        rsd_weight=float(rsd_weight),
                        nerf_beta=float(beta),
                        is_control=False,
                    )
                )
    if include_sigmoid_controls:
        for max_val in SIGMOID_MAX_VALS:
            for rsd_weight in SIGMOID_RSD_WEIGHTS:
                configs.append(
                    ParamConfig(
                        param_id=f"sigmoid__max{tag_float(max_val)}__rsd{tag_float(rsd_weight)}__beta1",
                        output_mode="sigmoid",
                        max_val=float(max_val),
                        rsd_weight=float(rsd_weight),
                        nerf_beta=1.0,
                        is_control=True,
                    )
                )
    return configs


def shard_items(items: list, *, shard_index: int, num_shards: int) -> list:
    if num_shards <= 1:
        return list(items)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_index]


def run_args_for_case(
    *,
    image: str,
    profile: SeidelProfile,
    param_config: ParamConfig,
    size: int,
    pretrain_iter: int,
    num_iter: int,
    seed: int,
    train_verbose: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        image=image,
        size=size,
        modes=["joint"],
        run_name=None,
        num_iter=num_iter,
        pretrain_iter=pretrain_iter,
        lr_obj=5e-3,
        lr_seidel=1e-2,
        rsd_weight=param_config.rsd_weight,
        tv_weight=0.0,
        pretrain_scalar=5.0,
        defocus_anchor_weight=1.0,
        defocus_index=5,
        scheduler="cosine",
        eta_min_ratio=1.0 / 25.0,
        max_val=param_config.max_val,
        nerf_beta=param_config.nerf_beta,
        output_mode=param_config.output_mode,
        seidel_convention="backend6",
        gt_preset="custom",
        gt_seidel_json=json.dumps(profile.seidel.astype(float).tolist()),
        gt_label=profile.profile_id,
        gt_source="custom",
        seed=seed,
        verbose=train_verbose,
    )


def augment_metrics(metrics: dict, *, stage: str, image: str, profile: SeidelProfile, param_config: ParamConfig, seed: int) -> dict:
    gt = np.asarray(metrics["seidel_gt"], dtype=np.float64)
    rec = np.asarray(metrics["seidel_final"], dtype=np.float64)
    gt_rms = field_weighted_wavefront_rms(gt)
    rec_rms = field_weighted_wavefront_rms(rec)
    err_rms = field_weighted_wavefront_rms(rec - gt)
    gt_hf = float(metrics.get("gt_hf_ratio", 0.0))
    meas_hf = float(metrics.get("measurement_hf_ratio", 0.0))
    recon_hf = float(metrics.get("recon_gain_hf_ratio", metrics.get("recon_raw_hf_ratio", 0.0)))
    gt_rsd = float(metrics.get("gt_std_over_mean", 0.0))
    recon_rsd = float(metrics.get("recon_gain_std_over_mean", metrics.get("recon_raw_std_over_mean", 0.0)))
    rel_wf = err_rms / max(gt_rms, 1e-12)
    hf_ratio = recon_hf / max(gt_hf, 1e-12)
    rsd_ratio = recon_rsd / max(gt_rsd, 1e-12)
    ssim = float(metrics.get("ssim_recon_gain_vs_gt") or 0.0)
    nrmse = float(metrics.get("nrmse_recon_gain_vs_gt") or 0.0)
    fake_sharpness_penalty = max(0.0, hf_ratio - 1.25) + max(0.0, rsd_ratio - 1.50)
    balanced_score = rel_wf + 0.50 * max(0.0, 0.90 - ssim) + 0.20 * nrmse + 0.25 * fake_sharpness_penalty
    metrics.update(
        {
            "stage": stage,
            "image": image,
            "seed": seed,
            "profile_id": profile.profile_id,
            "profile_wavefront_rms": profile.wavefront_rms,
            "param_id": param_config.param_id,
            "output_mode": param_config.output_mode,
            "max_val": param_config.max_val,
            "rsd_weight": param_config.rsd_weight,
            "nerf_beta": param_config.nerf_beta,
            "tv_weight": 0.0,
            "is_control": param_config.is_control,
            "wavefront_gt_rms": gt_rms,
            "wavefront_recovered_rms": rec_rms,
            "wavefront_error_rms": err_rms,
            "relative_wavefront_error": rel_wf,
            "seidel_l2_relative": float(np.linalg.norm(rec - gt) / max(float(np.linalg.norm(gt)), 1e-12)),
            "measurement_hf_drop": 1.0 - meas_hf / max(gt_hf, 1e-12),
            "recon_gain_hf_to_gt_hf": hf_ratio,
            "recon_gain_rsd_to_gt_rsd": rsd_ratio,
            "hf_ratio_abs_error": abs(hf_ratio - 1.0),
            "rsd_ratio_abs_error": abs(rsd_ratio - 1.0),
            "fake_sharpness_penalty": fake_sharpness_penalty,
            "balanced_score": balanced_score,
        }
    )
    return metrics


def case_dir_for(output_root: Path, stage: str, image: str, profile: SeidelProfile, param_config: ParamConfig, seed: int) -> Path:
    seed_dir = f"seed{seed}" if stage == "stage3" else ""
    case_key = f"{image}__{profile.profile_id}__{param_config.param_id}"
    return output_root / stage / seed_dir / case_key if seed_dir else output_root / stage / case_key


def run_case(
    *,
    output_root: Path,
    stage: str,
    image: str,
    profile: SeidelProfile,
    param_config: ParamConfig,
    size: int,
    pretrain_iter: int,
    num_iter: int,
    seed: int,
    force: bool,
    train_verbose: bool,
) -> dict:
    case_dir = case_dir_for(output_root, stage, image, profile, param_config, seed)
    metrics_path = case_dir / "joint" / "metrics.json"
    if metrics_path.is_file() and not force:
        metrics = json.loads(metrics_path.read_text())
        if "param_id" not in metrics or "relative_wavefront_error" not in metrics:
            metrics = augment_metrics(metrics, stage=stage, image=image, profile=profile, param_config=param_config, seed=seed)
            metrics_path.write_text(json.dumps(metrics, indent=2))
        return metrics

    case_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_args = run_args_for_case(
        image=image,
        profile=profile,
        param_config=param_config,
        size=size,
        pretrain_iter=pretrain_iter,
        num_iter=num_iter,
        seed=seed,
        train_verbose=train_verbose,
    )
    gt_vec = torch.tensor(profile.seidel, device=device, dtype=torch.float32)
    sharp_gt = cocoa.load_baboon_gt(size, path=cocoa.IMAGE_PATHS[image], device=device)
    meas_gt = cocoa.synthesize_measurement(sharp_gt, gt_vec, cocoa.SYS_PARAMS)
    print(
        f"[case] {stage} seed={seed} image={image} profile={profile.profile_id} "
        f"param={param_config.param_id} size={size} pre={pretrain_iter} joint={num_iter}",
        flush=True,
    )
    result, metrics = cocoa.run_one_mode(
        run_args,
        mode="joint",
        sharp_gt=sharp_gt,
        meas_gt=meas_gt,
        gt_vec=gt_vec,
        gt_np=profile.seidel,
        root_dir=case_dir,
        device=device,
    )
    metrics = augment_metrics(metrics, stage=stage, image=image, profile=profile, param_config=param_config, seed=seed)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    cocoa.save_summary_figure(case_dir, sharp_gt, meas_gt, [("joint", result, metrics)])
    (case_dir / "summary.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "image": image,
                "seed": seed,
                "profile": {
                    "profile_id": profile.profile_id,
                    "seidel": profile.seidel.tolist(),
                    "wavefront_rms": profile.wavefront_rms,
                },
                "param_config": param_config.__dict__,
                "metrics_path": str(metrics_path),
            },
            indent=2,
        )
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def collect_metrics(output_root: Path, stage: str) -> list[dict]:
    if stage in {"stage1", "stage2"}:
        paths = sorted((output_root / stage).glob("*/joint/metrics.json"))
    elif stage == "stage3":
        paths = sorted((output_root / stage).glob("seed*/*/joint/metrics.json"))
    else:
        raise ValueError(stage)
    return [json.loads(path.read_text()) for path in paths]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    preferred = [
        "stage",
        "image",
        "seed",
        "profile_id",
        "profile_wavefront_rms",
        "param_id",
        "output_mode",
        "max_val",
        "rsd_weight",
        "nerf_beta",
        "tv_weight",
        "is_control",
        "relative_wavefront_error",
        "wavefront_error_rms",
        "balanced_score",
        "seidel_l2_relative",
        "l2_seidel_vs_gt",
        "ssim_recon_gain_vs_gt",
        "nrmse_recon_gain_vs_gt",
        "gt_hf_ratio",
        "measurement_hf_ratio",
        "measurement_hf_drop",
        "recon_gain_hf_ratio",
        "recon_gain_hf_to_gt_hf",
        "gt_std_over_mean",
        "recon_gain_std_over_mean",
        "recon_gain_rsd_to_gt_rsd",
        "fake_sharpness_penalty",
        "elapsed_s",
        "seidel_gt",
        "seidel_final",
    ]
    extra = sorted({k for row in rows for k in row if k not in preferred and not isinstance(row.get(k), (dict, list))})
    fieldnames = [k for k in preferred if any(k in row for row in rows)] + extra
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                out[key] = value
            writer.writerow(out)


def mean_metric(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def summarize_config_rows(rows: list[dict], *, group_keys: list[str], min_mean_ssim: float, min_image_ssim: float) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)
    summaries = []
    for _, group in grouped.items():
        ssims = [float(row.get("ssim_recon_gain_vs_gt") or 0.0) for row in group]
        if float(np.mean(ssims)) < min_mean_ssim or float(np.min(ssims)) < min_image_ssim:
            continue
        first = group[0]
        rel = [float(row["relative_wavefront_error"]) for row in group]
        hf_err = [float(row["hf_ratio_abs_error"]) for row in group]
        nrmse = [float(row["nrmse_recon_gain_vs_gt"]) for row in group]
        out = {
            "profile_id": first.get("profile_id", "ALL"),
            "param_id": first["param_id"],
            "output_mode": first["output_mode"],
            "max_val": float(first["max_val"]),
            "rsd_weight": float(first["rsd_weight"]),
            "nerf_beta": float(first["nerf_beta"]),
            "is_control": bool(first.get("is_control", False)),
            "mean_relative_wavefront_error": float(np.mean(rel)),
            "median_relative_wavefront_error": float(np.median(rel)),
            "max_relative_wavefront_error": float(np.max(rel)),
            "std_relative_wavefront_error": float(np.std(rel)),
            "mean_balanced_score": mean_metric(group, "balanced_score"),
            "mean_ssim": float(np.mean(ssims)),
            "min_ssim": float(np.min(ssims)),
            "mean_nrmse": float(np.mean(nrmse)),
            "mean_hf_ratio_abs_error": float(np.mean(hf_err)),
            "mean_recon_gain_hf_to_gt_hf": mean_metric(group, "recon_gain_hf_to_gt_hf"),
            "mean_recon_gain_rsd_to_gt_rsd": mean_metric(group, "recon_gain_rsd_to_gt_rsd"),
            "mean_fake_sharpness_penalty": mean_metric(group, "fake_sharpness_penalty"),
            "num_runs": len(group),
            "num_images": len({row["image"] for row in group}),
            "num_profiles": len({row["profile_id"] for row in group}),
        }
        summaries.append(out)
    summaries.sort(key=lambda r: (r["mean_balanced_score"], r["mean_relative_wavefront_error"], r["mean_nrmse"]))
    return summaries


def select_stage2_param_ids(rows: list[dict], args: argparse.Namespace) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for profile_id in args.profiles:
        profile_rows = [row for row in rows if row["profile_id"] == profile_id]
        summaries = summarize_config_rows(
            profile_rows,
            group_keys=["profile_id", "param_id"],
            min_mean_ssim=args.min_mean_ssim,
            min_image_ssim=args.min_image_ssim,
        )
        if not summaries:
            print(
                f"[warn] Stage 1 profile {profile_id} has no configs passing object filters; "
                "falling back to unfiltered balanced ranking.",
                flush=True,
            )
            summaries = summarize_config_rows(
                profile_rows,
                group_keys=["profile_id", "param_id"],
                min_mean_ssim=0.0,
                min_image_ssim=0.0,
            )
        if not summaries:
            raise RuntimeError(f"No Stage 1 configs available for profile {profile_id}")
        profile_selected = [row["param_id"] for row in summaries[: args.stage2_global_top]]
        rank = {row["param_id"]: idx for idx, row in enumerate(summaries)}
        for image in args.images:
            image_rows = [
                row
                for row in profile_rows
                if row["image"] == image
                and float(row.get("ssim_recon_gain_vs_gt") or 0.0) >= args.min_image_ssim
            ]
            if not image_rows:
                image_rows = [row for row in profile_rows if row["image"] == image]
            if image_rows:
                image_rows.sort(key=lambda row: (row["balanced_score"], row["relative_wavefront_error"]))
                profile_selected.append(image_rows[0]["param_id"])
        profile_selected = sorted(set(profile_selected), key=lambda pid: rank.get(pid, 10**9))
        selected[profile_id] = profile_selected[: args.max_stage2_configs_per_profile]
    return selected


def select_stage3_param_ids(rows: list[dict], args: argparse.Namespace) -> list[str]:
    summaries = summarize_config_rows(
        rows,
        group_keys=["param_id"],
        min_mean_ssim=args.min_mean_ssim,
        min_image_ssim=args.min_image_ssim,
    )
    if not summaries:
        print(
            "[warn] Stage 2 has no configs passing object filters; "
            "falling back to unfiltered balanced ranking.",
            flush=True,
        )
        summaries = summarize_config_rows(
            rows,
            group_keys=["param_id"],
            min_mean_ssim=0.0,
            min_image_ssim=0.0,
        )
    if not summaries:
        raise RuntimeError("No Stage 2 configs available")
    selected = [row["param_id"] for row in summaries[: args.stage3_global_top]]
    rank = {row["param_id"]: idx for idx, row in enumerate(summaries)}
    for profile_id in args.profiles:
        profile_summaries = summarize_config_rows(
            [row for row in rows if row["profile_id"] == profile_id],
            group_keys=["profile_id", "param_id"],
            min_mean_ssim=args.min_mean_ssim,
            min_image_ssim=args.min_image_ssim,
        )
        if not profile_summaries:
            profile_summaries = summarize_config_rows(
                [row for row in rows if row["profile_id"] == profile_id],
                group_keys=["profile_id", "param_id"],
                min_mean_ssim=0.0,
                min_image_ssim=0.0,
            )
        if profile_summaries:
            selected.append(profile_summaries[0]["param_id"])
    return sorted(set(selected), key=lambda pid: rank.get(pid, 10**9))[: args.max_stage3_configs]


def stage3_stability_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["profile_id"], row["param_id"], row["image"]), []).append(row)
    out = []
    by_profile_param: dict[tuple[str, str], list[dict]] = {}
    for (profile_id, param_id, image), group in grouped.items():
        by_profile_param.setdefault((profile_id, param_id), []).extend(group)
        out.append(summarize_stability_group(group, profile_id=profile_id, param_id=param_id, image=image))
    for (profile_id, param_id), group in by_profile_param.items():
        out.append(summarize_stability_group(group, profile_id=profile_id, param_id=param_id, image="ALL"))
    out.sort(key=lambda r: (r["profile_id"], r["param_id"], r["image"] != "ALL", r["image"]))
    return out


def summarize_stability_group(rows: list[dict], *, profile_id: str, param_id: str, image: str) -> dict:
    rel = np.asarray([float(row["relative_wavefront_error"]) for row in rows], dtype=np.float64)
    first = rows[0]
    return {
        "stage": "stage3",
        "profile_id": profile_id,
        "image": image,
        "param_id": param_id,
        "output_mode": first["output_mode"],
        "max_val": float(first["max_val"]),
        "rsd_weight": float(first["rsd_weight"]),
        "nerf_beta": float(first["nerf_beta"]),
        "seeds": ",".join(str(int(row["seed"])) for row in sorted(rows, key=lambda r: int(r["seed"]))),
        "num_runs": len(rows),
        "mean_relative_wavefront_error": float(np.mean(rel)),
        "std_relative_wavefront_error": float(np.std(rel)),
        "min_relative_wavefront_error": float(np.min(rel)),
        "max_relative_wavefront_error": float(np.max(rel)),
        "mean_balanced_score": mean_metric(rows, "balanced_score"),
        "mean_ssim": mean_metric(rows, "ssim_recon_gain_vs_gt"),
        "min_ssim": float(np.min([float(row.get("ssim_recon_gain_vs_gt") or 0.0) for row in rows])),
        "mean_nrmse": mean_metric(rows, "nrmse_recon_gain_vs_gt"),
        "mean_hf_ratio_abs_error": mean_metric(rows, "hf_ratio_abs_error"),
        "mean_recon_gain_hf_to_gt_hf": mean_metric(rows, "recon_gain_hf_to_gt_hf"),
        "mean_recon_gain_rsd_to_gt_rsd": mean_metric(rows, "recon_gain_rsd_to_gt_rsd"),
        "mean_fake_sharpness_penalty": mean_metric(rows, "fake_sharpness_penalty"),
    }


def param_lookup(configs: list[ParamConfig]) -> dict[str, ParamConfig]:
    return {config.param_id: config for config in configs}


def profile_lookup(profiles: list[SeidelProfile]) -> dict[str, SeidelProfile]:
    return {profile.profile_id: profile for profile in profiles}


def stage_cases(images: list[str], profiles: list[SeidelProfile], configs: list[ParamConfig]) -> list[tuple[str, SeidelProfile, ParamConfig]]:
    return [(image, profile, config) for profile in profiles for image in images for config in configs]


def load_tensor_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_tensor_path(output_root: Path, stage: str, image: str, profile_id: str, param_id: str, seed: int = 0) -> Path | None:
    if stage == "stage3":
        path = output_root / stage / f"seed{seed}" / f"{image}__{profile_id}__{param_id}" / "joint" / "tensors.pt"
    else:
        path = output_root / stage / f"{image}__{profile_id}__{param_id}" / "joint" / "tensors.pt"
    return path if path.is_file() else None


def choose_overview_row(rows: list[dict], *, image: str, profile_id: str, param_id: str) -> dict | None:
    matches = [row for row in rows if row["image"] == image and row["profile_id"] == profile_id and row["param_id"] == param_id]
    if not matches:
        return None
    median = float(np.median([float(row["relative_wavefront_error"]) for row in matches]))
    return min(matches, key=lambda row: abs(float(row["relative_wavefront_error"]) - median))


def plot_heatmap_param_effects(rows: list[dict], output_root: Path) -> None:
    if not rows:
        return
    summaries = summarize_config_rows(rows, group_keys=["profile_id", "param_id"], min_mean_ssim=0.0, min_image_ssim=0.0)
    profiles = sorted({row["profile_id"] for row in summaries})
    params = sorted({row["param_id"] for row in summaries})
    matrix = np.full((len(params), len(profiles)), np.nan, dtype=np.float64)
    lookup = {(row["param_id"], row["profile_id"]): row for row in summaries}
    for i, param_id in enumerate(params):
        for j, profile_id in enumerate(profiles):
            row = lookup.get((param_id, profile_id))
            if row:
                matrix[i, j] = row["mean_relative_wavefront_error"]
    fig_h = max(9, 0.18 * len(params))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis_r", vmin=0, vmax=np.nanpercentile(matrix, 95))
    ax.set_xticks(range(len(profiles)), labels=profiles, rotation=20, ha="right")
    ax.set_yticks(range(len(params)), labels=params, fontsize=5)
    ax.set_title("Mean relative wavefront error by parameter config and Seidel profile")
    fig.colorbar(im, ax=ax, label="mean relative wavefront error")
    fig.tight_layout()
    fig.savefig(output_root / "heatmap_param_effects.png", dpi=160)
    plt.close(fig)


def plot_rsd_weight_vs_recovery(rows: list[dict], output_root: Path) -> None:
    if not rows:
        return
    soft = [row for row in rows if row["output_mode"] == "softplus"]
    summaries = summarize_config_rows(soft, group_keys=["rsd_weight"], min_mean_ssim=0.0, min_image_ssim=0.0)
    summaries.sort(key=lambda row: float(row["rsd_weight"]))
    x = [row["rsd_weight"] for row in summaries]
    fig, ax1 = plt.subplots(figsize=(7, 4.8))
    ax1.plot(x, [row["mean_relative_wavefront_error"] for row in summaries], marker="o", label="mean rel WF err")
    ax1.set_xscale("symlog", linthresh=1e-5)
    ax1.set_xlabel("RSD weight")
    ax1.set_ylabel("mean relative wavefront error")
    ax2 = ax1.twinx()
    ax2.plot(x, [row["mean_ssim"] for row in summaries], marker="s", color="#54a24b", label="mean SSIM")
    ax2.set_ylabel("mean object SSIM")
    ax1.set_title("RSD weight vs recovery / object quality")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_root / "rsd_weight_vs_recovery.png", dpi=160)
    plt.close(fig)


def plot_max_val_beta_grid(rows: list[dict], output_root: Path) -> None:
    soft = [row for row in rows if row["output_mode"] == "softplus"]
    if not soft:
        return
    summaries = summarize_config_rows(soft, group_keys=["max_val", "nerf_beta"], min_mean_ssim=0.0, min_image_ssim=0.0)
    max_vals = sorted({float(row["max_val"]) for row in summaries})
    betas = sorted({float(row["nerf_beta"]) for row in summaries})
    matrix = np.full((len(max_vals), len(betas)), np.nan)
    for row in summaries:
        i = max_vals.index(float(row["max_val"]))
        j = betas.index(float(row["nerf_beta"]))
        matrix[i, j] = row["mean_relative_wavefront_error"]
    fig, ax = plt.subplots(figsize=(6, 4.8))
    im = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(betas)), labels=[str(v) for v in betas])
    ax.set_yticks(range(len(max_vals)), labels=[str(int(v)) for v in max_vals])
    ax.set_xlabel("Softplus beta")
    ax.set_ylabel("max_val")
    ax.set_title("Softplus max_val × beta effect")
    for i in range(len(max_vals)):
        for j in range(len(betas)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean relative wavefront error")
    fig.tight_layout()
    fig.savefig(output_root / "max_val_beta_grid.png", dpi=160)
    plt.close(fig)


def plot_object_sharpness_vs_wavefront_error(rows: list[dict], output_root: Path) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [float(row["rsd_weight"]) for row in rows]
    scatter = ax.scatter(
        [row["recon_gain_hf_to_gt_hf"] for row in rows],
        [row["relative_wavefront_error"] for row in rows],
        c=colors,
        cmap="plasma",
        s=22,
        alpha=0.45,
        edgecolor="none",
    )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.45)
    ax.set_xlabel("recon HF / GT HF")
    ax.set_ylabel("relative wavefront error")
    ax.set_title("Object sharpness proxy vs Seidel recovery")
    fig.colorbar(scatter, ax=ax, label="RSD weight")
    fig.tight_layout()
    fig.savefig(output_root / "object_sharpness_vs_wavefront_error.png", dpi=160)
    plt.close(fig)


def plot_stage3_stability(rows: list[dict], output_root: Path) -> None:
    stability = [row for row in stage3_stability_rows(rows) if row["image"] == "ALL"]
    if not stability:
        return
    stability.sort(key=lambda row: (row["mean_balanced_score"], row["mean_relative_wavefront_error"]))
    labels = [f"{row['param_id']}\n{row['profile_id']}" for row in stability]
    x = np.arange(len(stability))
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(stability)), 5))
    ax.bar(x, [row["mean_relative_wavefront_error"] for row in stability], yerr=[row["std_relative_wavefront_error"] for row in stability], capsize=4)
    ax.set_xticks(x, labels=labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("mean relative wavefront error")
    ax.set_title("Stage 3 seed stability")
    fig.tight_layout()
    fig.savefig(output_root / "stage3_stability_bar.png", dpi=160)
    plt.close(fig)


def plot_overview(output_root: Path, rows: list[dict], images: list[str], profile_id: str, param_ids: list[str]) -> None:
    if not rows or not param_ids:
        return
    stage = rows[0]["stage"]
    fig, ax = plt.subplots(len(images), 1 + len(param_ids), figsize=(4 * (1 + len(param_ids)), 4 * len(images)))
    if ax.ndim == 1:
        ax = ax.reshape(len(images), 1 + len(param_ids))
    for row_idx, image in enumerate(images):
        first_row = choose_overview_row(rows, image=image, profile_id=profile_id, param_id=param_ids[0])
        first_path = find_tensor_path(output_root, stage, image, profile_id, param_ids[0], seed=int(first_row.get("seed", 0)) if first_row else 0)
        if first_path is None:
            continue
        first = load_tensor_dict(first_path)
        ax[row_idx, 0].imshow(norm01(first["sharp_gt"].numpy()), cmap="gray")
        ax[row_idx, 0].set_title(f"{image}\nGT")
        ax[row_idx, 0].axis("off")
        for col_idx, param_id in enumerate(param_ids, start=1):
            row = choose_overview_row(rows, image=image, profile_id=profile_id, param_id=param_id)
            tensor_path = find_tensor_path(output_root, stage, image, profile_id, param_id, seed=int(row.get("seed", 0)) if row else 0)
            if row is None or tensor_path is None:
                ax[row_idx, col_idx].axis("off")
                continue
            tensors = load_tensor_dict(tensor_path)
            ax[row_idx, col_idx].imshow(norm01(tensors["sharp_recon"].numpy()), cmap="gray")
            ax[row_idx, col_idx].set_title(
                f"{param_id}\nrelWF {row['relative_wavefront_error']:.3f} "
                f"SSIM {row['ssim_recon_gain_vs_gt']:.3f}",
                fontsize=7,
            )
            ax[row_idx, col_idx].axis("off")
    fig.suptitle(f"Top parameter configs overview ({stage}, {profile_id})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_root / "overview_top_param_configs.png", dpi=160)
    plt.close(fig)


def markdown_table(rows: list[dict], columns: list[tuple[str, str]], max_rows: int = 12) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = ["| " + " | ".join(label for _, label in columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows[:max_rows]:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                if abs(value) < 0.01 and value != 0:
                    values.append(f"{value:.1e}")
                else:
                    values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_best_param_configs(output_root: Path, args: argparse.Namespace) -> None:
    lines = ["# CoCoA-like Object Prior Parameter Sweep", ""]
    stage_rows = {}
    for stage in ["stage1", "stage2", "stage3"]:
        rows = collect_metrics(output_root, stage)
        if rows:
            stage_rows[stage] = rows

    for stage, rows in stage_rows.items():
        if stage == "stage3":
            stability = [row for row in stage3_stability_rows(rows) if row["image"] == "ALL"]
            stability.sort(key=lambda row: (row["mean_balanced_score"], row["mean_relative_wavefront_error"]))
            lines += [f"## {stage} balanced stability", ""]
            lines += markdown_table(
                stability,
                [
                    ("param_id", "param"),
                    ("profile_id", "profile"),
                    ("mean_relative_wavefront_error", "mean rel WF err"),
                    ("std_relative_wavefront_error", "std rel WF err"),
                    ("mean_ssim", "mean SSIM"),
                    ("mean_recon_gain_hf_to_gt_hf", "HF ratio"),
                    ("mean_fake_sharpness_penalty", "fake sharpness penalty"),
                ],
            )
            lines.append("")
            continue

        balanced = summarize_config_rows(
            rows,
            group_keys=["param_id"],
            min_mean_ssim=args.min_mean_ssim,
            min_image_ssim=args.min_image_ssim,
        )
        seidel_best = sorted(balanced, key=lambda row: (row["mean_relative_wavefront_error"], row["max_relative_wavefront_error"]))
        object_best = sorted(balanced, key=lambda row: (-row["mean_ssim"], row["mean_hf_ratio_abs_error"], row["mean_relative_wavefront_error"]))
        lines += [f"## {stage} balanced best", ""]
        lines += markdown_table(
            balanced,
            [
                ("param_id", "param"),
                ("mean_balanced_score", "balanced score"),
                ("mean_relative_wavefront_error", "mean rel WF err"),
                ("max_relative_wavefront_error", "max rel WF err"),
                ("mean_ssim", "mean SSIM"),
                ("mean_recon_gain_hf_to_gt_hf", "HF ratio"),
                ("mean_fake_sharpness_penalty", "fake sharpness penalty"),
            ],
        )
        lines += ["", f"## {stage} Seidel-accurate best", ""]
        lines += markdown_table(
            seidel_best,
            [
                ("param_id", "param"),
                ("mean_relative_wavefront_error", "mean rel WF err"),
                ("max_relative_wavefront_error", "max rel WF err"),
                ("mean_ssim", "mean SSIM"),
                ("mean_nrmse", "mean NRMSE"),
            ],
        )
        lines += ["", f"## {stage} object-sharp best", ""]
        lines += markdown_table(
            object_best,
            [
                ("param_id", "param"),
                ("mean_ssim", "mean SSIM"),
                ("mean_recon_gain_hf_to_gt_hf", "HF ratio"),
                ("mean_recon_gain_rsd_to_gt_rsd", "RSD ratio"),
                ("mean_relative_wavefront_error", "mean rel WF err"),
            ],
        )
        lines.append("")

    final_rows = stage_rows.get("stage3") or stage_rows.get("stage2") or stage_rows.get("stage1")
    if final_rows:
        if "stage3" in stage_rows:
            final = [row for row in stage3_stability_rows(stage_rows["stage3"]) if row["image"] == "ALL"]
            final.sort(key=lambda row: (row["mean_balanced_score"], row["mean_relative_wavefront_error"]))
        else:
            final = summarize_config_rows(
                final_rows,
                group_keys=["param_id"],
                min_mean_ssim=args.min_mean_ssim,
                min_image_ssim=args.min_image_ssim,
            )
        if final:
            top = final[0]
            lines += ["## Recommendation", ""]
            lines.append(
                f"Recommended config so far: `{top['param_id']}` "
                f"(mean relative WF error {top['mean_relative_wavefront_error']:.3f}, "
                f"mean SSIM {top['mean_ssim']:.3f})."
            )
            lines.append(
                "Watch fake sharpness when HF/RSD ratios are far above 1; those cases may look sharp "
                "while moving away from the true object/statistics."
            )
            lines.append("")

    output_root.joinpath("best_param_configs.md").write_text("\n".join(lines))


def generate_reports(output_root: Path, args: argparse.Namespace) -> None:
    stage1 = collect_metrics(output_root, "stage1")
    stage2 = collect_metrics(output_root, "stage2")
    stage3 = collect_metrics(output_root, "stage3")
    write_csv(stage1, output_root / "stage1_metrics.csv")
    write_csv(stage2, output_root / "stage2_metrics.csv")
    write_csv(stage3, output_root / "stage3_metrics_raw.csv")
    write_csv(stage3_stability_rows(stage3), output_root / "stage3_seed_stability.csv")
    if stage1:
        plot_heatmap_param_effects(stage1, output_root)
        plot_rsd_weight_vs_recovery(stage1, output_root)
        plot_max_val_beta_grid(stage1, output_root)
        plot_object_sharpness_vs_wavefront_error(stage1, output_root)
    if stage3:
        plot_stage3_stability(stage3, output_root)
    overview_rows = stage3 or stage2 or stage1
    if overview_rows:
        if stage3:
            summary = [row for row in stage3_stability_rows(stage3) if row["image"] == "ALL"]
            summary.sort(key=lambda row: (row["mean_balanced_score"], row["mean_relative_wavefront_error"]))
            profile_id = summary[0]["profile_id"] if summary else args.profiles[0]
            param_ids = []
            for row in summary:
                if row["profile_id"] == profile_id and row["param_id"] not in param_ids:
                    param_ids.append(row["param_id"])
                if len(param_ids) >= 3:
                    break
        else:
            summary = summarize_config_rows(
                overview_rows,
                group_keys=["profile_id", "param_id"],
                min_mean_ssim=args.min_mean_ssim,
                min_image_ssim=args.min_image_ssim,
            )
            profile_id = summary[0]["profile_id"] if summary else args.profiles[0]
            param_ids = [row["param_id"] for row in summary if row["profile_id"] == profile_id][:3]
        plot_overview(output_root, overview_rows, args.images, profile_id, param_ids)
    write_best_param_configs(output_root, args)


def run_stage1(output_root: Path, args: argparse.Namespace, profiles: list[SeidelProfile], configs: list[ParamConfig]) -> None:
    cases = stage_cases(args.images, profiles, configs)
    cases = shard_items(cases, shard_index=args.shard_index, num_shards=args.num_shards)
    for image, profile, config in cases:
        run_case(
            output_root=output_root,
            stage="stage1",
            image=image,
            profile=profile,
            param_config=config,
            size=args.stage1_size,
            pretrain_iter=args.stage1_pretrain_iter,
            num_iter=args.stage1_num_iter,
            seed=0,
            force=args.force,
            train_verbose=args.train_verbose,
        )


def run_stage2(output_root: Path, args: argparse.Namespace, profiles: list[SeidelProfile], configs: list[ParamConfig]) -> dict[str, list[str]]:
    stage1_rows = collect_metrics(output_root, "stage1")
    selected_by_profile = select_stage2_param_ids(stage1_rows, args)
    lookup = param_lookup(configs)
    cases = []
    for profile in profiles:
        for image in args.images:
            for param_id in selected_by_profile.get(profile.profile_id, []):
                cases.append((image, profile, lookup[param_id]))
    cases = shard_items(cases, shard_index=args.shard_index, num_shards=args.num_shards)
    for image, profile, config in cases:
        run_case(
            output_root=output_root,
            stage="stage2",
            image=image,
            profile=profile,
            param_config=config,
            size=args.stage2_size,
            pretrain_iter=args.stage2_pretrain_iter,
            num_iter=args.stage2_num_iter,
            seed=0,
            force=args.force,
            train_verbose=args.train_verbose,
        )
    (output_root / "stage2_selected_configs.json").write_text(json.dumps(selected_by_profile, indent=2))
    return selected_by_profile


def run_stage3(output_root: Path, args: argparse.Namespace, profiles: list[SeidelProfile], configs: list[ParamConfig]) -> list[str]:
    stage2_rows = collect_metrics(output_root, "stage2")
    selected = select_stage3_param_ids(stage2_rows, args)
    lookup = param_lookup(configs)
    all_cases = []
    for seed in args.stage3_seeds:
        for image, profile, config in stage_cases(args.images, profiles, [lookup[param_id] for param_id in selected]):
            all_cases.append((seed, image, profile, config))
    all_cases = shard_items(all_cases, shard_index=args.shard_index, num_shards=args.num_shards)
    for seed, image, profile, config in all_cases:
        run_case(
            output_root=output_root,
            stage="stage3",
            image=image,
            profile=profile,
            param_config=config,
            size=args.stage3_size,
            pretrain_iter=args.stage3_pretrain_iter,
            num_iter=args.stage3_num_iter,
            seed=seed,
            force=args.force,
            train_verbose=args.train_verbose,
        )
    return selected


def parse_float_list(values: list[str] | None, defaults: list[float]) -> list[float]:
    if values is None:
        return list(defaults)
    return [float(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--stage", choices=["all", "stage1", "stage2", "stage3", "report"], default="all")
    parser.add_argument("--images", nargs="+", choices=sorted(cocoa.IMAGE_PATHS), default=IMAGES)
    parser.add_argument("--profiles", nargs="+", choices=sorted(SEIDEL_PROFILES), default=list(SEIDEL_PROFILES))
    parser.add_argument("--param-ids", nargs="+", default=None)
    parser.add_argument("--max-vals", nargs="+", default=[str(v) for v in SOFTPLUS_MAX_VALS])
    parser.add_argument("--rsd-weights", nargs="+", default=[str(v) for v in SOFTPLUS_RSD_WEIGHTS])
    parser.add_argument("--nerf-betas", nargs="+", default=[str(v) for v in SOFTPLUS_BETAS])
    parser.add_argument("--no-sigmoid-controls", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train-verbose", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--min-mean-ssim", type=float, default=0.90)
    parser.add_argument("--min-image-ssim", type=float, default=0.75)
    parser.add_argument("--stage2-global-top", type=int, default=8)
    parser.add_argument("--max-stage2-configs-per-profile", type=int, default=12)
    parser.add_argument("--stage3-global-top", type=int, default=3)
    parser.add_argument("--max-stage3-configs", type=int, default=5)
    parser.add_argument("--stage3-seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--stage1-size", type=int, default=STAGE1["size"])
    parser.add_argument("--stage1-pretrain-iter", type=int, default=STAGE1["pretrain_iter"])
    parser.add_argument("--stage1-num-iter", type=int, default=STAGE1["num_iter"])
    parser.add_argument("--stage2-size", type=int, default=STAGE2["size"])
    parser.add_argument("--stage2-pretrain-iter", type=int, default=STAGE2["pretrain_iter"])
    parser.add_argument("--stage2-num-iter", type=int, default=STAGE2["num_iter"])
    parser.add_argument("--stage3-size", type=int, default=STAGE3["size"])
    parser.add_argument("--stage3-pretrain-iter", type=int, default=STAGE3["pretrain_iter"])
    parser.add_argument("--stage3-num-iter", type=int, default=STAGE3["num_iter"])
    args = parser.parse_args()
    args.max_vals = parse_float_list(args.max_vals, SOFTPLUS_MAX_VALS)
    args.rsd_weights = parse_float_list(args.rsd_weights, SOFTPLUS_RSD_WEIGHTS)
    args.nerf_betas = parse_float_list(args.nerf_betas, SOFTPLUS_BETAS)
    if args.run_name is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"object_prior_param_sweep_{stamp}"
    if args.stage == "all" and args.num_shards > 1:
        raise ValueError("Use --stage stage1/stage2/stage3 for sharded execution, not --stage all")
    return args


def main() -> None:
    args = parse_args()
    profiles = make_profiles(args.profiles)
    configs = make_param_configs(
        max_vals=args.max_vals,
        rsd_weights=args.rsd_weights,
        nerf_betas=args.nerf_betas,
        include_sigmoid_controls=not args.no_sigmoid_controls,
    )
    if args.param_ids is not None:
        wanted = set(args.param_ids)
        known = {config.param_id for config in configs}
        missing = sorted(wanted - known)
        if missing:
            raise ValueError(f"Unknown --param-ids: {missing}")
        configs = [config for config in configs if config.param_id in wanted]

    output_root = cocoa.PROJECT_ROOT / "outputs" / "cocoa_like_2d_mechanism" / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "param_grid_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "profiles": [
                    {"profile_id": profile.profile_id, "seidel": profile.seidel.tolist(), "wavefront_rms": profile.wavefront_rms}
                    for profile in profiles
                ],
                "param_configs": [config.__dict__ for config in configs],
            },
            indent=2,
        )
    )

    if args.stage in {"all", "stage1"}:
        run_stage1(output_root, args, profiles, configs)
        generate_reports(output_root, args)
    if args.stage in {"all", "stage2"}:
        selected = run_stage2(output_root, args, profiles, configs)
        print(f"[stage2] selected param configs: {selected}", flush=True)
        generate_reports(output_root, args)
    if args.stage in {"all", "stage3"}:
        selected = run_stage3(output_root, args, profiles, configs)
        print(f"[stage3] selected param configs: {selected}", flush=True)
        generate_reports(output_root, args)
    if args.stage == "report":
        generate_reports(output_root, args)
    print(f"[done] sweep root: {output_root}", flush=True)


if __name__ == "__main__":
    main()
