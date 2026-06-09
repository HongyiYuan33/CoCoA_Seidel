"""Run fixed-Seidel oracle controls with even Seidel terms sign-flipped.

The reference oracle-control experiment used GT Seidel to synthesize the
measurement and then fixed Seidel to GT during recovery.  This script keeps the
measurement synthesis unchanged, but fixes the recovery Seidel vector to:

    even terms [W040, W222, W220, Wd] -> -GT
    odd terms  [W131, W311]           ->  GT

It reads the previous oracle CSV so the new rows can be compared case-by-case
against `seidel_gt_fixed`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_cocoa_like_2d_mechanism as cocoa  # noqa: E402
from run_cocoa_like_seidel_accuracy_sweep import (  # noqa: E402
    field_weighted_wavefront_rms,
)


REFERENCE_RUN_NAME = (
    "seidel_oracle_controls_4D_6D_4imgs_2dirs_rms006_020_040_"
    "seed0_noRMS_pre400_joint1000_20260607"
)
REFERENCE_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "cocoa_like_2d_mechanism"
    / REFERENCE_RUN_NAME
    / "oracle_controls_operator_input.csv"
)
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "cocoa_like_2d_mechanism"

EVEN_SEIDEL_INDICES = (0, 2, 3, 5)
ODD_SEIDEL_INDICES = (1, 4)
MODE_NAME = "even_flip_fixed"
LOSS_FAMILY = "even_sign_flip_fixed_recovery"


def parse_vector(text: str | list[float] | tuple[float, ...]) -> np.ndarray:
    if isinstance(text, (list, tuple)):
        arr = np.asarray(text, dtype=np.float64)
    else:
        arr = np.asarray(json.loads(str(text)), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.size != 6:
        raise ValueError(f"Expected a backend-6 Seidel vector, got {arr.size} values")
    return arr


def parse_int_list(text: str | list[int] | tuple[int, ...] | None) -> list[int]:
    if text in (None, ""):
        return []
    if isinstance(text, (list, tuple)):
        return [int(v) for v in text]
    return [int(v) for v in json.loads(str(text))]


def jsonish(value: Any) -> str:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return str(value)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        tmp.write_text("")
        tmp.replace(path)
        return
    preferred = [
        "stage",
        "run_root",
        "metrics_path",
        "reference_metrics_path",
        "seidel_convention",
        "dimension",
        "oracle_mode",
        "runner_mode",
        "loss_family",
        "image",
        "seed",
        "candidate_id",
        "direction",
        "target_wavefront_rms",
        "actual_wavefront_rms",
        "wavefront_gt_rms",
        "wavefront_recovered_rms",
        "wavefront_recovered_over_gt_rms",
        "relative_wavefront_error",
        "wavefront_error_rms",
        "l2_seidel_vs_gt",
        "ssim_recon_gain_vs_gt",
        "nrmse_recon_gain_vs_gt",
        "nrmse_meas_pred_vs_meas",
        "ssim_meas_pred_vs_meas",
        "measurement_hf_ratio",
        "recon_raw_hf_ratio",
        "pred_measurement_hf_ratio",
        "fixed_seidel_indices",
        "even_flip_indices",
        "odd_preserved_indices",
        "seidel_gt",
        "seidel_final",
    ]
    keys = sorted({key for row in rows for key in row})
    fieldnames = [key for key in preferred if key in keys] + [
        key for key in keys if key not in preferred
    ]
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonish(row.get(key, "")) for key in fieldnames})
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def reference_metrics_path(row: dict[str, str]) -> Path | None:
    raw = row.get("metrics_path") or ""
    if not raw:
        return None
    path = Path(raw)
    candidates = [
        path,
        PROJECT_ROOT / path,
        OUTPUT_BASE / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_reference_metrics(row: dict[str, str]) -> dict[str, Any]:
    path = reference_metrics_path(row)
    if path is None:
        return {}
    return json.loads(path.read_text())


def flip_even_seidel(gt: np.ndarray, fixed_indices: list[int]) -> np.ndarray:
    flipped = np.asarray(gt, dtype=np.float64).copy()
    flipped[list(EVEN_SEIDEL_INDICES)] *= -1.0
    if fixed_indices:
        flipped[fixed_indices] = 0.0
    return flipped


def coeff_rms(theta: np.ndarray) -> float:
    theta = np.asarray(theta, dtype=np.float64)
    return float(math.sqrt(float(np.mean(theta * theta))))


def add_wavefront_metrics(metrics: dict[str, Any], gt: np.ndarray, recovered: np.ndarray) -> None:
    gt_rms = field_weighted_wavefront_rms(gt)
    rec_rms = field_weighted_wavefront_rms(recovered)
    err_rms = field_weighted_wavefront_rms(recovered - gt)
    metrics.update(
        {
            "wavefront_gt_rms": gt_rms,
            "wavefront_recovered_rms": rec_rms,
            "wavefront_recovered_over_gt_rms": rec_rms / max(gt_rms, 1e-12),
            "wavefront_error_rms": err_rms,
            "relative_wavefront_error": err_rms / max(gt_rms, 1e-12),
            "coeff_gt_rms": coeff_rms(gt),
            "coeff_recovered_rms": coeff_rms(recovered),
            "coeff_recovered_over_gt_rms": coeff_rms(recovered) / max(coeff_rms(gt), 1e-12),
            "final_seidel_wavefront_rms_estimate": rec_rms,
            "final_seidel_coeff_rms_estimate": coeff_rms(recovered),
        }
    )


def default_config_from_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "image": row["image"],
        "size": int(row.get("size") or args.size),
        "modes": [],
        "run_name": None,
        "num_iter": int(args.num_iter),
        "pretrain_iter": int(args.pretrain_iter),
        "lr_obj": float(args.lr_obj),
        "lr_seidel": float(args.lr_seidel),
        "seidel_optimizer": args.seidel_optimizer,
        "rsd_weight": float(args.rsd_weight),
        "tv_weight": float(args.tv_weight),
        "pretrain_scalar": float(args.pretrain_scalar),
        "defocus_anchor_weight": float(args.defocus_anchor_weight),
        "defocus_index": int(args.defocus_index),
        "seidel_rms_floor_weight": 0.0,
        "seidel_rms_floor_alpha": 0.8,
        "seidel_rms_floor_target": None,
        "seidel_rms_floor_field_samples": 21,
        "seidel_rms_floor_pupil_samples": 51,
        "scheduler": "cosine",
        "eta_min_ratio": 0.04,
        "max_val": 40.0,
        "nerf_beta": 1.0,
        "output_mode": "softplus",
        "nerf_depth": 6,
        "nerf_width": 128,
        "nerf_skips": [2, 4, 6],
        "fourier_num_angles": 60,
        "fourier_num_octaves": 7,
        "seidel_convention": row["seidel_convention"],
        "gt_preset": "custom",
        "gt_seidel_json": row["seidel_gt"],
        "gt_label": row["candidate_id"],
        "gt_source": "custom",
        "seed": int(row.get("seed") or 0),
        "verbose": bool(args.train_verbose),
    }


def run_args_from_reference(
    row: dict[str, str],
    reference_metrics: dict[str, Any],
    args: argparse.Namespace,
    *,
    gt: np.ndarray,
) -> SimpleNamespace:
    config = dict(default_config_from_row(row, args))
    config.update(reference_metrics.get("config", {}))
    config["image"] = row["image"]
    config["size"] = int(row.get("size") or config.get("size", args.size))
    config["num_iter"] = int(config.get("num_iter", args.num_iter))
    config["pretrain_iter"] = int(config.get("pretrain_iter", args.pretrain_iter))
    config["seidel_optimizer"] = str(config.get("seidel_optimizer", args.seidel_optimizer))
    config["seidel_convention"] = row["seidel_convention"]
    config["gt_preset"] = "custom"
    config["gt_seidel_json"] = json.dumps(gt.astype(float).tolist())
    config["gt_label"] = row["candidate_id"]
    config["gt_source"] = "custom"
    config["seed"] = int(row.get("seed") or config.get("seed", 0))
    config["verbose"] = bool(args.train_verbose)
    config["modes"] = []
    config["run_name"] = None
    config["oracle_mode"] = MODE_NAME
    config["runner_mode"] = MODE_NAME
    config["loss_family"] = LOSS_FAMILY
    config["fixed_seidel_indices_override"] = parse_int_list(row.get("fixed_seidel_indices"))

    skips = config.get("nerf_skips", [2, 4, 6])
    if isinstance(skips, str):
        skips = cocoa.parse_nerf_skips(skips)
    else:
        skips = tuple(int(v) for v in skips)
    config["nerf_skips"] = skips
    if config.get("scheduler") == "none":
        config["scheduler"] = None
    return SimpleNamespace(**config)


def run_fixed_recovery(
    run_args: SimpleNamespace,
    *,
    mode: str,
    sharp_gt: torch.Tensor,
    meas_gt: torch.Tensor,
    measurement_gt_vec: torch.Tensor,
    recovery_vec: torch.Tensor,
    gt_np: np.ndarray,
    root_dir: Path,
    device: torch.device,
) -> tuple[cocoa.CocoaLikeResult, dict[str, Any]]:
    out_dir = root_dir / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(run_args.seed))
    np.random.seed(int(run_args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(run_args.seed))

    net_obj = cocoa.CocoaLikeObject2D(
        max_val=float(run_args.max_val),
        beta=float(run_args.nerf_beta),
        output_mode=str(run_args.output_mode),
        depth=int(run_args.nerf_depth),
        width=int(run_args.nerf_width),
        skips=tuple(run_args.nerf_skips),
        fourier_num_angles=int(run_args.fourier_num_angles),
        fourier_num_octaves=int(run_args.fourier_num_octaves),
    ).to(device)

    seidel = nn.Parameter(recovery_vec.detach().clone().to(device=device, dtype=sharp_gt.dtype))
    seidel.requires_grad_(False)

    print(
        f"[start] mode={mode} image={run_args.image} size={run_args.size} "
        f"device={device} pretrain={run_args.pretrain_iter} joint={run_args.num_iter} "
        f"mlp={run_args.nerf_depth}x{run_args.nerf_width} "
        f"skips={cocoa.format_nerf_skips(run_args.nerf_skips)} "
        f"fourier={run_args.fourier_num_angles}x{run_args.fourier_num_octaves} "
        f"fixed_recovery_seidel=[{', '.join(f'{x:.4g}' for x in recovery_vec.detach().cpu())}]",
        flush=True,
    )
    t0 = time.time()
    pretrain_history = cocoa.pretrain_cocoa_like(
        net_obj,
        meas_gt,
        num_iter=int(run_args.pretrain_iter),
        lr=float(run_args.lr_obj),
        measurement_scalar=float(run_args.pretrain_scalar),
        verbose=bool(run_args.verbose),
    )
    with torch.no_grad():
        pretrain_render = net_obj.render(*meas_gt.shape).detach()
        pretrain_target = (float(run_args.pretrain_scalar) * meas_gt.detach()).detach()
        pretrain_abs_error = torch.abs(pretrain_render - pretrain_target).detach()

    result = cocoa.train_cocoa_like(
        net_obj,
        seidel,
        meas_gt,
        cocoa.SYS_PARAMS,
        mode=mode,
        num_iter=int(run_args.num_iter),
        lr_obj=float(run_args.lr_obj),
        lr_seidel=float(run_args.lr_seidel),
        seidel_optimizer=str(run_args.seidel_optimizer),
        rsd_weight=float(run_args.rsd_weight),
        tv_weight=float(run_args.tv_weight),
        defocus_anchor_weight=float(run_args.defocus_anchor_weight),
        defocus_index=int(run_args.defocus_index),
        seidel_model_dim=cocoa.trace_model_dim(run_args.seidel_convention),
        fixed_seidel_indices=cocoa.resolved_fixed_seidel_indices(run_args),
        scheduler=run_args.scheduler,
        eta_min_ratio=float(run_args.eta_min_ratio),
        seidel_rms_floor_weight=float(run_args.seidel_rms_floor_weight),
        seidel_rms_floor_alpha=float(run_args.seidel_rms_floor_alpha),
        seidel_rms_floor_target=run_args.seidel_rms_floor_target,
        seidel_rms_floor_field_samples=int(run_args.seidel_rms_floor_field_samples),
        seidel_rms_floor_pupil_samples=int(run_args.seidel_rms_floor_pupil_samples),
        pretrain_history=pretrain_history,
        verbose=bool(run_args.verbose),
    )
    result = result._replace(elapsed_s=time.time() - t0)
    metrics = cocoa.compute_metrics(sharp_gt, meas_gt, result, gt_np, mode=mode, args=run_args)

    pretrain_render_np = pretrain_render.detach().cpu().numpy()
    pretrain_target_np = pretrain_target.detach().cpu().numpy()
    metrics.update(
        {
            "pretrain_final_loss": (
                float(pretrain_history[-1]) if pretrain_history else float("nan")
            ),
            "pretrain_render_ssim_vs_target": (
                float(1.0 - pretrain_history[-1]) if pretrain_history else float("nan")
            ),
            "pretrain_render_nrmse_vs_target": cocoa.compute_nrmse(
                pretrain_target_np,
                pretrain_render_np,
            ),
            "pretrain_render_hf_ratio": cocoa.high_frequency_ratio(pretrain_render_np),
            "measurement_seidel": measurement_gt_vec.detach().cpu().tolist(),
            "recovery_fixed_seidel": recovery_vec.detach().cpu().tolist(),
            "even_flip_indices": list(EVEN_SEIDEL_INDICES),
            "odd_preserved_indices": list(ODD_SEIDEL_INDICES),
        }
    )
    add_wavefront_metrics(metrics, gt_np, recovery_vec.detach().cpu().numpy())

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(
        {
            "sharp_gt": sharp_gt.detach().cpu(),
            "measurement_gt": meas_gt.detach().cpu(),
            "pretrain_target": pretrain_target.detach().cpu(),
            "pretrain_render": pretrain_render.detach().cpu(),
            "pretrain_abs_error": pretrain_abs_error.detach().cpu(),
            "sharp_recon": result.sharp_final.detach().cpu(),
            "measurement_pred": result.measurement_pred.detach().cpu(),
            "seidel_final": result.seidel_final.detach().cpu(),
            "seidel_gt": measurement_gt_vec.detach().cpu(),
            "loss_history": result.loss_history,
            "ssim_history": result.ssim_history,
            "rsd_history": result.rsd_history,
            "tv_history": result.tv_history,
            "anchor_history": result.anchor_history,
            "seidel_rms_floor_history": result.seidel_rms_floor_history,
            "seidel_wavefront_rms_history": result.seidel_wavefront_rms_history,
            "pretrain_history": result.pretrain_history,
        },
        out_dir / "tensors.pt",
    )
    cocoa.save_mode_figures(out_dir, sharp_gt, meas_gt, result, metrics, title=mode)
    print(
        f"[done] mode={mode} "
        f"SSIM_gain={metrics['ssim_recon_gain_vs_gt']:.4f} "
        f"NRMSE_gain={metrics['nrmse_recon_gain_vs_gt']:.4f} "
        f"HF_recon={metrics['recon_raw_hf_ratio']:.4f} "
        f"HF_meas={metrics['measurement_hf_ratio']:.4f} "
        f"L2_seidel={metrics['l2_seidel_vs_gt']:.4f} "
        f"elapsed={metrics['elapsed_s']:.1f}s",
        flush=True,
    )
    return result, metrics


def case_dir(output_root: Path, row: dict[str, str]) -> Path:
    return (
        output_root
        / row["seidel_convention"]
        / f"seed{int(row.get('seed') or 0)}"
        / f"{row['image']}__{row['candidate_id']}"
    )


def run_case(
    *,
    output_root: Path,
    row: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    root_dir = case_dir(output_root, row)
    metrics_path = root_dir / MODE_NAME / "metrics.json"
    if metrics_path.is_file() and not args.force:
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("sweep_case_complete") is True:
            return metrics

    reference_metrics = load_reference_metrics(row)
    gt = parse_vector(row["seidel_gt"])
    fixed_indices = parse_int_list(row.get("fixed_seidel_indices"))
    recovery = flip_even_seidel(gt, fixed_indices)
    run_args = run_args_from_reference(row, reference_metrics, args, gt=gt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gt_vec = torch.tensor(gt, device=device, dtype=torch.float32)
    recovery_vec = torch.tensor(recovery, device=device, dtype=torch.float32)
    sharp_gt = cocoa.load_baboon_gt(
        int(run_args.size),
        path=cocoa.IMAGE_PATHS[row["image"]],
        device=device,
    )
    meas_gt = cocoa.synthesize_measurement(sharp_gt, gt_vec, cocoa.SYS_PARAMS)

    print(
        f"[case] {row['seidel_convention']} seed={int(row.get('seed') or 0)} "
        f"image={row['image']} candidate={row['candidate_id']} oracle={MODE_NAME} "
        f"size={run_args.size} pre={run_args.pretrain_iter} joint={run_args.num_iter}",
        flush=True,
    )
    _, metrics = run_fixed_recovery(
        run_args,
        mode=MODE_NAME,
        sharp_gt=sharp_gt,
        meas_gt=meas_gt,
        measurement_gt_vec=gt_vec,
        recovery_vec=recovery_vec,
        gt_np=gt,
        root_dir=root_dir,
        device=device,
    )
    rel_metrics_path = metrics_path.relative_to(OUTPUT_BASE) if metrics_path.is_relative_to(OUTPUT_BASE) else metrics_path
    metrics.update(
        {
            "stage": "even_flip_oracle_controls",
            "run_root": str(output_root),
            "metrics_path": str(rel_metrics_path),
            "reference_metrics_path": row.get("metrics_path", ""),
            "seidel_convention": row["seidel_convention"],
            "dimension": row.get("dimension") or row["seidel_convention"].replace("classical", "").upper(),
            "oracle_mode": MODE_NAME,
            "runner_mode": MODE_NAME,
            "loss_family": LOSS_FAMILY,
            "image": row["image"],
            "seed": int(row.get("seed") or 0),
            "candidate_id": row["candidate_id"],
            "direction": row["direction"],
            "target_wavefront_rms": float(row["target_wavefront_rms"]),
            "actual_wavefront_rms": float(row.get("actual_wavefront_rms") or row["target_wavefront_rms"]),
            "fixed_seidel_indices": fixed_indices,
            "sweep_case_complete": True,
        }
    )
    write_json_atomic(metrics_path, metrics)
    write_json_atomic(
        root_dir / "summary.json",
        {
            "stage": "even_flip_oracle_controls",
            "image": row["image"],
            "seed": int(row.get("seed") or 0),
            "candidate_id": row["candidate_id"],
            "seidel_convention": row["seidel_convention"],
            "measurement_seidel_gt": gt.astype(float).tolist(),
            "recovery_fixed_seidel": recovery.astype(float).tolist(),
            "metrics_path": str(metrics_path),
        },
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def select_reference_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv(args.reference_csv)
        if row.get("oracle_mode") == "seidel_gt_fixed"
        and row.get("seidel_convention") in set(args.seidel_conventions)
    ]
    rows.sort(
        key=lambda row: (
            row["seidel_convention"],
            row["image"],
            row["direction"],
            float(row["target_wavefront_rms"]),
            int(row.get("seed") or 0),
        )
    )
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if args.num_shards > 1:
        rows = [
            row
            for idx, row in enumerate(rows)
            if idx % int(args.num_shards) == int(args.shard_index)
        ]
    return rows


def collect_new_metrics(output_root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text())
        for path in sorted(output_root.glob(f"classical*/seed*/*/{MODE_NAME}/metrics.json"))
    ]


def comparison_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["seidel_convention"]),
        str(row["image"]),
        str(row["candidate_id"]),
        int(row.get("seed") or 0),
    )


def build_comparison_rows(
    reference_rows: list[dict[str, str]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    old_by_key = {comparison_key(row): row for row in reference_rows}
    new_by_key = {comparison_key(row): row for row in new_rows}
    out: list[dict[str, Any]] = []
    for key in sorted(set(old_by_key) & set(new_by_key)):
        old = old_by_key[key]
        new = new_by_key[key]
        out.append(
            {
                "seidel_convention": key[0],
                "dimension": old.get("dimension") or new.get("dimension"),
                "image": key[1],
                "candidate_id": key[2],
                "seed": key[3],
                "direction": old["direction"],
                "target_wavefront_rms": float(old["target_wavefront_rms"]),
                "gt_fixed_ssim": float(old["ssim_recon_gain_vs_gt"]),
                "even_flip_ssim": float(new["ssim_recon_gain_vs_gt"]),
                "delta_ssim_even_minus_gtfixed": float(new["ssim_recon_gain_vs_gt"])
                - float(old["ssim_recon_gain_vs_gt"]),
                "gt_fixed_nrmse": float(old["nrmse_recon_gain_vs_gt"]),
                "even_flip_nrmse": float(new["nrmse_recon_gain_vs_gt"]),
                "delta_nrmse_even_minus_gtfixed": float(new["nrmse_recon_gain_vs_gt"])
                - float(old["nrmse_recon_gain_vs_gt"]),
                "gt_fixed_pred_meas_nrmse": float(old["nrmse_meas_pred_vs_meas"]),
                "even_flip_pred_meas_nrmse": float(new["nrmse_meas_pred_vs_meas"]),
                "delta_pred_meas_nrmse_even_minus_gtfixed": float(new["nrmse_meas_pred_vs_meas"])
                - float(old["nrmse_meas_pred_vs_meas"]),
                "gt_fixed_recon_hf": float(old["recon_raw_hf_ratio"]),
                "even_flip_recon_hf": float(new["recon_raw_hf_ratio"]),
                "measurement_hf_ratio": float(new["measurement_hf_ratio"]),
                "even_flip_l2_seidel_vs_gt": float(new["l2_seidel_vs_gt"]),
                "even_flip_relative_wavefront_error": float(new["relative_wavefront_error"]),
                "seidel_gt": old["seidel_gt"],
                "even_flip_seidel": new["seidel_final"],
                "even_flip_metrics_path": new["metrics_path"],
                "gt_fixed_metrics_path": old["metrics_path"],
            }
        )
    return out


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")


def summarize_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted(
        {
            (
                row["seidel_convention"],
                row["dimension"],
                row["direction"],
                float(row["target_wavefront_rms"]),
            )
            for row in rows
        },
        key=lambda item: (item[0], item[2], item[3]),
    )
    summary: list[dict[str, Any]] = []
    for convention, dimension, direction, target in groups:
        group = [
            row
            for row in rows
            if row["seidel_convention"] == convention
            and row["direction"] == direction
            and float(row["target_wavefront_rms"]) == target
        ]
        summary.append(
            {
                "seidel_convention": convention,
                "dimension": dimension,
                "direction": direction,
                "target_wavefront_rms": target,
                "num_rows": len(group),
                "mean_gt_fixed_ssim": mean([float(r["gt_fixed_ssim"]) for r in group]),
                "mean_even_flip_ssim": mean([float(r["even_flip_ssim"]) for r in group]),
                "mean_delta_ssim_even_minus_gtfixed": mean(
                    [float(r["delta_ssim_even_minus_gtfixed"]) for r in group]
                ),
                "mean_gt_fixed_nrmse": mean([float(r["gt_fixed_nrmse"]) for r in group]),
                "mean_even_flip_nrmse": mean([float(r["even_flip_nrmse"]) for r in group]),
                "mean_delta_nrmse_even_minus_gtfixed": mean(
                    [float(r["delta_nrmse_even_minus_gtfixed"]) for r in group]
                ),
                "mean_even_flip_pred_meas_nrmse": mean(
                    [float(r["even_flip_pred_meas_nrmse"]) for r in group]
                ),
                "mean_even_flip_relative_wavefront_error": mean(
                    [float(r["even_flip_relative_wavefront_error"]) for r in group]
                ),
            }
        )
    return summary


def write_readme(output_root: Path, comparison: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Even-Flip Fixed Seidel Oracle Control",
        "",
        "Measurement synthesis uses the same GT Seidel vectors as the previous oracle-control run.",
        "Recovery fixes even Seidel terms `[W040, W222, W220, Wd]` to the opposite sign and",
        "keeps odd terms `[W131, W311]` at their GT values.",
        "",
        "Files:",
        "- `even_flip_operator_input.csv`: per-case metrics for the new run.",
        "- `comparison_vs_seidel_gt_fixed.csv`: case-by-case comparison to the previous `seidel_gt_fixed` rows.",
        "- `summary_by_dimension_direction_rms.csv`: image-averaged summary by convention, direction, and RMS.",
        "",
    ]
    if summary:
        lines += [
            "## Summary",
            "",
            "| convention | direction | RMS | rows | GT-fixed SSIM | even-flip SSIM | delta SSIM | even-flip NRMSE |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary:
            lines.append(
                f"| {row['seidel_convention']} | {row['direction']} | "
                f"{row['target_wavefront_rms']:.2f} | {row['num_rows']} | "
                f"{row['mean_gt_fixed_ssim']:.4f} | {row['mean_even_flip_ssim']:.4f} | "
                f"{row['mean_delta_ssim_even_minus_gtfixed']:.4f} | "
                f"{row['mean_even_flip_nrmse']:.4f} |"
            )
        lines.append("")
    if comparison:
        worst = sorted(comparison, key=lambda row: float(row["delta_ssim_even_minus_gtfixed"]))[:8]
        lines += [
            "## Largest SSIM Drops",
            "",
            "| convention | image | candidate | delta SSIM | even-flip SSIM | GT-fixed SSIM |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in worst:
            lines.append(
                f"| {row['seidel_convention']} | {row['image']} | {row['candidate_id']} | "
                f"{row['delta_ssim_even_minus_gtfixed']:.4f} | "
                f"{row['even_flip_ssim']:.4f} | {row['gt_fixed_ssim']:.4f} |"
            )
        lines.append("")
    write_text_atomic(output_root / "README.md", "\n".join(lines))


def generate_reports(
    output_root: Path,
    reference_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    new_rows = collect_new_metrics(output_root)
    write_csv(new_rows, output_root / "even_flip_operator_input.csv")
    comparison = build_comparison_rows(reference_rows, new_rows)
    summary = summarize_comparison(comparison)
    write_csv(comparison, output_root / "comparison_vs_seidel_gt_fixed.csv")
    write_csv(summary, output_root / "summary_by_dimension_direction_rms.csv")
    write_readme(output_root, comparison, summary)
    return comparison, summary


def run_operator_eval(output_root: Path, args: argparse.Namespace) -> None:
    rows = collect_new_metrics(output_root)
    for convention in sorted({row["seidel_convention"] for row in rows}):
        group = [row for row in rows if row["seidel_convention"] == convention]
        if not group:
            continue
        dim = int(group[0].get("size") or args.size)
        input_csv = output_root / f"even_flip_operator_input_{convention}.csv"
        eval_dir = output_root / f"operator_eval_{convention}_dim{dim}"
        write_csv(group, input_csv)
        cmd = [
            sys.executable,
            str(HERE / "evaluate_seidel_physical_operator_sweep.py"),
            str(input_csv),
            str(eval_dir),
            "--dim",
            str(dim),
            "--theta-convention",
            convention,
            "--dataset-twin-invariance-pass",
            args.dataset_twin_invariance_pass,
            "--resume",
        ]
        print(f"[operator-eval] convention={convention} rows={len(group)}", flush=True)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = dt.datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-csv", type=Path, default=REFERENCE_CSV)
    parser.add_argument(
        "--run-name",
        default=(
            "seidel_even_flip_fixed_oracle_controls_4D_6D_4imgs_2dirs_"
            f"rms006_020_040_seed0_noRMS_pre400_joint1000_{stamp}"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seidel-conventions", nargs="+", default=["classical4d", "classical6d"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--train-verbose", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--run-operator-eval", action="store_true")
    parser.add_argument(
        "--dataset-twin-invariance-pass",
        choices=["auto", "true", "false"],
        default="auto",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num-iter", type=int, default=1000)
    parser.add_argument("--pretrain-iter", type=int, default=400)
    parser.add_argument("--lr-obj", type=float, default=5e-3)
    parser.add_argument("--lr-seidel", type=float, default=1e-2)
    parser.add_argument("--seidel-optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--rsd-weight", type=float, default=5e-4)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--pretrain-scalar", type=float, default=5.0)
    parser.add_argument("--defocus-anchor-weight", type=float, default=1.0)
    parser.add_argument("--defocus-index", type=int, default=5)
    args = parser.parse_args(argv)
    if args.output_root is None:
        args.output_root = OUTPUT_BASE / args.run_name
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    reference_rows = select_reference_rows(args)
    if args.report_only:
        full_args = argparse.Namespace(**vars(args))
        full_args.limit = None
        full_args.num_shards = 1
        full_args.shard_index = 0
        reference_rows = select_reference_rows(full_args)
        comparison, summary = generate_reports(args.output_root, reference_rows)
        if args.run_operator_eval:
            run_operator_eval(args.output_root, args)
        print(
            f"[report-only] cases={len(collect_new_metrics(args.output_root))} "
            f"comparison_rows={len(comparison)} summary_rows={len(summary)}",
            flush=True,
        )
        return

    write_json_atomic(
        args.output_root / "run_config.json",
        {
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "mode": MODE_NAME,
            "reference_csv": str(args.reference_csv),
            "even_flip_indices": list(EVEN_SEIDEL_INDICES),
            "odd_preserved_indices": list(ODD_SEIDEL_INDICES),
            "selected_cases": len(reference_rows),
        },
    )
    print(
        f"[start] run={args.run_name} selected_cases={len(reference_rows)} "
        f"output={args.output_root}",
        flush=True,
    )
    for row in reference_rows:
        run_case(output_root=args.output_root, row=row, args=args)
    if args.skip_report:
        comparison: list[dict[str, Any]] = []
        summary: list[dict[str, Any]] = []
    else:
        comparison, summary = generate_reports(args.output_root, reference_rows)
    if args.run_operator_eval and not args.skip_report:
        run_operator_eval(args.output_root, args)
    print(
        f"[done] cases={len(collect_new_metrics(args.output_root))} "
        f"comparison_rows={len(comparison)} summary_rows={len(summary)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
