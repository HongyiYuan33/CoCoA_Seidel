#!/usr/bin/env python3
"""Run gauge-aware multi-start / beam search for 4D Seidel recovery."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_cocoa_like_2d_mechanism as cocoa  # noqa: E402
import run_cocoa_like_seidel_accuracy_sweep as accuracy  # noqa: E402
from build_lambda0_vs_10000_rcp_pairs import (  # noqa: E402
    build_case,
    coeff_ylim,
    collect_display_ranges,
    display_path,
    draw_coeff_card,
    draw_image_panel,
    parse_float,
    rms_label,
    safe_name,
)


OUTPUT_BASE = cocoa.PROJECT_ROOT / "outputs" / "cocoa_like_2d_mechanism"
RUN_NAME = (
    "seidel4d_gauge_beam_tunedadam_size256_4imgs_2dirs_rms006_020_040_"
    "G6_B2_short300_total1000_pre400_20260608"
)
BASELINE_RUN = "seidelopt_sgd4d_tunedprior_size256_four_images_pre400_joint1000_20260608"
IMAGES = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]
DIRECTIONS = ["cocoa_signed", "signed_balanced"]
STRENGTHS = [0.06, 0.20, 0.40]
GAUGE_SIGN_ALIASES = {
    "I": "I",
    "mirror_x": "mirror_x",
    "mirror_y": "I",
    "rot180": "mirror_x",
    "twin": "twin",
    "twin_mirror_x": "twin_mirror",
}
GAUGE_IMAGE_TRANSFORMS = {
    "I": "identity",
    "mirror_x": "mirror_x",
    "mirror_y": "mirror_y",
    "rot180": "rot180",
    "twin": "identity",
    "twin_mirror_x": "mirror_x",
}
SEIDEL_SIGNS = {
    "I": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    "mirror_x": (1.0, -1.0, 1.0, 1.0, -1.0, 1.0),
    "twin": (-1.0, 1.0, -1.0, -1.0, 1.0, -1.0),
    "twin_mirror": (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
}
IMAGE_ORDER = {name: idx for idx, name in enumerate(IMAGES)}
DIRECTION_ORDER = {name: idx for idx, name in enumerate(DIRECTIONS)}


@dataclass
class BranchState:
    gauge_chart: str
    image_transform: str
    seidel_transform: str
    sign_tensor: torch.Tensor
    net_obj: cocoa.CocoaLikeObject2D
    seidel: nn.Parameter
    optimizer: torch.optim.Optimizer
    scheduler: CosineAnnealingLR | None
    loss_history: list[float]
    ssim_history: list[float]
    rsd_history: list[float]
    tv_history: list[float]
    anchor_history: list[float]
    seidel_rms_floor_history: list[float]
    seidel_wavefront_rms_history: list[float]
    seidel_coeff_rms_history: list[float]
    elapsed_s: float = 0.0
    short_score: float = math.nan
    final_score: float = math.nan
    short_rank: int | None = None
    final_rank: int | None = None


def tag_float(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def score_tail(values: Sequence[float], n: int = 20) -> float:
    if not values:
        return math.inf
    tail = [float(value) for value in values[-int(n):] if math.isfinite(float(value))]
    if not tail:
        return math.inf
    return float(np.mean(tail))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, separators=(",", ":"))
                out[key] = value
            writer.writerow(out)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def rel_to_output_base(path: Path) -> str:
    try:
        return str(path.relative_to(OUTPUT_BASE))
    except ValueError:
        return str(path)


def resolve_output_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if (cocoa.PROJECT_ROOT / path).exists():
        return cocoa.PROJECT_ROOT / path
    return OUTPUT_BASE / path


def case_key(image: str, candidate_id: str, seed: int) -> str:
    return f"seed{int(seed)}__{safe_name(image)}__{safe_name(candidate_id)}"


def case_dir(output_root: Path, image: str, candidate_id: str, seed: int) -> Path:
    return output_root / "cases" / case_key(image, candidate_id, seed)


def primary_metrics_path(output_root: Path, image: str, candidate_id: str, seed: int) -> Path:
    return case_dir(output_root, image, candidate_id, seed) / "primary" / "metrics.json"


def make_run_args(args: argparse.Namespace, image: str, candidate: accuracy.Candidate) -> SimpleNamespace:
    return SimpleNamespace(
        image=image,
        size=int(args.size),
        modes=["joint"],
        run_name=None,
        num_iter=int(args.total_steps),
        pretrain_iter=int(args.pretrain_iter),
        lr_obj=float(args.lr_obj),
        lr_seidel=float(args.lr_seidel),
        rsd_weight=float(args.rsd_weight),
        tv_weight=float(args.tv_weight),
        pretrain_scalar=float(args.pretrain_scalar),
        defocus_anchor_weight=float(args.defocus_anchor_weight),
        defocus_index=int(args.defocus_index),
        seidel_parameterization="direct",
        seidel_rms_prior_mode="floor",
        seidel_rms_prior_measure="wavefront",
        seidel_rms_floor_weight=0.0,
        seidel_rms_floor_alpha=0.8,
        seidel_rms_floor_target=float(candidate.actual_rms),
        target_wavefront_rms=float(candidate.actual_rms),
        target_coeff_rms=float(candidate.coeff_rms),
        seidel_rms_floor_field_samples=int(args.seidel_rms_floor_field_samples),
        seidel_rms_floor_pupil_samples=int(args.seidel_rms_floor_pupil_samples),
        gt_fixed_seidel_indices=[],
        seidel_lr_multipliers=None,
        scheduler=None if args.scheduler == "none" else args.scheduler,
        eta_min_ratio=float(args.eta_min_ratio),
        max_val=float(args.max_val),
        nerf_beta=float(args.nerf_beta),
        output_mode=str(args.output_mode),
        nerf_depth=int(args.nerf_depth),
        nerf_width=int(args.nerf_width),
        nerf_skips=tuple(args.nerf_skips),
        fourier_num_angles=int(args.fourier_num_angles),
        fourier_num_octaves=int(args.fourier_num_octaves),
        seidel_convention="classical4d",
        gt_preset="custom",
        gt_seidel_json=json.dumps(candidate.seidel.astype(float).tolist()),
        gt_label=candidate.candidate_id,
        gt_source="custom",
        seed=int(args.seed),
        verbose=bool(args.train_verbose),
    )


def make_object(args: argparse.Namespace, device: torch.device) -> cocoa.CocoaLikeObject2D:
    return cocoa.CocoaLikeObject2D(
        max_val=float(args.max_val),
        beta=float(args.nerf_beta),
        output_mode=str(args.output_mode),
        depth=int(args.nerf_depth),
        width=int(args.nerf_width),
        skips=tuple(args.nerf_skips),
        fourier_num_angles=int(args.fourier_num_angles),
        fourier_num_octaves=int(args.fourier_num_octaves),
    ).to(device)


def clone_object_from_pretrained(
    pretrained: cocoa.CocoaLikeObject2D,
    args: argparse.Namespace,
    device: torch.device,
) -> cocoa.CocoaLikeObject2D:
    net = make_object(args, device)
    net.load_state_dict(pretrained.state_dict())
    return net


def image_gauge(tensor: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "identity":
        return tensor
    if transform == "mirror_x":
        return torch.flip(tensor, dims=(-1,))
    if transform == "mirror_y":
        return torch.flip(tensor, dims=(-2,))
    if transform == "rot180":
        return torch.flip(tensor, dims=(-2, -1))
    raise ValueError(f"Unknown image transform {transform!r}")


def build_branch(
    *,
    gauge_chart: str,
    pretrained: cocoa.CocoaLikeObject2D,
    args: argparse.Namespace,
    device: torch.device,
) -> BranchState:
    image_transform = GAUGE_IMAGE_TRANSFORMS[gauge_chart]
    seidel_transform = GAUGE_SIGN_ALIASES[gauge_chart]
    sign = torch.as_tensor(SEIDEL_SIGNS[seidel_transform], dtype=torch.float32, device=device)
    net = clone_object_from_pretrained(pretrained, args, device)
    seidel = nn.Parameter(torch.zeros(6, dtype=torch.float32, device=device))
    param_groups = [
        {"params": net.parameters(), "lr": float(args.lr_obj)},
        {"params": [seidel], "lr": float(args.lr_seidel)},
    ]
    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)
    scheduler = (
        CosineAnnealingLR(
            optimizer,
            T_max=int(args.total_steps),
            eta_min=float(args.lr_seidel) * float(args.eta_min_ratio),
        )
        if args.scheduler == "cosine"
        else None
    )
    return BranchState(
        gauge_chart=gauge_chart,
        image_transform=image_transform,
        seidel_transform=seidel_transform,
        sign_tensor=sign,
        net_obj=net,
        seidel=seidel,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_history=[],
        ssim_history=[],
        rsd_history=[],
        tv_history=[],
        anchor_history=[],
        seidel_rms_floor_history=[],
        seidel_wavefront_rms_history=[],
        seidel_coeff_rms_history=[],
    )


def canonical_seidel(branch: BranchState) -> torch.Tensor:
    fixed = cocoa.fixed_seidel_indices_for_convention("classical4d")
    out = branch.seidel
    if fixed:
        mask = torch.ones_like(out)
        mask[fixed] = 0.0
        out = out * mask
    return out


def train_branch(
    branch: BranchState,
    *,
    measurement_gt: torch.Tensor,
    args: argparse.Namespace,
    start_step: int,
    end_step: int,
    verbose_prefix: str,
) -> cocoa.CocoaLikeResult:
    height, width = measurement_gt.shape
    resolved_sys = cocoa.build_sys_params(height, cocoa.SYS_PARAMS)
    log_every = max(1, int(args.total_steps) // 10)
    t0 = time.time()
    sharp = torch.zeros_like(measurement_gt)
    measurement_pred = torch.zeros_like(measurement_gt)
    final_theta = canonical_seidel(branch)

    for step in range(int(start_step), int(end_step)):
        sharp = branch.net_obj.render(height, width)
        sharp_g = image_gauge(sharp, branch.image_transform)
        theta = canonical_seidel(branch)
        theta_forward = theta * branch.sign_tensor.to(device=theta.device, dtype=theta.dtype)
        pred_g = cocoa.blur_ring_trainable(sharp_g, theta_forward, resolved_sys)
        measurement_pred = image_gauge(pred_g, branch.image_transform)

        loss_ssim = cocoa.ssim_loss(measurement_pred, measurement_gt)
        loss_rsd = cocoa.reciprocal_std_contrast_loss(sharp)
        loss_tv = cocoa.tv_2d(sharp) if float(args.tv_weight) != 0.0 else torch.zeros_like(loss_ssim)
        loss_anchor = torch.zeros_like(loss_ssim)
        loss_rms_floor = torch.zeros_like(loss_ssim)
        seidel_wavefront_rms = torch.zeros_like(loss_ssim)
        seidel_coeff_rms = cocoa.torch_seidel_coefficient_rms(theta)
        loss = (
            loss_ssim
            + float(args.rsd_weight) * loss_rsd
            + float(args.tv_weight) * loss_tv
            + float(args.defocus_anchor_weight) * loss_anchor
        )

        branch.optimizer.zero_grad()
        loss.backward()
        branch.optimizer.step()
        with torch.no_grad():
            branch.seidel[4] = 0.0
            branch.seidel[5] = 0.0
        if branch.scheduler is not None:
            branch.scheduler.step()

        branch.loss_history.append(float(loss.item()))
        branch.ssim_history.append(float(loss_ssim.item()))
        branch.rsd_history.append(float(loss_rsd.item()))
        branch.tv_history.append(float(loss_tv.item()))
        branch.anchor_history.append(float(loss_anchor.item()))
        branch.seidel_rms_floor_history.append(float(loss_rms_floor.item()))
        branch.seidel_wavefront_rms_history.append(float(seidel_wavefront_rms.item()))
        branch.seidel_coeff_rms_history.append(float(seidel_coeff_rms.item()))
        final_theta = canonical_seidel(branch)

        if args.train_verbose and (step % log_every == 0 or step == int(end_step) - 1):
            coeffs = ", ".join(f"{value:.4f}" for value in final_theta.detach().cpu())
            print(
                f"[{verbose_prefix} {step:04d}] gauge={branch.gauge_chart} "
                f"total={loss.item():.6f} ssim={loss_ssim.item():.6f} "
                f"rsd={loss_rsd.item():.6f} seidel=[{coeffs}]",
                flush=True,
            )

    with torch.no_grad():
        sharp = branch.net_obj.render(height, width)
        theta = canonical_seidel(branch)
        theta_forward = theta * branch.sign_tensor.to(device=theta.device, dtype=theta.dtype)
        pred_g = cocoa.blur_ring_trainable(
            image_gauge(sharp, branch.image_transform),
            theta_forward,
            resolved_sys,
        )
        measurement_pred = image_gauge(pred_g, branch.image_transform)
        final_theta = theta

    branch.elapsed_s += time.time() - t0
    return cocoa.CocoaLikeResult(
        sharp_final=sharp.detach(),
        seidel_final=final_theta.detach().clone(),
        seidel_direction_raw_final=branch.seidel.detach().clone(),
        measurement_pred=measurement_pred.detach(),
        loss_history=list(branch.loss_history),
        ssim_history=list(branch.ssim_history),
        rsd_history=list(branch.rsd_history),
        tv_history=list(branch.tv_history),
        anchor_history=list(branch.anchor_history),
        seidel_rms_floor_history=list(branch.seidel_rms_floor_history),
        seidel_wavefront_rms_history=list(branch.seidel_wavefront_rms_history),
        seidel_coeff_rms_history=list(branch.seidel_coeff_rms_history),
        seidel_amplitude_history=[],
        seidel_direction_rms_history=[],
        pretrain_history=[],
        elapsed_s=float(branch.elapsed_s),
    )


def augment_branch_metrics(
    metrics: dict[str, Any],
    *,
    output_root: Path,
    metrics_path: Path,
    args: argparse.Namespace,
    candidate: accuracy.Candidate,
    branch: BranchState,
    phase: str,
    selected: bool,
    primary: bool,
) -> dict[str, Any]:
    metrics.update(
        {
            "stage": "gauge_beam",
            "run_root": str(output_root),
            "metrics_path": rel_to_output_base(metrics_path),
            "image": str(args.image),
            "seed": int(args.seed),
            "candidate_id": candidate.candidate_id,
            "direction": candidate.direction,
            "target_wavefront_rms": float(candidate.target_rms),
            "actual_wavefront_rms": float(candidate.actual_rms),
            "target_coeff_rms": float(candidate.coeff_rms),
            "seidel_convention": "classical4d",
            "dimension": "4D",
            "loss_family": "direct_no_RMS_original",
            "lambda": 0.0,
            "gauge_chart": branch.gauge_chart,
            "gauge_image_transform": branch.image_transform,
            "seidel_forward_gauge": branch.seidel_transform,
            "beam_phase": phase,
            "beam_selected": bool(selected),
            "beam_primary": bool(primary),
            "beam_width": int(args.beam_width),
            "num_gauges": len(args.gauges),
            "short_steps": int(args.short_steps),
            "total_steps": int(args.total_steps),
            "branch_score_source": "mean_last20_total_training_loss",
            "branch_score_short": float(branch.short_score),
            "branch_score_final": float(branch.final_score),
            "branch_short_rank": None if branch.short_rank is None else int(branch.short_rank),
            "branch_final_rank": None if branch.final_rank is None else int(branch.final_rank),
            **cocoa.convention_metadata("classical4d"),
        }
    )
    config = dict(metrics.get("config", {}))
    config.update(
        {
            "gauge_chart": branch.gauge_chart,
            "gauge_image_transform": branch.image_transform,
            "seidel_forward_gauge": branch.seidel_transform,
            "beam_phase": phase,
            "beam_selected": bool(selected),
            "beam_primary": bool(primary),
            "beam_width": int(args.beam_width),
            "num_gauges": len(args.gauges),
            "short_steps": int(args.short_steps),
            "total_steps": int(args.total_steps),
            "branch_score_source": "mean_last20_total_training_loss",
        }
    )
    metrics["config"] = config
    return metrics


def save_result(
    out_dir: Path,
    *,
    result: cocoa.CocoaLikeResult,
    metrics: dict[str, Any],
    sharp_gt: torch.Tensor,
    meas_gt: torch.Tensor,
    save_figures: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(
        {
            "sharp_gt": sharp_gt.detach().cpu(),
            "measurement_gt": meas_gt.detach().cpu(),
            "sharp_recon": result.sharp_final.detach().cpu(),
            "measurement_pred": result.measurement_pred.detach().cpu(),
            "seidel_final": result.seidel_final.detach().cpu(),
            "seidel_direction_raw_final": result.seidel_direction_raw_final.detach().cpu(),
            "loss_history": result.loss_history,
            "ssim_history": result.ssim_history,
            "rsd_history": result.rsd_history,
            "tv_history": result.tv_history,
            "anchor_history": result.anchor_history,
            "seidel_rms_floor_history": result.seidel_rms_floor_history,
            "seidel_wavefront_rms_history": result.seidel_wavefront_rms_history,
            "seidel_coeff_rms_history": result.seidel_coeff_rms_history,
            "pretrain_history": result.pretrain_history,
        },
        out_dir / "tensors.pt",
    )
    if save_figures:
        cocoa.save_mode_figures(out_dir, sharp_gt, meas_gt, result, metrics, title=out_dir.name)


def run_case(output_root: Path, args: argparse.Namespace, image: str, candidate: accuracy.Candidate) -> dict[str, Any]:
    metrics_path = primary_metrics_path(output_root, image, candidate.candidate_id, int(args.seed))
    if metrics_path.is_file() and not args.force:
        print(f"[skip] {image} {candidate.candidate_id}", flush=True)
        return json.loads(metrics_path.read_text())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_args = make_run_args(args, image, candidate)
    args.image = image
    gt_vec = torch.as_tensor(candidate.seidel, dtype=torch.float32, device=device)
    sharp_gt = cocoa.load_baboon_gt(int(args.size), path=cocoa.IMAGE_PATHS[image], device=device)
    meas_gt = cocoa.synthesize_measurement(sharp_gt, gt_vec, cocoa.SYS_PARAMS)
    case_root = case_dir(output_root, image, candidate.candidate_id, int(args.seed))
    case_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[case] image={image} candidate={candidate.candidate_id} "
        f"gauges={','.join(args.gauges)} B={args.beam_width} "
        f"pre={args.pretrain_iter} short={args.short_steps} total={args.total_steps}",
        flush=True,
    )

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    pretrained = make_object(args, device)
    pretrain_history = cocoa.pretrain_cocoa_like(
        pretrained,
        meas_gt,
        num_iter=int(args.pretrain_iter),
        lr=float(args.lr_obj),
        measurement_scalar=float(args.pretrain_scalar),
        verbose=bool(args.train_verbose),
    )

    branches = [
        build_branch(gauge_chart=gauge, pretrained=pretrained, args=args, device=device)
        for gauge in args.gauges
    ]
    short_rows: list[dict[str, Any]] = []
    for branch in branches:
        result = train_branch(
            branch,
            measurement_gt=meas_gt,
            args=args,
            start_step=0,
            end_step=int(args.short_steps),
            verbose_prefix="short",
        )
        result = result._replace(pretrain_history=list(pretrain_history))
        branch.short_score = score_tail(branch.loss_history, int(args.score_tail))
        metrics = cocoa.compute_metrics(
            sharp_gt,
            meas_gt,
            result,
            candidate.seidel,
            mode="joint",
            args=run_args,
        )
        branch_dir = case_root / "branches_short" / branch.gauge_chart
        metrics = augment_branch_metrics(
            metrics,
            output_root=output_root,
            metrics_path=branch_dir / "metrics.json",
            args=args,
            candidate=candidate,
            branch=branch,
            phase="short",
            selected=False,
            primary=False,
        )
        save_result(branch_dir, result=result, metrics=metrics, sharp_gt=sharp_gt, meas_gt=meas_gt, save_figures=False)
        short_rows.append(metrics)

    ranked = sorted(branches, key=lambda branch: (branch.short_score, branch.gauge_chart))
    for rank, branch in enumerate(ranked, start=1):
        branch.short_rank = rank
    selected = ranked[: int(args.beam_width)]
    selected_names = {branch.gauge_chart for branch in selected}
    for branch in branches:
        branch_path = case_root / "branches_short" / branch.gauge_chart / "metrics.json"
        if branch_path.is_file():
            metrics = json.loads(branch_path.read_text())
            metrics["branch_short_rank"] = int(branch.short_rank)
            metrics["beam_selected"] = branch.gauge_chart in selected_names
            metrics["metrics_path"] = rel_to_output_base(branch_path)
            config = dict(metrics.get("config", {}))
            config["branch_short_rank"] = int(branch.short_rank)
            config["beam_selected"] = branch.gauge_chart in selected_names
            metrics["config"] = config
            branch_path.write_text(json.dumps(metrics, indent=2))

    selected_metrics: list[dict[str, Any]] = []
    for branch in selected:
        result = train_branch(
            branch,
            measurement_gt=meas_gt,
            args=args,
            start_step=int(args.short_steps),
            end_step=int(args.total_steps),
            verbose_prefix="full",
        )
        result = result._replace(pretrain_history=list(pretrain_history))
        branch.final_score = score_tail(branch.loss_history, int(args.score_tail))
        metrics = cocoa.compute_metrics(
            sharp_gt,
            meas_gt,
            result,
            candidate.seidel,
            mode="joint",
            args=run_args,
        )
        out_dir = case_root / "selected" / f"rank{int(branch.short_rank):02d}__{branch.gauge_chart}"
        metrics = augment_branch_metrics(
            metrics,
            output_root=output_root,
            metrics_path=out_dir / "metrics.json",
            args=args,
            candidate=candidate,
            branch=branch,
            phase="selected_final",
            selected=True,
            primary=False,
        )
        save_result(out_dir, result=result, metrics=metrics, sharp_gt=sharp_gt, meas_gt=meas_gt, save_figures=True)
        selected_metrics.append(metrics)

    final_ranked = sorted(selected, key=lambda branch: (branch.final_score, branch.short_score, branch.gauge_chart))
    for rank, branch in enumerate(final_ranked, start=1):
        branch.final_rank = rank
        selected_path = case_root / "selected" / f"rank{int(branch.short_rank):02d}__{branch.gauge_chart}" / "metrics.json"
        if selected_path.is_file():
            metrics = json.loads(selected_path.read_text())
            metrics["branch_final_rank"] = int(branch.final_rank)
            metrics["branch_score_final"] = float(branch.final_score)
            metrics["metrics_path"] = rel_to_output_base(selected_path)
            config = dict(metrics.get("config", {}))
            config["branch_final_rank"] = int(branch.final_rank)
            config["branch_score_final"] = float(branch.final_score)
            metrics["config"] = config
            selected_path.write_text(json.dumps(metrics, indent=2))
    primary_branch = final_ranked[0]
    primary_source = case_root / "selected" / f"rank{int(primary_branch.short_rank):02d}__{primary_branch.gauge_chart}"
    primary_metrics = json.loads((primary_source / "metrics.json").read_text())
    primary_tensors = torch.load(primary_source / "tensors.pt", map_location="cpu")
    primary_result = cocoa.CocoaLikeResult(
        sharp_final=primary_tensors["sharp_recon"].to(device),
        seidel_final=primary_tensors["seidel_final"].to(device),
        seidel_direction_raw_final=primary_tensors["seidel_direction_raw_final"].to(device),
        measurement_pred=primary_tensors["measurement_pred"].to(device),
        loss_history=list(primary_tensors["loss_history"]),
        ssim_history=list(primary_tensors["ssim_history"]),
        rsd_history=list(primary_tensors["rsd_history"]),
        tv_history=list(primary_tensors["tv_history"]),
        anchor_history=list(primary_tensors["anchor_history"]),
        seidel_rms_floor_history=list(primary_tensors["seidel_rms_floor_history"]),
        seidel_wavefront_rms_history=list(primary_tensors["seidel_wavefront_rms_history"]),
        seidel_coeff_rms_history=list(primary_tensors["seidel_coeff_rms_history"]),
        seidel_amplitude_history=[],
        seidel_direction_rms_history=[],
        pretrain_history=list(primary_tensors["pretrain_history"]),
        elapsed_s=float(primary_metrics.get("elapsed_s", 0.0)),
    )
    primary_metrics["beam_primary"] = True
    primary_metrics["beam_phase"] = "primary"
    primary_metrics["branch_final_rank"] = 1
    primary_metrics["metrics_path"] = rel_to_output_base(metrics_path)
    primary_config = dict(primary_metrics.get("config", {}))
    primary_config["beam_primary"] = True
    primary_config["beam_phase"] = "primary"
    primary_metrics["config"] = primary_config
    save_result(
        case_root / "primary",
        result=primary_result,
        metrics=primary_metrics,
        sharp_gt=sharp_gt,
        meas_gt=meas_gt,
        save_figures=True,
    )
    cocoa.save_summary_figure(case_root, sharp_gt, meas_gt, [("gauge_beam_primary", primary_result, primary_metrics)])

    (case_root / "beam_case_summary.json").write_text(
        json.dumps(
            {
                "image": image,
                "candidate_id": candidate.candidate_id,
                "direction": candidate.direction,
                "target_wavefront_rms": float(candidate.target_rms),
                "seed": int(args.seed),
                "gauges": list(args.gauges),
                "beam_width": int(args.beam_width),
                "short_steps": int(args.short_steps),
                "total_steps": int(args.total_steps),
                "primary_gauge_chart": primary_branch.gauge_chart,
                "primary_short_rank": int(primary_branch.short_rank),
                "primary_final_rank": int(primary_branch.final_rank),
                "branch_short_scores": {
                    branch.gauge_chart: float(branch.short_score) for branch in branches
                },
                "selected_final_scores": {
                    branch.gauge_chart: float(branch.final_score) for branch in selected
                },
            },
            indent=2,
        )
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return primary_metrics


def build_cases(args: argparse.Namespace) -> list[tuple[str, accuracy.Candidate]]:
    candidates = accuracy.make_candidates(
        list(args.directions),
        [float(value) for value in args.strengths],
        seidel_convention="classical4d",
    )
    return [(image, candidate) for image in args.images for candidate in candidates]


def collect_case_rows(output_root: Path, args: argparse.Namespace, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image, candidate in build_cases(args):
        root = case_dir(output_root, image, candidate.candidate_id, int(args.seed))
        if phase == "primary":
            paths = [root / "primary" / "metrics.json"]
        elif phase == "selected":
            paths = sorted((root / "selected").glob("*/metrics.json"))
        elif phase == "short":
            paths = sorted((root / "branches_short").glob("*/metrics.json"))
        else:
            raise ValueError(phase)
        for path in paths:
            if path.is_file():
                rows.append(json.loads(path.read_text()))
    return sort_rows(rows)


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            IMAGE_ORDER.get(str(row.get("image")), 99),
            DIRECTION_ORDER.get(str(row.get("direction")), 99),
            float(row.get("target_wavefront_rms", 0.0)),
            str(row.get("gauge_chart", "")),
        ),
    )


def write_report_inputs(output_root: Path, args: argparse.Namespace) -> dict[str, int]:
    primary = collect_case_rows(output_root, args, "primary")
    selected = collect_case_rows(output_root, args, "selected")
    short = collect_case_rows(output_root, args, "short")
    write_csv(primary, output_root / "beam_primary_operator_input.csv")
    write_csv(selected, output_root / "beam_all_selected_operator_input.csv")
    write_csv(short, output_root / "beam_branch_short_summary.csv")
    expected = len(build_cases(args))
    status = {
        "expected_primary": expected,
        "completed_primary": len(primary),
        "expected_selected": expected * int(args.beam_width),
        "completed_selected": len(selected),
        "expected_short": expected * len(args.gauges),
        "completed_short": len(short),
    }
    (output_root / "run_status.json").write_text(json.dumps(status, indent=2))
    print(f"[report] primary {len(primary)}/{expected} selected {len(selected)} short {len(short)}", flush=True)
    return status


def collect_baseline_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    baseline_root = resolve_output_path(args.baseline_root)
    rows: list[dict[str, Any]] = []
    for image, candidate in build_cases(args):
        path = baseline_root / "stage1" / f"{image}__{candidate.candidate_id}" / "joint" / "metrics.json"
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text())
        metrics.update(
            {
                "stage": "baseline_Seidel4D_SGD_256_v1",
                "baseline_label": "Seidel4D-SGD-256-v1",
                "run_root": str(baseline_root),
                "metrics_path": rel_to_output_base(path),
                "image": image,
                "seed": int(args.seed),
                "candidate_id": candidate.candidate_id,
                "direction": candidate.direction,
                "target_wavefront_rms": float(candidate.target_rms),
                "actual_wavefront_rms": float(candidate.actual_rms),
                "target_coeff_rms": float(candidate.coeff_rms),
                "seidel_convention": "classical4d",
                "dimension": "4D",
                "loss_family": "direct_no_RMS_original",
                "lambda": 0.0,
                **cocoa.convention_metadata("classical4d"),
            }
        )
        rows.append(metrics)
    return sort_rows(rows)


def run_eval_pipeline(input_csv: Path, output_dir: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(HERE / "run_gauge_aware_operator_eval_pipeline.py"),
        str(input_csv),
        str(output_dir),
        "--dim",
        str(int(args.operator_eval_dim)),
        "--dataset-twin-invariance-pass",
        str(args.operator_eval_dataset_twin_invariance_pass),
        "--resume",
    ]
    print("[eval]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cocoa.PROJECT_ROOT), check=True)


def maybe_run_operator_eval(output_root: Path, args: argparse.Namespace) -> None:
    if args.skip_operator_eval:
        print("[eval] skipped by --skip-operator-eval", flush=True)
        return
    primary_input = output_root / "beam_primary_operator_input.csv"
    selected_input = output_root / "beam_all_selected_operator_input.csv"
    baseline_input = output_root / "baseline_Seidel4D_SGD_256_v1_operator_input.csv"
    baseline_rows = collect_baseline_rows(args)
    write_csv(baseline_rows, baseline_input)
    if len(baseline_rows) != len(build_cases(args)):
        raise RuntimeError(f"Baseline rows incomplete: {len(baseline_rows)}/{len(build_cases(args))}")
    run_eval_pipeline(primary_input, output_root / "beam_primary_gauge_aware_operator_eval_dim256", args)
    run_eval_pipeline(selected_input, output_root / "beam_selected_gauge_aware_operator_eval_dim256", args)
    run_eval_pipeline(baseline_input, output_root / "baseline_gauge_aware_operator_eval_dim256", args)


def row_key(row: dict[str, Any]) -> tuple[str, str, float]:
    return (
        str(row["image"]),
        str(row["direction"]),
        round(float(row["target_wavefront_rms"]), 6),
    )


def plot_metric_by_rms(rows: list[dict[str, Any]], metric: str, out_path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"baseline": "#6f6f6f", "beam": "#1f77b4"}
    for method in ["baseline", "beam"]:
        ys = []
        xs = []
        for rms in sorted({float(row["target_wavefront_rms"]) for row in rows}):
            vals = [
                float(row[f"{method}_{metric}"])
                for row in rows
                if math.isclose(float(row["target_wavefront_rms"]), rms, abs_tol=1e-8)
                and row.get(f"{method}_{metric}") not in (None, "")
                and math.isfinite(float(row[f"{method}_{metric}"]))
            ]
            if vals:
                xs.append(rms)
                ys.append(float(np.mean(vals)))
        ax.plot(xs, ys, marker="o", linewidth=2, label=method, color=colors[method])
    ax.set_xlabel("GT Seidel wavefront RMS")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_match_vector(value: Any) -> list[bool | None]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        parsed = value
    else:
        parsed = json.loads(str(value))
    out: list[bool | None] = []
    for item in parsed:
        if item is None:
            out.append(None)
        elif isinstance(item, bool):
            out.append(bool(item))
        else:
            out.append(str(item).lower() in {"true", "1", "yes"})
    return out


def plot_sign_by_coeff(baseline_rows: list[dict[str, Any]], beam_rows: list[dict[str, Any]], out_path: Path) -> None:
    labels = ["W040", "W131", "W222", "W220", "W311", "Wd"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    width = 0.34
    x = np.arange(len(labels), dtype=np.float64)
    for offset, name, rows, color in [
        (-width / 2, "baseline", baseline_rows, "#777777"),
        (width / 2, "beam", beam_rows, "#1f77b4"),
    ]:
        rates = []
        for idx in range(len(labels)):
            total = 0
            matches = 0
            for row in rows:
                vec = parse_match_vector(row.get("canonical_sign_match_gauge"))
                if idx >= len(vec) or vec[idx] is None:
                    continue
                total += 1
                matches += int(bool(vec[idx]))
            rates.append(float(matches / total) if total else math.nan)
        ax.bar(x + offset, rates, width=width, label=name, color=color)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("gauge-canonical sign match rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gauge_counts(rows: list[dict[str, Any]], out_path: Path) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        gauge = str(row.get("gauge_chart", "unknown"))
        counts[gauge] = counts.get(gauge, 0) + 1
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = sorted(counts)
    ax.bar(names, [counts[name] for name in names], color="#1f77b4")
    ax.set_ylabel("primary winner count")
    ax.set_title("Gauge-beam primary selected gauge counts")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_short_vs_final(rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for gauge in sorted({str(row.get("gauge_chart", "")) for row in rows}):
        sub = [row for row in rows if str(row.get("gauge_chart", "")) == gauge]
        x = [float(row["branch_score_short"]) for row in sub]
        y = [float(row["operator_error_calibrated"]) for row in sub]
        ax.scatter(x, y, label=gauge, alpha=0.78)
    ax.set_xlabel("short branch blind score (mean last-20 total loss)")
    ax.set_ylabel("final operator_error_calibrated")
    ax.set_title("Diagnostic only: short score vs final operator error")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_comparison_panel(
    baseline_case: dict[str, Any],
    beam_case: dict[str, Any],
    *,
    out_path: Path,
) -> dict[str, Any]:
    cases = [baseline_case, beam_case]
    ranges = collect_display_ranges(cases)
    ylimits = coeff_ylim(cases)
    row = beam_case["row"]
    target = parse_float(row, "target_wavefront_rms")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20.0, 9.0), dpi=150)
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.16, 1.0],
        height_ratios=[1, 1],
        left=0.027,
        right=0.985,
        top=0.91,
        bottom=0.055,
        wspace=0.045,
        hspace=0.22,
    )
    fig.suptitle(
        f"Baseline vs gauge-beam | 4D | {row['image']} | {row['direction']} | GT RMS {target:.2f}",
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    labels = [
        "baseline | Seidel4D-SGD-256-v1",
        f"gauge-beam | winner={row.get('gauge_chart', '?')} | short_rank={row.get('branch_short_rank', '?')}",
    ]
    for idx, (case, label) in enumerate(zip(cases, labels)):
        draw_image_panel(fig, outer[idx, 0], case, ranges)
        draw_coeff_card(fig, outer[idx, 1], case, ylimits, lambda_label=label)
        fig.text(0.013, 0.70 - idx * 0.42, label, rotation=90, ha="center", va="center", fontsize=10)
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "image": row["image"],
        "direction": row["direction"],
        "target_wavefront_rms": target,
        "gauge_chart": row.get("gauge_chart", ""),
        "branch_short_rank": row.get("branch_short_rank", ""),
        "branch_score_short": row.get("branch_score_short", ""),
        "path": display_path(out_path),
    }


def make_comparison_reports(output_root: Path, args: argparse.Namespace) -> None:
    report_dir = output_root / "baseline_vs_beam_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    baseline_eval = read_csv(output_root / "baseline_gauge_aware_operator_eval_dim256" / "gauge_aware_operator_metrics.csv")
    beam_eval = read_csv(output_root / "beam_primary_gauge_aware_operator_eval_dim256" / "gauge_aware_operator_metrics.csv")
    selected_eval = read_csv(output_root / "beam_selected_gauge_aware_operator_eval_dim256" / "gauge_aware_operator_metrics.csv")
    baseline_by_key = {row_key(row): row for row in baseline_eval}
    beam_by_key = {row_key(row): row for row in beam_eval}
    matched: list[dict[str, Any]] = []
    for key in sorted(beam_by_key, key=lambda item: (IMAGE_ORDER.get(item[0], 99), DIRECTION_ORDER.get(item[1], 99), item[2])):
        if key not in baseline_by_key:
            continue
        base = baseline_by_key[key]
        beam = beam_by_key[key]
        row = {
            "image": key[0],
            "direction": key[1],
            "target_wavefront_rms": key[2],
            "baseline_operator_error_calibrated": parse_float(base, "operator_error_calibrated"),
            "beam_operator_error_calibrated": parse_float(beam, "operator_error_calibrated"),
            "delta_operator_error_calibrated": parse_float(beam, "operator_error_calibrated")
            - parse_float(base, "operator_error_calibrated"),
            "baseline_canonical_recovered_over_gt_wavefront_rms_gauge": parse_float(
                base, "canonical_recovered_over_gt_wavefront_rms_gauge"
            ),
            "beam_canonical_recovered_over_gt_wavefront_rms_gauge": parse_float(
                beam, "canonical_recovered_over_gt_wavefront_rms_gauge"
            ),
            "baseline_canonical_sign_match_rate_gauge": parse_float(base, "canonical_sign_match_rate_gauge"),
            "beam_canonical_sign_match_rate_gauge": parse_float(beam, "canonical_sign_match_rate_gauge"),
            "beam_gauge_chart": beam.get("gauge_chart", ""),
            "beam_branch_short_rank": beam.get("branch_short_rank", ""),
            "beam_branch_score_short": beam.get("branch_score_short", ""),
        }
        matched.append(row)
    write_csv(matched, report_dir / "baseline_vs_beam_comparison.csv")
    plot_metric_by_rms(
        matched,
        "operator_error_calibrated",
        report_dir / "baseline_vs_beam_operator_error_by_rms.png",
        "mean operator_error_calibrated",
    )
    plot_metric_by_rms(
        matched,
        "canonical_recovered_over_gt_wavefront_rms_gauge",
        report_dir / "baseline_vs_beam_recovered_over_gt_rms_by_rms.png",
        "mean gauge-canonical recovered / GT wavefront RMS",
    )
    plot_sign_by_coeff(
        baseline_eval,
        beam_eval,
        report_dir / "baseline_vs_beam_sign_agreement_by_coeff.png",
    )
    plot_gauge_counts(beam_eval, report_dir / "beam_selected_gauge_counts.png")
    plot_short_vs_final(selected_eval, report_dir / "beam_short_score_vs_final_operator_error.png")

    output_base = OUTPUT_BASE
    manifest_rows: list[dict[str, Any]] = []
    rcp_dir = output_root / "baseline_vs_beam_RCP"
    for key, beam_row in beam_by_key.items():
        base_row = baseline_by_key.get(key)
        if base_row is None:
            continue
        try:
            base_case = build_case(base_row, output_base)
            beam_case = build_case(beam_row, output_base)
        except Exception as exc:
            print(f"[RCP skip] {key}: {exc}", flush=True)
            continue
        image, direction, target = key
        out_path = (
            rcp_dir
            / safe_name(image)
            / safe_name(direction)
            / rms_label(float(target))
            / f"baseline_vs_gauge_beam__{safe_name(image)}__{safe_name(direction)}__{rms_label(float(target))}.png"
        )
        manifest_rows.append(make_comparison_panel(base_case, beam_case, out_path=out_path))
    write_csv(manifest_rows, rcp_dir / "manifest.csv")
    (report_dir / "README.md").write_text(
        "\n".join(
            [
                "# Baseline vs Gauge-Beam Reports",
                "",
                "- Baseline: existing `Seidel4D-SGD-256-v1` results.",
                "- Beam branch selection uses blind training loss only.",
                "- `beam_short_score_vs_final_operator_error.png` is diagnostic-only and must not be read as a training selector.",
                f"- Matched comparison rows: {len(matched)}",
                f"- RCP panels: {len(manifest_rows)}",
                "",
            ]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--baseline-root", default=BASELINE_RUN)
    parser.add_argument("--images", nargs="+", choices=sorted(cocoa.IMAGE_PATHS), default=IMAGES)
    parser.add_argument("--directions", nargs="+", choices=sorted(accuracy.DIRECTIONS), default=DIRECTIONS)
    parser.add_argument("--strengths", nargs="+", default=[str(v) for v in STRENGTHS])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--pretrain-iter", type=int, default=400)
    parser.add_argument("--short-steps", type=int, default=300)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--score-tail", type=int, default=20)
    parser.add_argument("--gauges", nargs="+", default=list(GAUGE_SIGN_ALIASES))
    parser.add_argument("--lr-obj", type=float, default=5e-3)
    parser.add_argument("--lr-seidel", type=float, default=1e-2)
    parser.add_argument("--rsd-weight", type=float, default=5e-4)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--pretrain-scalar", type=float, default=5.0)
    parser.add_argument("--defocus-anchor-weight", type=float, default=1.0)
    parser.add_argument("--defocus-index", type=int, default=5)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--eta-min-ratio", type=float, default=1.0 / 25.0)
    parser.add_argument("--max-val", type=float, default=40.0)
    parser.add_argument("--nerf-beta", type=float, default=1.0)
    parser.add_argument("--output-mode", choices=["softplus", "sigmoid"], default="softplus")
    parser.add_argument("--nerf-depth", type=int, default=6)
    parser.add_argument("--nerf-width", type=int, default=128)
    parser.add_argument("--nerf-skips", default="2,4,6")
    parser.add_argument("--fourier-num-angles", type=int, default=60)
    parser.add_argument("--fourier-num-octaves", type=int, default=7)
    parser.add_argument("--seidel-rms-floor-field-samples", type=int, default=21)
    parser.add_argument("--seidel-rms-floor-pupil-samples", type=int, default=51)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train-verbose", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--skip-operator-eval", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--operator-eval-dim", type=int, default=256)
    parser.add_argument("--operator-eval-dataset-twin-invariance-pass", default="auto")
    args = parser.parse_args()
    args.nerf_skips = cocoa.parse_nerf_skips(args.nerf_skips)
    args.strengths = [str(float(v)) for v in args.strengths]
    args.gauges = list(args.gauges)
    bad_gauges = [gauge for gauge in args.gauges if gauge not in GAUGE_SIGN_ALIASES]
    if bad_gauges:
        raise ValueError(f"Unknown gauges: {bad_gauges}")
    if args.short_steps <= 0 or args.total_steps <= args.short_steps:
        raise ValueError("--total-steps must be greater than --short-steps")
    if args.beam_width < 1 or args.beam_width > len(args.gauges):
        raise ValueError("--beam-width must be in [1, len(gauges)]")
    if args.num_shards < 1 or args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    return args


def main() -> None:
    args = parse_args()
    output_root = OUTPUT_BASE / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args)
    selected_cases = [
        case for idx, case in enumerate(cases) if idx % int(args.num_shards) == int(args.shard_index)
    ]
    (output_root / "gauge_beam_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "gauge_sign_aliases": GAUGE_SIGN_ALIASES,
                "gauge_image_transforms": GAUGE_IMAGE_TRANSFORMS,
                "selection_score": "mean_last20_total_training_loss",
                "seidel_convention": "classical4d",
            },
            indent=2,
        )
    )
    print(
        f"[start] run={args.run_name} shard={args.shard_index}/{args.num_shards} "
        f"cases={len(selected_cases)}/{len(cases)}",
        flush=True,
    )
    if not args.report_only:
        for image, candidate in selected_cases:
            run_case(output_root, args, image, candidate)
    status = write_report_inputs(output_root, args)
    if not args.skip_operator_eval:
        if status["completed_primary"] == status["expected_primary"] and (
            int(args.num_shards) == 1 or args.report_only
        ):
            maybe_run_operator_eval(output_root, args)
            if not args.skip_comparison:
                make_comparison_reports(output_root, args)
        else:
            print("[report] operator eval waits for all shards; rerun with --report-only", flush=True)
    print(f"[done] output_root={output_root}", flush=True)


if __name__ == "__main__":
    main()
