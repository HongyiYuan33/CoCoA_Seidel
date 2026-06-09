"""Joint training of the 2-D NeRF and the 6 Seidel coefficients.

Given a blurred measurement produced by a known sharp image and a known
Seidel vector, :func:`train` jointly optimises a fresh
:class:`hybrid_ring_cocoa.NeuralObject2D` and a 6-element Seidel parameter
to recover both.  The loss is a CoCoA-style combination of ``ssim_loss`` on
the predicted measurement, a 2-D TV regulariser on the rendered sharp image,
and a hinge defocus anchor.

Reference: ``CoCoA-master/main/wf_cocoa_demo.ipynb`` (joint-optimisation
cell).  Two deliberate simplifications versus upstream:

* Seidel basis (6 coefficients) instead of Zernike (12).
* 2-D output instead of a 3-D z-stack — the TV regulariser is 2-D and there
  is no RSD (reciprocal-std) term.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..optics.ring_forward import (
    blur_ring_trace_trainable,
    blur_ring_trainable,
    blur_ring_with_psfs,
)
from ..optics.seidel_psf import (
    build_sys_params,
    get_trainable_ring_psfs,
    get_trainable_trace_ring_psfs,
)
from .losses import single_mode_control, ssim_loss, tv_2d


class TrainResult(NamedTuple):
    """Return value of :func:`train`.

    ``*_history`` are Python ``list[float]`` of length ``num_iter`` containing
    the unweighted component losses at each step (useful for diagnostic
    plots).  ``loss_history`` is the weighted total loss.
    """

    sharp_final: torch.Tensor
    seidel_final: torch.Tensor
    measurement_pred: torch.Tensor
    loss_history: list[float]
    ssim_history: list[float]
    tv_history: list[float]
    anchor_history: list[float]


def pretrain_object(
    net_obj,
    measurement_gt: torch.Tensor,
    *,
    num_iter: int = 200,
    lr: float = 1e-2,
    measurement_scalar: float = 5.0,
    verbose: bool = False,
) -> list[float]:
    """Optional helper: fit ``net_obj.render`` to a scaled measurement.

    Mirrors the NeRF-only pretraining phase in CoCoA's demo, before the
    optics are jointly optimised.  Uses SSIM only — no TV, no optics.
    Defaults to *off* (``pretrain_iter=0``) in :func:`train`; expose this so
    follow-up experiments can enable it without reimplementation.

    Returns the per-step loss history.
    """
    H, W = measurement_gt.shape
    target = measurement_scalar * measurement_gt.detach()
    optimizer = torch.optim.Adam(net_obj.parameters(), lr=lr)

    history: list[float] = []
    for step in range(num_iter):
        sharp = net_obj.render(H, W)
        loss = ssim_loss(sharp, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if verbose and (step % max(1, num_iter // 10) == 0):
            print(f"[pretrain {step:04d}] ssim={loss.item():.4f}")
    return history


def train(
    net_obj,
    seidel_coeffs: torch.nn.Parameter,
    measurement_gt: torch.Tensor,
    sys_params: dict | None = None,
    *,
    num_iter: int = 500,
    lr_obj: float = 5e-3,
    lr_seidel: float = 1e-2,
    seidel_optimizer: str = "adam",
    ssim_weight: float = 1.0,
    tv_weight: float = 1e-5,
    defocus_anchor_weight: float = 1.0,
    defocus_index: int = 5,
    seidel_model_dim: int | str | None = None,
    scheduler: str | None = "cosine",
    eta_min_ratio: float = 1.0 / 25.0,
    pretrain_iter: int = 0,
    verbose: bool = False,
) -> TrainResult:
    """Jointly train ``net_obj`` and ``seidel_coeffs`` against ``measurement_gt``.

    Parameters
    ----------
    net_obj : NeuralObject2D
        Freshly initialised (or pretrained) 2-D coordinate network.
    seidel_coeffs : nn.Parameter of shape (6,)
        Trainable Seidel vector.  Caller owns initialisation — a common
        choice is ``nn.Parameter(torch.zeros(6))``.
    measurement_gt : (N, N) tensor
        Supervision target.  Square, even-sided.
    sys_params : dict, optional
        Optical system parameters; forwarded to ``blur_ring_trainable``.
    num_iter : int
        Number of joint-optimisation steps.
    lr_obj, lr_seidel : float
        Learning rates for the two parameter groups.
    seidel_optimizer : {"adam", "sgd"}
        Optimizer for trainable Seidel coefficients.  The object MLP always
        uses Adam; ``"adam"`` preserves the historical shared-Adam path, while
        ``"sgd"`` uses plain SGD with ``momentum=0`` for Seidel coefficients.
    ssim_weight, tv_weight, defocus_anchor_weight : float
        Loss weights.  ``tv_weight`` scales a summed (not mean) TV, so
        1e-5 is a reasonable starting point for a 128×128 image in ``[0, 1]``.
    defocus_index : int
        Which Seidel coefficient to anchor at zero.  Defaults to ``5`` —
        ``Wd`` (defocus) is the *last* slot in rdmpy's coefficient ordering,
        not the first.  See ``optics/seidel_psf.py`` module docstring.
    seidel_model_dim : {None, 3, 4, 5, "trace3", "trace4", "trace5"}
        ``None`` preserves the classical backend optimizer, which is the
        current default path.  Trace-separated models are paused for default
        experiments and retained only for explicit reproduction; when used,
        they optimize public reduced coefficients by expansion before the
        unchanged RDM forward path. ``trace5`` hard-enforces only ``Wd=0`` and
        retains distortion through backend ``W311``; ``trace4``/``trace3``
        keep the existing ``W311=Wd=0`` semantics.
    scheduler : {"cosine", None}
        Learning-rate schedule.  ``"cosine"`` applies
        :class:`CosineAnnealingLR` with ``eta_min = lr_seidel * eta_min_ratio``
        to all parameter groups (matches CoCoA's demo).  ``None`` disables.
    eta_min_ratio : float
        Floor of the cosine schedule, as a fraction of ``lr_seidel``.
    pretrain_iter : int
        If positive, run :func:`pretrain_object` for this many steps before
        joint optimisation.  Defaults to 0 (skip pretraining).
    verbose : bool
        Print per-step progress every ``num_iter // 10`` steps.

    Returns
    -------
    result : TrainResult
    """
    if measurement_gt.ndim != 2 or measurement_gt.shape[0] != measurement_gt.shape[1]:
        raise ValueError(
            f"measurement_gt must be square 2-D, got {tuple(measurement_gt.shape)}"
        )

    H, W = measurement_gt.shape
    resolved_sys = build_sys_params(H, sys_params)

    if pretrain_iter > 0:
        pretrain_object(
            net_obj, measurement_gt, num_iter=pretrain_iter, verbose=verbose
        )

    seidel_optimizer = str(seidel_optimizer).lower()
    if seidel_optimizer not in {"adam", "sgd"}:
        raise ValueError(f"Unsupported seidel_optimizer={seidel_optimizer!r}")

    eta_min = lr_seidel * eta_min_ratio
    optimizers: list[torch.optim.Optimizer] = []
    lr_schedulers: list[CosineAnnealingLR] = []
    if seidel_optimizer == "adam":
        param_groups: list[dict] = [{"params": net_obj.parameters(), "lr": lr_obj}]
        if seidel_coeffs.requires_grad:
            param_groups.append({"params": [seidel_coeffs], "lr": lr_seidel})
        optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)
        optimizers.append(optimizer)
        if scheduler == "cosine":
            lr_schedulers.append(CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=eta_min))
    else:
        obj_optimizer = torch.optim.Adam(net_obj.parameters(), lr=lr_obj, betas=(0.9, 0.999), eps=1e-8)
        optimizers.append(obj_optimizer)
        if scheduler == "cosine":
            lr_schedulers.append(CosineAnnealingLR(obj_optimizer, T_max=num_iter, eta_min=eta_min))
        if seidel_coeffs.requires_grad:
            seidel_sgd = torch.optim.SGD([seidel_coeffs], lr=lr_seidel, momentum=0.0)
            optimizers.append(seidel_sgd)
            if scheduler == "cosine":
                lr_schedulers.append(CosineAnnealingLR(seidel_sgd, T_max=num_iter, eta_min=eta_min))

    loss_history: list[float] = []
    ssim_history: list[float] = []
    tv_history: list[float] = []
    anchor_history: list[float] = []

    sharp = torch.zeros_like(measurement_gt)
    measurement_pred = torch.zeros_like(measurement_gt)
    log_every = max(1, num_iter // 10)

    # When seidel_coeffs is frozen the PSF stack is bit-identical every step,
    # so compute it once and reuse via blur_ring_with_psfs. autograd still
    # flows through `sharp` because blur_ring_with_psfs is differentiable in
    # the image argument; the cached PSF is detached so no graph leaks.
    psfs_cached: torch.Tensor | None = None
    if not seidel_coeffs.requires_grad:
        if seidel_model_dim is None:
            psfs_cached = get_trainable_ring_psfs(
                seidel_coeffs, H, resolved_sys, device=measurement_gt.device,
            ).detach()
        else:
            psfs_cached = get_trainable_trace_ring_psfs(
                seidel_coeffs,
                H,
                resolved_sys,
                model_dim=seidel_model_dim,
                device=measurement_gt.device,
            ).detach()

    for step in range(num_iter):
        sharp = net_obj.render(H, W)
        if psfs_cached is not None:
            measurement_pred = blur_ring_with_psfs(sharp, psfs_cached)
        elif seidel_model_dim is not None:
            measurement_pred = blur_ring_trace_trainable(
                sharp,
                seidel_coeffs,
                resolved_sys,
                model_dim=seidel_model_dim,
            )
        else:
            measurement_pred = blur_ring_trainable(sharp, seidel_coeffs, resolved_sys)

        loss_ssim = ssim_loss(measurement_pred, measurement_gt)
        loss_tv = tv_2d(sharp)
        if seidel_model_dim is None:
            loss_anchor = single_mode_control(seidel_coeffs, defocus_index, 0.0, 0.0)
        else:
            loss_anchor = torch.zeros_like(loss_ssim)

        loss = (
            ssim_weight * loss_ssim
            + tv_weight * loss_tv
            + defocus_anchor_weight * loss_anchor
        )

        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

        loss_history.append(loss.item())
        ssim_history.append(loss_ssim.item())
        tv_history.append(loss_tv.item())
        anchor_history.append(loss_anchor.item())

        if verbose and step % log_every == 0:
            print(
                f"[train {step:04d}] total={loss.item():.4f} "
                f"ssim={loss_ssim.item():.4f} "
                f"tv={loss_tv.item():.4f} "
                f"anchor={loss_anchor.item():.4f}"
            )

    return TrainResult(
        sharp_final=sharp.detach(),
        seidel_final=seidel_coeffs.detach().clone(),
        measurement_pred=measurement_pred.detach(),
        loss_history=loss_history,
        ssim_history=ssim_history,
        tv_history=tv_history,
        anchor_history=anchor_history,
    )
