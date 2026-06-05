"""Loss and regulariser primitives for Milestone C joint training.

CPU-portable port of the CoCoA originals:

* :func:`ssim_loss`  — ``CoCoA-master/misc/losses.py:59-76``
* :func:`tv_2d`      — ``CoCoA-master/misc/losses.py:27-32`` (native 2-D)
* :func:`single_mode_control` — ``CoCoA-master/misc/utils.py:29-30``

The two deliberate departures from upstream are:

1. **No ``torch.cuda.FloatTensor`` coercion.**  CoCoA's originals end in
   ``return loss.type(dtype)`` where ``dtype = torch.cuda.FloatTensor`` is
   a module-level constant.  That makes the code CUDA-only.  Here we return
   the loss tensor as-is so it inherits the caller's device/dtype, which is
   what lets the same code run on CPU, on waller's GPU 0, or in a notebook
   reviewer's laptop without editing the source.

2. **``tv_2d`` takes a 2-D ``(H, W)`` image directly**, rolling on
   ``dims=(0, 1)``.  CoCoA's version assumed ``D × W × H`` and rolled on
   ``dims=(1, 2)``; we are 2-D throughout this milestone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Global-mean SSIM loss between two images, in ``[0, 1]``.

    This is the scalar-statistic SSIM used by CoCoA, not the 11×11 windowed
    SSIM from the original Wang et al. paper.  It is cheap and differentiable,
    but has a trivial optimum at any constant image whose global mean and
    variance match the target.  Rely on regularisers (e.g. :func:`tv_2d`) and
    the defocus anchor to break that degeneracy.

    # TODO(milestone-D): swap in a windowed SSIM (e.g. ``kornia.losses.SSIMLoss``)
    # once we move to real microscopy data, to avoid the global-mean collapse
    # mode on smooth or low-contrast targets.

    Parameters
    ----------
    img1, img2 : tensor
        Same-shape tensors.  Any dimensionality is accepted; all elements
        contribute to the global statistics.

    Returns
    -------
    loss : scalar tensor
        ``1 - SSIM(img1, img2)``.  Zero when the images are identical.
    """
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    mu1_mu2 = mu1 * mu2

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = torch.mean(img1 * img1) - mu1_sq
    sigma2_sq = torch.mean(img2 * img2) - mu2_sq
    sigma12 = torch.mean(img1 * img2) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = torch.clamp(
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2),
        min=1e-8,
        max=1e8,
    )
    return 1.0 - numerator / denominator


def tv_2d(img: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation of a 2-D image, sum-reduced.

    Computes ``sum(|img - roll(img, 1, dims=0)|) + sum(|img - roll(img, 1, dims=1)|)``.
    Zero on a constant image; strictly positive otherwise.

    Parameters
    ----------
    img : (H, W) tensor

    Returns
    -------
    tv : scalar tensor
    """
    if img.ndim != 2:
        raise ValueError(f"tv_2d expects a 2-D image, got shape {tuple(img.shape)}")
    h_variance = torch.sum(torch.abs(img - torch.roll(img, 1, dims=0)))
    w_variance = torch.sum(torch.abs(img - torch.roll(img, 1, dims=1)))
    return h_variance + w_variance


def single_mode_control(
    coeffs: torch.Tensor,
    num: int,
    vmin: float = 0.0,
    vmax: float = 0.0,
) -> torch.Tensor:
    """Hinge penalty forcing ``coeffs[num]`` into ``[vmin, vmax]``.

    The penalty is ``ReLU(coeffs[num] - vmax) + ReLU(-coeffs[num] + vmin)``.
    With the default ``vmin = vmax = 0`` it reduces to ``|coeffs[num]|`` and
    anchors that coefficient at zero — our use case for the defocus term
    (``num = 0`` in the Seidel convention, see
    ``hybrid_ring_cocoa/optics/seidel_psf.py``).

    Parameters
    ----------
    coeffs : 1-D tensor
        Coefficient vector.  Must be indexable with ``[num]``.
    num : int
        Index of the coefficient to constrain.
    vmin, vmax : float
        Allowed range.  Defaults to the zero-anchor case ``(0, 0)``.

    Returns
    -------
    penalty : scalar tensor
    """
    return F.relu(coeffs[num] - vmax) + F.relu(-coeffs[num] + vmin)


# ── Additional CoCoA-derived regularisers, ported in the standalone refactor.
# Source line numbers refer to the upstream ``CoCoA-master/misc/losses.py``
# before that repo was deleted.  Each one drops the upstream
# ``return loss.type(dtype)`` (where ``dtype = torch.cuda.FloatTensor``) for the
# CPU/GPU portability reasons documented at the top of this file.


def npcc_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """1 - normalised Pearson cross-correlation, in ``[0, 2]``.

    Adapted from ``CoCoA-master/misc/losses.py:51-56``.  Dimensionality-agnostic.
    Zero when the two tensors are perfectly correlated.
    """
    up = torch.mean(
        (y_pred - torch.mean(y_pred)) * (y_true - torch.mean(y_true))
    )
    down = torch.std(y_pred) * torch.std(y_true)
    return 1.0 - up / down


def nonlinear_diffusion_loss_2d(
    img: torch.Tensor, upper_bound: float
) -> torch.Tensor:
    """Bounded total-variation prior for 2-D images.

    Adapted from ``CoCoA-master/misc/losses.py:79-89`` (3-D → 2-D).  Penalises
    pixel-pair differences but caps each contribution at ``upper_bound``, so
    sharp edges (large jumps) cost the same as a moderate one — favours
    piecewise-smooth solutions without blurring true edges.
    """
    h_abs_grad = torch.abs(img[:-1, :] - img[1:, :])
    w_abs_grad = torch.abs(img[:, :-1] - img[:, 1:])

    def _bounded(grad: torch.Tensor, ubd: float) -> torch.Tensor:
        return torch.minimum(grad, torch.full_like(grad, ubd))

    return torch.sum(_bounded(h_abs_grad, upper_bound)) + torch.sum(
        _bounded(w_abs_grad, upper_bound)
    )


def tv_range_loss_2d(img: torch.Tensor, lower_bound: float) -> torch.Tensor:
    """TV prior with a *lower* clamp.

    Adapted from ``CoCoA-master/misc/losses.py:92-102`` (3-D → 2-D).  Each
    pixel-pair difference is clamped to be **at least** ``lower_bound``, so
    perfectly flat regions still pay a baseline cost.  Used in CoCoA to prevent
    collapse to a constant image during early joint training.
    """
    h_abs_grad = torch.abs(img[:-1, :] - img[1:, :])
    w_abs_grad = torch.abs(img[:, :-1] - img[:, 1:])

    def _floored(grad: torch.Tensor, lbd: float) -> torch.Tensor:
        return torch.maximum(grad, torch.full_like(grad, lbd))

    return torch.sum(_floored(h_abs_grad, lower_bound)) + torch.sum(
        _floored(w_abs_grad, lower_bound)
    )


def second_order_diff_loss_2d(
    img: torch.Tensor, upper_bound: float
) -> torch.Tensor:
    """Bounded curvature prior for 2-D images.

    Adapted from ``CoCoA-master/misc/losses.py:105-113`` (3-D → 2-D).  Operates
    on the 2-D mixed second difference
    ``|img[:-1, :-1] - img[:-1, 1:] - img[1:, :-1] + img[1:, 1:]|``, capped at
    ``upper_bound``.  Smooths out high-frequency oscillations while preserving
    edges (which contribute a saturated cost).
    """
    diff = torch.abs(
        img[:-1, :-1] - img[:-1, 1:] - img[1:, :-1] + img[1:, 1:]
    )
    return torch.sum(torch.minimum(diff, torch.full_like(diff, upper_bound)))


def fourier_loss(F1: torch.Tensor, F2: torch.Tensor) -> torch.Tensor:
    """Mean cosine distance between two complex Fourier spectra.

    Byte-copy from ``CoCoA-master/misc/losses.py:124-127``.  Inputs are
    expected to be complex tensors (e.g. outputs of ``torch.fft.*``).  Returns
    ``1 - mean(|F1 ⋅ F2*| / (|F1| ⋅ |F2|))`` — zero when the two spectra are
    aligned in phase.
    """
    projection = torch.abs(F1 * torch.conj(F2)) / torch.abs(F1) / torch.abs(F2)
    return 1.0 - projection.mean()
