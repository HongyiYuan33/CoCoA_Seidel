"""Image-quality metrics adapted from CoCoA.

Ports from ``CoCoA-master/misc/metrics.py`` and ``misc/utils.py``.  Two
adjustments relative to upstream:

* ``ssim`` and ``msssim`` are **2-D** here (``F.conv2d`` / ``F.avg_pool2d``).
  CoCoA was 3-D widefield; our problem is 2-D ring blur.
* The upstream module-level ``dtype = torch.cuda.FloatTensor`` and any
  ``.type(dtype)`` calls are removed so the metrics work on CPU and GPU
  identically — same convention as ``training/losses.py``.

Note: the original CoCoA repo will be deleted as part of the standalone
refactor; the line numbers in the docstrings refer to its pre-deletion state.
"""

from __future__ import annotations

from math import exp

import numpy as np
import torch
import torch.nn.functional as F


# ── Gaussian window helpers (CoCoA-master/misc/metrics.py:7-16) ─────────────


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int = 1) -> torch.Tensor:
    """Return a normalised 2-D Gaussian window for SSIM convolution."""
    _1d = gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


# ── Windowed SSIM (CoCoA-master/misc/metrics.py:27-78, conv3d → conv2d) ────


def _as_4d(img: torch.Tensor) -> torch.Tensor:
    """Accept (H, W) / (C, H, W) / (B, C, H, W) and return (B, C, H, W)."""
    if img.ndim == 2:
        return img.unsqueeze(0).unsqueeze(0)
    if img.ndim == 3:
        return img.unsqueeze(0)
    if img.ndim == 4:
        return img
    raise ValueError(f"Expected 2-D, 3-D, or 4-D image, got shape {tuple(img.shape)}")


def ssim_window(
    img1: torch.Tensor,
    img2: torch.Tensor,
    *,
    window_size: int = 11,
    window: torch.Tensor | None = None,
    size_average: bool = True,
    full: bool = False,
    val_range: float | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Windowed 2-D SSIM.  Returns a scalar in ``[-1, 1]`` when ``size_average``."""
    img1 = _as_4d(img1)
    img2 = _as_4d(img2)

    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        L = max_val - min_val
    else:
        L = val_range

    _, channel, h, w = img1.shape
    if window is None:
        real_size = min(window_size, h, w)
        window = create_window(real_size, channel=channel).to(
            device=img1.device, dtype=img1.dtype
        )

    mu1 = F.conv2d(img1, window, padding=0, groups=channel)
    mu2 = F.conv2d(img2, window, padding=0, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs_out = cs.mean()
        ret = ssim_map.mean()
    else:
        cs_out = cs.mean(dim=(1, 2, 3))
        ret = ssim_map.mean(dim=(1, 2, 3))

    if full:
        return ret, cs_out
    return ret


# ── Multi-scale SSIM (CoCoA-master/misc/metrics.py:81-115, avg_pool3d → 2d) ─


def msssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    *,
    window_size: int = 11,
    size_average: bool = True,
    val_range: float | None = None,
    normalize: str | None = None,
) -> torch.Tensor:
    """Multi-scale SSIM (5 levels, IWSSIM weights).  2-D adaptation."""
    img1 = _as_4d(img1)
    img2 = _as_4d(img2)

    weights = torch.tensor(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        device=img1.device,
        dtype=img1.dtype,
    )
    levels = weights.numel()
    ssims, mcs = [], []
    for _ in range(levels):
        sim, cs = ssim_window(
            img1,
            img2,
            window_size=window_size,
            size_average=size_average,
            full=True,
            val_range=val_range,
        )
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims_t = torch.stack(ssims)
    mcs_t = torch.stack(mcs)
    if normalize == "simple":
        ssims_t = (ssims_t + 1) / 2
        mcs_t = (mcs_t + 1) / 2

    pow1 = mcs_t**weights
    pow2 = ssims_t**weights
    return torch.prod(pow1[:-1]) * pow2[-1]


# ── Numpy scalar metrics (CoCoA-master/misc/utils.py:160-186) ───────────────


def compute_rms_contrast(inp: np.ndarray) -> float:
    return float(np.std(inp))


def compute_nrmse(target: np.ndarray, recon: np.ndarray) -> float:
    err = np.sum(np.square(target - recon))
    err /= np.sum(np.square(target))
    return float(np.sqrt(err))


def compute_snr(
    inp: np.ndarray,
    vmax: float | None = None,
    vmin: float | None = None,
    s0: float = 0,
    n_read: float = 1.6 / 0.46,
    normalized: bool = True,
) -> float:
    """Photon-limited SNR estimator from CoCoA's microscopy pipeline."""
    if normalized:
        inp = inp * (vmax - vmin) + vmin
    return float(np.mean(inp - s0) / np.sqrt(np.mean(inp - s0) + n_read**2))


def compute_mic_contrast(
    inp: np.ndarray,
    vmax: float | None = None,
    vmin: float | None = None,
    normalized: bool = True,
) -> float:
    """Michelson contrast ``(max - min) / (max + min)``."""
    if normalized:
        inp = inp * (vmax - vmin) + vmin
    return float((np.max(inp) - np.min(inp)) / (np.max(inp) + np.min(inp)))


__all__ = [
    "gaussian",
    "create_window",
    "ssim_window",
    "msssim",
    "compute_rms_contrast",
    "compute_nrmse",
    "compute_snr",
    "compute_mic_contrast",
]
