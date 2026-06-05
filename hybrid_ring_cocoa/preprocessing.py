"""Preprocessing utilities adapted from CoCoA.

Image registration and signal-to-background-ratio (SBR) estimation, ported
from ``CoCoA-master/misc/utils.py``.  Adjustments versus upstream:

* Each function comes in a 3-D *and* a 2-D variant — CoCoA was widefield
  microscopy (volumes); we are 2-D ring blur.  When the user has a single
  plane, the 2-D variants skip the maximum/mean intensity projection.
* ``scikit-learn`` is imported lazily inside
  :func:`signal_to_background_ratio_gaussian_mixture` so the package still
  imports (and the rest of the API still works) on environments without it.
* The ``return loss.type(dtype)`` CUDA pattern doesn't apply here — these are
  numpy-only functions.

Note: the original CoCoA repo will be deleted as part of the standalone
refactor; the line numbers in the docstrings refer to its pre-deletion state.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation


# ── Image registration (CoCoA-master/misc/utils.py:81-106) ─────────────────


def find_translation_and_fix(
    ref: np.ndarray,
    moving: np.ndarray,
    mode: str = "mean",
) -> tuple[np.ndarray, tuple, tuple, tuple]:
    """Register a 3-D moving volume to a 3-D reference along all three axes.

    Byte-copy of ``CoCoA-master/misc/utils.py:81-106``.  Pipeline: project
    along z (XY shift) → apply → project along y (XZ) → apply → project along
    x (YZ) → apply.  For a 2-D input use :func:`find_translation_and_fix_2d`.

    Parameters
    ----------
    ref, moving : (D, H, W) numpy arrays
    mode : {'mean', 'max'}
        Projection used for each axis pair.

    Returns
    -------
    moving_mv : registered volume, same shape as input.
    shift_xy, shift_xz, shift_yz : raw outputs from
        ``skimage.registration.phase_cross_correlation`` (subpixel shifts +
        error + diffphase).
    """
    if mode == "max":
        shift_xy = phase_cross_correlation(
            ref.max(0), moving.max(0), upsample_factor=100, normalization=None
        )
    else:
        shift_xy = phase_cross_correlation(
            ref.mean(0), moving.mean(0), upsample_factor=100, normalization=None
        )
    moving_mv = fourier_shift(
        np.fft.fftn(moving), (0, shift_xy[0][0], shift_xy[0][1])
    )
    moving_mv = np.fft.ifftn(moving_mv).real

    if mode == "max":
        shift_xz = phase_cross_correlation(
            ref.max(1), moving_mv.max(1), upsample_factor=100, normalization=None
        )
    else:
        shift_xz = phase_cross_correlation(
            ref.mean(1), moving_mv.mean(1), upsample_factor=100, normalization=None
        )
    moving_mv = fourier_shift(
        np.fft.fftn(moving_mv), (shift_xz[0][0], 0, shift_xz[0][1])
    )
    moving_mv = np.fft.ifftn(moving_mv).real

    if mode == "max":
        shift_yz = phase_cross_correlation(
            ref.max(2), moving_mv.max(2), upsample_factor=100, normalization=None
        )
    else:
        shift_yz = phase_cross_correlation(
            ref.mean(2), moving_mv.mean(2), upsample_factor=100, normalization=None
        )
    moving_mv = fourier_shift(
        np.fft.fftn(moving_mv), (shift_yz[0][0], shift_yz[0][1], 0)
    )
    moving_mv = np.fft.ifftn(moving_mv).real

    return moving_mv, shift_xy, shift_xz, shift_yz


def find_translation_and_fix_2d(
    ref: np.ndarray, moving: np.ndarray
) -> tuple[np.ndarray, tuple]:
    """Register a 2-D moving image to a 2-D reference (subpixel).

    Equivalent to one ``phase_cross_correlation`` pass on the raw image (no
    MIP, since there is no third axis).  Returns the registered image and the
    raw ``phase_cross_correlation`` output.
    """
    shift = phase_cross_correlation(
        ref, moving, upsample_factor=100, normalization=None
    )
    moving_mv = fourier_shift(np.fft.fftn(moving), shift[0])
    moving_mv = np.fft.ifftn(moving_mv).real
    return moving_mv, shift


# ── Signal-to-background ratio (CoCoA-master/misc/utils.py:231-265) ────────


def signal_to_background_ratio(
    tar: np.ndarray,
    M: int = 20,
    N: int = 20,
    return_imgs: bool = False,
    use_mean: bool = True,
) -> float | tuple:
    """Tile-based SBR estimator.

    Adapted from ``CoCoA-master/misc/utils.py:231-265``.  For a 3-D ``tar``
    the original behaviour is preserved (we project along axis 0 with
    ``np.max``).  For a 2-D image we operate directly on the input.

    Foreground tiles are those whose intra-tile std exceeds the mean (or
    median) over all tiles.  SBR = ``mean(signal pixels) / mean(background
    pixels)``.
    """
    im = np.max(tar, 0) if tar.ndim == 3 else tar
    xx = np.arange(0, im.shape[0])
    yy = np.arange(0, im.shape[1])
    X, Y = np.meshgrid(xx, yy)

    tiles = [
        np.std(im[x : x + M, y : y + N])
        for x in range(0, im.shape[0], M)
        for y in range(0, im.shape[1], N)
    ]
    tiles_X = [
        X[x : x + M, y : y + N]
        for x in range(0, im.shape[0], M)
        for y in range(0, im.shape[1], N)
    ]
    tiles_Y = [
        Y[x : x + M, y : y + N]
        for x in range(0, im.shape[0], M)
        for y in range(0, im.shape[1], N)
    ]

    tiles = np.stack(tiles)
    tiles_X = np.stack(tiles_X)
    tiles_Y = np.stack(tiles_Y)

    threshold = np.mean(tiles) if use_mean else np.median(tiles)
    tiles_X_s = tiles_X[tiles > threshold]
    tiles_Y_s = tiles_Y[tiles > threshold]

    im_b = np.copy(im)
    for j in range(tiles_X_s.shape[0]):
        im_b[
            tiles_Y_s[j, 0, 0] : tiles_Y_s[j, -1, 0] + 1,
            tiles_X_s[j, 0, 0] : tiles_X_s[j, 0, -1] + 1,
        ] = 0

    im_s = im - im_b
    sbr = float(np.mean(im_s[im_s > 0]) / np.mean(im_b[im_b > 0]))

    if return_imgs:
        return sbr, im_b, im_s
    return sbr


def signal_to_background_ratio_gaussian_mixture(
    tar: np.ndarray,
    l: int = 400,
    n_lp: int = 10,
    n_tone: int = 200,
    kernel_size: int | None = 3,
    return_imgs: bool = False,
) -> float | tuple:
    """SBR via 2-component Gaussian mixture on a high-pass-filtered image.

    Adapted from ``CoCoA-master/misc/utils.py:268-294``.  Like
    :func:`signal_to_background_ratio` the 3-D input gets a leading-axis MIP;
    a 2-D input is used directly.

    ``scikit-learn`` is imported lazily so the rest of ``hybrid_ring_cocoa``
    still works without it; a clear ``ImportError`` is raised here only.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError(
            "signal_to_background_ratio_gaussian_mixture requires scikit-learn. "
            "Install it via `pip install scikit-learn` (or rerun "
            "`scripts/remote_setup.sh` on the server)."
        ) from exc

    im = np.max(tar, 0) if tar.ndim == 3 else tar

    im_lp = ndimage.gaussian_filter(im, sigma=l / (4.0 * n_lp))
    im_corr = im - im_lp

    if kernel_size is not None:
        im_corr = scipy.signal.medfilt(im_corr, kernel_size)
    else:
        im_corr = ndimage.gaussian_filter(im_corr, sigma=l / (4.0 * n_tone))

    classif = GaussianMixture(n_components=2)
    classif.fit(im_corr.reshape((im_corr.size, 1)))
    threshold = np.mean(classif.means_)

    im_b = np.copy(im)
    im_b[im_corr > threshold] = 0
    im_s = im - im_b

    sbr = float(np.mean(im_s[im_s > 0]) / np.mean(im_b[im_b > 0]))

    if return_imgs:
        return sbr, im_b, im_s, im_corr
    return sbr


__all__ = [
    "find_translation_and_fix",
    "find_translation_and_fix_2d",
    "signal_to_background_ratio",
    "signal_to_background_ratio_gaussian_mixture",
]
