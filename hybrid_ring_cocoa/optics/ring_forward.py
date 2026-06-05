"""2-D ring blur: forward-only spatially varying convolution.

Provides three entry points at different abstraction levels:

* :func:`blur_ring` — public API using rdmpy's released path (frozen PSFs).
* :func:`blur_ring_trainable` — same physics, but keeps the autograd graph
  alive through the Seidel coefficients.
* :func:`blur_ring_with_psfs` — lowest level; applies pre-computed PSFs.
"""

from __future__ import annotations

from typing import Any

import torch

from .._rdm import ring_convolve
from .seidel_psf import (
    build_sys_params,
    expand_trace_seidel,
    get_reference_ring_psfs,
    get_trainable_ring_psfs,
    normalize_seidel_coeffs,
    validate_square_even_image,
)


def blur_ring_with_psfs(
    image_2d: Any,
    psf_data: Any,
    *,
    patch_size: int = 0,
) -> torch.Tensor:
    """Apply rdmpy's ring convolution given pre-computed PSF data.

    Parameters
    ----------
    image_2d : array-like, shape (N, N)
        Square, even-sided input image.
    psf_data : tensor or array
        RoFT-packed PSF stack (from :func:`get_reference_ring_psfs` or
        :func:`get_trainable_ring_psfs`).
    patch_size : int
        Isoplanatic annulus width.  ``0`` means ring-by-ring (full mode).

    Returns
    -------
    blurred : (N, N) tensor
    """
    image = validate_square_even_image(image_2d)

    if not torch.is_tensor(psf_data):
        psfs = torch.as_tensor(psf_data, dtype=image.dtype, device=image.device)
    else:
        psfs = psf_data.to(device=image.device)

    return ring_convolve(image, psfs, patch_size=patch_size, device=image.device)


def blur_ring(
    image_2d: Any,
    seidel_coeffs: Any,
    sys_params: dict[str, Any] | None = None,
    patch_size: int = 0,
) -> torch.Tensor:
    """Forward-only 2-D ring blur — reference path.

    This is the primary public API for Milestone A.  It synthesises PSFs from
    Seidel coefficients using rdmpy's released code, then applies the ring
    convolution.  The PSFs are *not* connected to the autograd graph.

    Parameters
    ----------
    image_2d : array-like, shape (N, N)
        Square, even-sided input image.
    seidel_coeffs : array-like, length 6
        ``[W040, W131, W222, W220, W311, Wd]`` =
        ``[spherical, coma, astigmatism, field_curvature, distortion, defocus]``.
        See ``optics/seidel_psf.py`` module docstring for the full story.
    sys_params : dict or None
        Optical system parameters (``lamb``, ``NA``, ``L``, ``samples``).
        Defaults are filled from rdmpy's conventions.
    patch_size : int
        Isoplanatic annulus width.  ``0`` = full ring-by-ring mode.

    Returns
    -------
    blurred : (N, N) tensor
    """
    image = validate_square_even_image(image_2d)
    coeffs = normalize_seidel_coeffs(seidel_coeffs, device=image.device)
    params = build_sys_params(image.shape[0], sys_params)
    psfs = get_reference_ring_psfs(
        coeffs,
        image.shape[0],
        params,
        patch_size=patch_size,
        device=image.device,
    )
    return blur_ring_with_psfs(image, psfs, patch_size=patch_size)


def blur_ring_trainable(
    image_2d: Any,
    seidel_coeffs: Any,
    sys_params: dict[str, Any] | None = None,
    patch_size: int = 0,
) -> torch.Tensor:
    """Forward ring blur with autograd-safe coefficient path.

    Same physics as :func:`blur_ring`, but the PSF generation preserves
    the computation graph so that gradients flow back through
    *seidel_coeffs*.  Use this when the coefficients are ``nn.Parameter``
    objects being optimised.

    Currently only ``patch_size=0`` is supported for this path.
    """
    image = validate_square_even_image(image_2d)
    coeffs = normalize_seidel_coeffs(seidel_coeffs, device=image.device)
    params = build_sys_params(image.shape[0], sys_params)
    psfs = get_trainable_ring_psfs(
        coeffs,
        image.shape[0],
        params,
        patch_size=patch_size,
        device=image.device,
    )
    return blur_ring_with_psfs(image, psfs, patch_size=patch_size)


def blur_ring_trace(
    image_2d: Any,
    theta_trace: Any,
    sys_params: dict[str, Any] | None = None,
    *,
    model_dim: int | str | None = None,
    patch_size: int = 0,
) -> torch.Tensor:
    """Forward-only ring blur from public trace-separated coefficients.

    Production path is only:
    ``theta_trace -> expand_trace_seidel -> existing blur_ring``.
    """
    image = validate_square_even_image(image_2d)
    theta_backend6 = expand_trace_seidel(
        theta_trace,
        model_dim=model_dim,
        device=image.device if torch.is_tensor(theta_trace) else None,
        dtype=image.dtype if torch.is_tensor(theta_trace) else torch.float32,
    )
    return blur_ring(image, theta_backend6, sys_params, patch_size=patch_size)


def blur_ring_trace_trainable(
    image_2d: Any,
    theta_trace: Any,
    sys_params: dict[str, Any] | None = None,
    *,
    model_dim: int | str | None = None,
    patch_size: int = 0,
) -> torch.Tensor:
    """Autograd-safe ring blur from public trace-separated coefficients.

    Production path is only:
    ``theta_trace -> expand_trace_seidel -> existing blur_ring_trainable``.
    """
    image = validate_square_even_image(image_2d)
    theta_backend6 = expand_trace_seidel(
        theta_trace,
        model_dim=model_dim,
        device=image.device if torch.is_tensor(theta_trace) else None,
        dtype=image.dtype if torch.is_tensor(theta_trace) else torch.float32,
    )
    return blur_ring_trainable(
        image,
        theta_backend6,
        sys_params,
        patch_size=patch_size,
    )
