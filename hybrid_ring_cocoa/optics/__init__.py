"""Optics subpackage — Seidel PSF synthesis and ring-symmetric forward blur."""

from .ring_forward import (
    blur_ring,
    blur_ring_trainable,
    blur_ring_with_psfs,
)
from .seidel_psf import (
    build_sys_params,
    normalize_seidel_coeffs,
    validate_square_even_image,
)

__all__ = [
    "blur_ring",
    "blur_ring_trainable",
    "blur_ring_with_psfs",
    "build_sys_params",
    "normalize_seidel_coeffs",
    "validate_square_even_image",
]
