"""Ground-truth image loading and measurement synthesis for Milestone C.

We use ``hybrid_ring_cocoa/data/baboon.png`` as the sanity-check sharp image
(512×512 8-bit grayscale, vendored from rdmpy's test fixtures).  It is a
natural image with strong edges and fine texture — easier than a sparse
microscopy target for the purpose of demonstrating that the forward chain
trains at all.  Real microscopy validation is deferred to Milestone D.

The measurement is synthesised by calling :func:`blur_ring` (non-trainable
forward path) with a **known** ground-truth Seidel vector, under
``torch.no_grad()``.  Callers then use this measurement as the supervision
target when optimising a freshly-initialised NeRF + Seidel pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage import img_as_float
from skimage.io import imread
from skimage.transform import resize

from ..optics.ring_forward import blur_ring, blur_ring_trace

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_BABOON_PATH = _PACKAGE_DIR / "data" / "baboon.png"


def load_baboon_gt(
    image_size: int = 128,
    *,
    path: str | Path | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load and preprocess the baboon test image for ring-blur experiments.

    Pipeline: read → grayscale → center-crop to square → resize with
    anti-aliasing to ``image_size × image_size`` → normalise to ``[0, 1]``.

    Parameters
    ----------
    image_size : int
        Output side length.  Must be positive and even (required by
        :func:`hybrid_ring_cocoa.blur_ring`).
    path : str or Path, optional
        Override for the baboon path.  Defaults to
        ``hybrid_ring_cocoa/data/baboon.png`` (shipped as package data).
    device, dtype
        Target device and dtype for the returned tensor.  When ``device``
        is ``None`` (default), uses ``cuda:0`` if CUDA is available, else
        ``cpu`` — the standard PyTorch idiom, so a CPU-only machine still
        works.

    Returns
    -------
    image : (image_size, image_size) tensor in ``[0, 1]``
    """
    if image_size <= 0 or image_size % 2 != 0:
        raise ValueError(
            f"image_size must be positive and even, got {image_size}"
        )

    resolved = Path(path) if path is not None else _DEFAULT_BABOON_PATH
    if not resolved.is_file():
        raise FileNotFoundError(f"baboon image not found at {resolved}")

    # ``imread(as_gray=True)`` returns float in the image's native integer
    # range for already-grayscale PNGs (e.g. uint8 → [0, 255]).  Only RGB(A)
    # → grayscale conversion normalises implicitly.  ``img_as_float`` forces
    # [0, 1] regardless of input dtype/shape.
    raw = img_as_float(imread(str(resolved), as_gray=True))
    raw = np.asarray(raw, dtype=np.float64)

    h, w = raw.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = raw[top : top + side, left : left + side]

    resized = resize(
        cropped,
        (image_size, image_size),
        anti_aliasing=True,
        preserve_range=True,
    )
    resized = np.clip(resized, 0.0, 1.0).astype(np.float64)

    if device is None:
        device = (
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    return torch.as_tensor(resized, dtype=dtype, device=device)


def synthesize_measurement(
    sharp: torch.Tensor,
    seidel_coeffs_gt: Any,
    sys_params: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Synthesise a blurred measurement from a known sharp image and coeffs.

    Runs :func:`hybrid_ring_cocoa.blur_ring` inside ``torch.no_grad`` — the
    returned tensor is detached, which is what we want for a supervision
    target.

    Parameters
    ----------
    sharp : (N, N) tensor
        Ground-truth sharp image.
    seidel_coeffs_gt : array-like, length 6
        Ground-truth Seidel coefficients
        ``[W040, W131, W222, W220, W311, Wd]`` =
        ``[spherical, coma, astigmatism, field_curvature, distortion, defocus]``.
    sys_params : dict, optional
        Optical system parameters; forwarded to :func:`blur_ring`.

    Returns
    -------
    measurement : (N, N) tensor
        Detached, same device/dtype as ``sharp``.
    """
    with torch.no_grad():
        measurement = blur_ring(sharp, seidel_coeffs_gt, sys_params)
    return measurement.detach()


def synthesize_trace_measurement(
    sharp: torch.Tensor,
    theta_trace_gt: Any,
    sys_params: dict[str, Any] | None = None,
    *,
    model_dim: int | str | None = None,
) -> torch.Tensor:
    """Synthesize a measurement from public trace-separated coefficients.

    This is a thin wrapper around the frozen RDM forward path:
    ``theta_trace -> expand_trace_seidel -> blur_ring``.
    """
    with torch.no_grad():
        measurement = blur_ring_trace(
            sharp,
            theta_trace_gt,
            sys_params,
            model_dim=model_dim,
        )
    return measurement.detach()
