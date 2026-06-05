"""Radial Fourier feature encoding for 2-D coordinates.

CPU-portable port of ``CoCoA-master/misc/models.py:123-144`` (``radial_encoding``).
Differences from upstream:

* No ``.cuda(0)`` — everything is allocated on the device of the input
  ``coords`` tensor.
* Parameter names are cleaned up: ``dia_digree`` → ``num_angles`` and
  ``L_xy`` → ``num_octaves``.  The math is unchanged.
* Default ``(num_angles=60, num_octaves=7)`` gives an output width of
  ``2 * 60 * 7 = 840`` features — the same value used by
  ``CoCoA-master/main/wf_cocoa_demo.ipynb``.
"""

from __future__ import annotations

import math

import torch


def radial_fourier_encoding(
    coords: torch.Tensor,
    num_angles: int = 60,
    num_octaves: int = 7,
) -> torch.Tensor:
    """Project 2-D coordinates onto multi-angle, multi-octave Fourier features.

    Parameters
    ----------
    coords : (N, 2) tensor
        Normalised 2-D pixel coordinates — typically the output of
        :func:`hybrid_ring_cocoa.object.coords.make_coord_grid_2d`.  Only the
        first two columns are used, so a wider tensor is also accepted.
    num_angles : int
        Number of angle directions uniformly covering :math:`[0, \pi)`.
    num_octaves : int
        Number of frequency octaves — each octave contributes one ``sin``
        and one ``cos`` channel per direction.

    Returns
    -------
    features : (N, 2 * num_angles * num_octaves) tensor
        Default output width is ``840``.  Dtype and device follow ``coords``.
    """
    if coords.ndim != 2 or coords.shape[-1] < 2:
        raise ValueError(
            f"Expected coords of shape (N, >=2), got {tuple(coords.shape)}"
        )
    if num_angles <= 0:
        raise ValueError(f"num_angles must be positive, got {num_angles}")
    if num_octaves <= 0:
        raise ValueError(f"num_octaves must be positive, got {num_octaves}")

    device = coords.device
    dtype = coords.dtype

    # 1. num_angles evenly-spaced directions on [0, π)
    step = 180.0 / num_angles
    angles_deg = torch.arange(0.0, 180.0, step, device=device, dtype=dtype)
    angles_rad = angles_deg * (math.pi / 180.0)

    # 2. fourier_mapping — stack sin/cos into a (2, num_angles) matrix
    fourier_mapping = torch.stack(
        [torch.sin(angles_rad), torch.cos(angles_rad)],
        dim=0,
    )  # (2, num_angles)

    # 3. project each 2-D coord onto every direction
    xy_freq = coords[:, :2] @ fourier_mapping  # (N, num_angles)

    # 4. multi-octave sin/cos expansion
    feats: list[torch.Tensor] = []
    for l in range(num_octaves):
        scale = (2.0 ** l) * math.pi
        feats.append(torch.sin(scale * xy_freq))
        feats.append(torch.cos(scale * xy_freq))

    # 5. concatenate → (N, 2 * num_angles * num_octaves)
    return torch.cat(feats, dim=-1)
