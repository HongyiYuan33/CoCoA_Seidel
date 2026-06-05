"""2-D coordinate grid for the neural object representation.

CPU-portable port of ``CoCoA-master/misc/models.py:107-112`` (``input_coord_2d``).
Differences from upstream:

* No ``einops`` dependency — uses :meth:`torch.Tensor.reshape` instead of
  ``rearrange``.
* No hardcoded ``.cuda(0)``; the grid lives on whatever device the caller
  passes. When ``device`` is ``None``, defaults to ``cuda:0`` if
  ``torch.cuda.is_available()`` else ``cpu`` — the standard PyTorch idiom,
  so the code still runs on a CPU-only machine.
"""

from __future__ import annotations

import torch


def make_coord_grid_2d(
    width: int,
    height: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a normalised 2-D pixel-coordinate grid in :math:`[-1, 1]^2`.

    Parameters
    ----------
    width : int
        Number of rows in the grid (outer axis).
    height : int
        Number of columns in the grid (inner axis).
    device : torch.device or str, optional
        Device on which to allocate the grid.  When ``None`` (default),
        uses ``cuda:0`` if CUDA is available, else ``cpu``.
    dtype : torch.dtype
        Floating-point dtype for the grid.  Defaults to ``torch.float32``.

    Returns
    -------
    coords : (width * height, 2) tensor
        Each row is ``(y, x)`` in ``[-1, 1]``.  Order matches
        ``torch.meshgrid(..., indexing="ij")`` flattened row-major.
    """
    if width <= 0 or height <= 0:
        raise ValueError(
            f"width and height must be positive, got ({width}, {height})"
        )

    if device is None:
        device = (
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    ys = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    my, mx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([my, mx], dim=-1)  # (width, height, 2)
    return coords.reshape(width * height, 2)
