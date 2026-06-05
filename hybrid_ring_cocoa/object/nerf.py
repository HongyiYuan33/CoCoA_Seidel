"""2-D coordinate-based neural object representation.

CPU-portable port of ``CoCoA-master/misc/models.py:315-381`` (``class NeRF``),
specialised to ``out_channels = 1`` so that :meth:`NeuralObject2D.render`
returns a single-channel 2-D image ``(width, height)`` rather than a 3-D
volume ``(Z, width, height)``.

Default architecture follows the hyperparameters used in
``CoCoA-master/main/wf_cocoa_demo.ipynb`` (``--nerf_num_layers=6``,
``--nerf_num_filters=128``, ``--nerf_skips=[2,4,6]``), not the original
class defaults in ``models.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .coords import make_coord_grid_2d
from .encoding import radial_fourier_encoding


class NeuralObject2D(nn.Module):
    """Coordinate MLP that maps Fourier-encoded 2-D coords to a 2-D image.

    Parameters
    ----------
    in_features : int
        Width of the encoded input feature vector.  The default ``840``
        matches :func:`radial_fourier_encoding` at its own defaults
        (``num_angles=60, num_octaves=7``).
    out_channels : int
        Output channels per pixel.  Defaults to ``1`` — this is the main
        departure from CoCoA's 3-D ``out_channels = Z`` design.
    depth : int
        Number of hidden layers in the MLP trunk.  Default ``6``.
    width : int
        Hidden width of each trunk layer.  Default ``128``.
    skips : tuple of int
        Indices of trunk layers at which the original ``features`` are
        concatenated back into the hidden state.  Default ``(2, 4, 6)``.
    """

    def __init__(
        self,
        in_features: int = 840,
        out_channels: int = 1,
        depth: int = 6,
        width: int = 128,
        skips: tuple[int, ...] = (2, 4, 6),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_channels = out_channels
        self.depth = depth
        self.width = width
        self.skips = tuple(skips)

        # Trunk: a stack of Linear + ReLU layers, with skip concatenation at
        # the indices listed in ``skips``.  Stored as ``enc_1 … enc_{depth}``
        # to match the CoCoA attribute-naming convention.
        for i in range(depth):
            if i == 0:
                in_dim = in_features
            elif i in self.skips:
                in_dim = width + in_features
            else:
                in_dim = width
            layer = nn.Sequential(nn.Linear(in_dim, width), nn.ReLU(inplace=True))
            setattr(self, f"enc_{i + 1}", layer)

        self.enc_last = nn.Linear(width, width)

        # Output head: width → width//2 → out_channels (default 1).
        self.post = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(inplace=True),
            nn.Linear(width // 2, out_channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map encoded 2-D coordinates to per-pixel intensities.

        Parameters
        ----------
        features : (N, in_features) tensor

        Returns
        -------
        out : (N, out_channels) tensor
        """
        h = features
        for i in range(self.depth):
            if i in self.skips:
                h = torch.cat([features, h], dim=-1)
            h = getattr(self, f"enc_{i + 1}")(h)
        h = self.enc_last(h)
        return self.post(h)

    def render(self, width: int, height: int) -> torch.Tensor:
        """Produce a ``(width, height)`` 2-D image directly.

        Convenience helper that chains
        ``make_coord_grid_2d → radial_fourier_encoding → forward → reshape``.
        The intermediate tensors live on the same device / dtype as the
        module's parameters so that autograd stays consistent.

        Parameters
        ----------
        width, height : int
            Output image dimensions.

        Returns
        -------
        image : (width, height) tensor
            A single-channel 2-D image — **not** a ``(Z, W, H)`` volume.
        """
        reference = next(self.parameters())
        device = reference.device
        dtype = reference.dtype

        coords = make_coord_grid_2d(width, height, device=device, dtype=dtype)
        features = radial_fourier_encoding(coords)
        raw = self.forward(features)  # (W * H, out_channels)
        return raw.view(width, height) if self.out_channels == 1 else raw.view(
            width, height, self.out_channels
        )
