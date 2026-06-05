"""hybrid_ring_cocoa.object — coordinate-based neural object representation."""

from .coords import make_coord_grid_2d
from .encoding import radial_fourier_encoding
from .nerf import NeuralObject2D

__all__ = [
    "make_coord_grid_2d",
    "radial_fourier_encoding",
    "NeuralObject2D",
]
