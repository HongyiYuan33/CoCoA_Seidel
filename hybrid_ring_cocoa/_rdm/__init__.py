"""Vendored subset of rdmpy (formerly the sibling ``rdmpy-main/`` checkout).

Only the three entry points actually used by ``hybrid_ring_cocoa`` are re-exported;
everything else (deblur, dl_models, calibrate_rdm/sdm, opt, kornia) was dropped.
The leading underscore signals that this is private — the public API for ring
blur lives in ``hybrid_ring_cocoa.optics``.
"""

from .blur import ring_convolve
from .calibrate import get_rdm_psfs
from ._src.psf_model import compute_rdm_psfs

__all__ = ["ring_convolve", "get_rdm_psfs", "compute_rdm_psfs"]
