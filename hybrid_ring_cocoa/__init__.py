"""hybrid_ring_cocoa — 2-D ring blur combining CoCoA patterns with rdmpy optics."""

from .metrics import (
    compute_mic_contrast,
    compute_nrmse,
    compute_rms_contrast,
    compute_snr,
    create_window,
    msssim,
    ssim_window,
)
from .preprocessing import (
    find_translation_and_fix,
    find_translation_and_fix_2d,
    signal_to_background_ratio,
    signal_to_background_ratio_gaussian_mixture,
)
from .object.coords import make_coord_grid_2d
from .object.encoding import radial_fourier_encoding
from .object.nerf import NeuralObject2D
from .optics.ring_forward import (
    blur_ring,
    blur_ring_trainable,
    blur_ring_with_psfs,
)
from .optics.seidel_psf import (
    build_sys_params,
    normalize_seidel_coeffs,
    validate_square_even_image,
)
from .training.data import load_baboon_gt, synthesize_measurement
from .training.losses import (
    fourier_loss,
    nonlinear_diffusion_loss_2d,
    npcc_loss,
    second_order_diff_loss_2d,
    single_mode_control,
    ssim_loss,
    tv_2d,
    tv_range_loss_2d,
)
from .training.train import TrainResult, pretrain_object, train

__all__ = [
    "blur_ring",
    "blur_ring_trainable",
    "blur_ring_with_psfs",
    "build_sys_params",
    "normalize_seidel_coeffs",
    "validate_square_even_image",
    "NeuralObject2D",
    "make_coord_grid_2d",
    "radial_fourier_encoding",
    "load_baboon_gt",
    "synthesize_measurement",
    "ssim_loss",
    "tv_2d",
    "single_mode_control",
    "train",
    "pretrain_object",
    "TrainResult",
    # New in standalone refactor: CoCoA-derived metrics
    "ssim_window",
    "msssim",
    "create_window",
    "compute_rms_contrast",
    "compute_nrmse",
    "compute_snr",
    "compute_mic_contrast",
    # New in standalone refactor: CoCoA-derived regularisers
    "npcc_loss",
    "nonlinear_diffusion_loss_2d",
    "tv_range_loss_2d",
    "second_order_diff_loss_2d",
    "fourier_loss",
    # New in standalone refactor: CoCoA-derived preprocessing
    "find_translation_and_fix",
    "find_translation_and_fix_2d",
    "signal_to_background_ratio",
    "signal_to_background_ratio_gaussian_mixture",
]
