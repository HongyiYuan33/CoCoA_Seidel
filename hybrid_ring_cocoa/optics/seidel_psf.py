"""Seidel coefficient handling and PSF generation for ring deconvolution.

Wraps rdmpy's PSF synthesis into validated, autograd-friendly helpers.

Coefficient order (matching the actual code in
``rdmpy/_src/psf_model.py:61-68`` — defocus is **last**, not first)::

    [W040, W131, W222, W220, W311, Wd]
    = [spherical, coma, astigmatism, field_curvature, distortion, defocus]

The comment at ``rdmpy/_src/psf_model.py:21`` claims the opposite order
(``Wd`` first); that comment is wrong.  The authoritative source is the
``compute_pupil_phase`` body, corroborated by ``rdmpy/deblur.py:334`` and
``rdmpy/calibrate.py:232``.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch

import torch.nn.functional as F

from .._rdm import get_rdm_psfs as _get_rdm_psfs
from .._rdm._src.psf_model import (
    circ as _circ,
    compute_pupil_phase as _compute_pupil_phase,
    compute_rdm_psfs as _compute_rdm_psfs,
)
from .._rdm._src import polar_transform as _polar_transform
from .._rdm._src import util as _rdm_util

# ── physical defaults (matching rdmpy) ──────────────────────────────────────

DEFAULT_LAMBDA: float = 0.55e-6  # 550 nm green light
DEFAULT_NA: float = 0.5
NUM_SEIDEL: int = 6
# Trace-separated names/helpers are paused for default experiments and kept
# only for explicit reproduction of earlier trace3/trace4/trace5 runs.
TRACE5_COEFF_NAMES: tuple[str, ...] = ("S", "C", "A", "F", "D")
TRACE4_COEFF_NAMES: tuple[str, ...] = ("S", "C", "A", "F")
TRACE3_COEFF_NAMES: tuple[str, ...] = ("S", "C", "A")


# ── input validation ────────────────────────────────────────────────────────

def validate_square_even_image(image_2d: Any) -> torch.Tensor:
    """Coerce *image_2d* to a float tensor and enforce rdmpy's constraints.

    Requirements: 2-D, square, even side-length.
    """
    if not torch.is_tensor(image_2d):
        image = torch.as_tensor(image_2d, dtype=torch.float32)
    else:
        image = image_2d if torch.is_floating_point(image_2d) else image_2d.float()

    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {tuple(image.shape)}")
    if image.shape[0] != image.shape[1]:
        raise ValueError(f"Expected a square image, got shape {tuple(image.shape)}")
    if image.shape[0] % 2 != 0:
        raise ValueError(
            f"Expected even side-length, got {image.shape[0]}"
        )
    return image


def normalize_seidel_coeffs(
    seidel_coeffs: Any,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Flatten and validate a 6-element Seidel coefficient vector.

    Accepts tensors, numpy arrays, or plain lists.  Returns a 1-D tensor of
    shape ``(6,)`` on the requested *device* / *dtype*.

    Coefficient order (see module docstring for the full story)::

        [W040, W131, W222, W220, W311, Wd]
        = [spherical, coma, astigmatism, field_curvature, distortion, defocus]
    """
    if torch.is_tensor(seidel_coeffs):
        coeffs = seidel_coeffs.reshape(-1)
        target_device = device or coeffs.device
        target_dtype = dtype or coeffs.dtype
        coeffs = coeffs.to(device=target_device, dtype=target_dtype)
    else:
        coeffs = torch.as_tensor(
            seidel_coeffs, dtype=dtype, device=device
        ).reshape(-1)

    if coeffs.numel() != NUM_SEIDEL:
        raise ValueError(
            f"Expected {NUM_SEIDEL} Seidel coefficients "
            "[W040, W131, W222, W220, W311, Wd] "
            "(spherical, coma, astigmatism, field_curvature, distortion, defocus), "
            f"got {coeffs.numel()}"
        )
    return coeffs


def _normalize_trace_model_dim(model_dim: int | str | None, length: int | None = None) -> int:
    if isinstance(model_dim, str):
        aliases = {
            "trace5": 5,
            "5d": 5,
            "trace5_no_defocus": 5,
            "no_defocus_5d": 5,
            "trace4": 4,
            "4d": 4,
            "no_distortion_no_defocus_4d": 4,
            "trace3": 3,
            "3d": 3,
            "no_field_curvature_3d": 3,
            "strict_no_scalar_quadratic": 3,
        }
        key = model_dim.strip().lower()
        if key not in aliases:
            raise ValueError(f"Unknown trace Seidel model_dim={model_dim!r}")
        model_dim = aliases[key]
    if model_dim is None:
        if length not in (3, 4, 5):
            raise ValueError(
                "model_dim must be 3, 4, or 5 when theta_trace length is not available"
            )
        return int(length)
    if int(model_dim) not in (3, 4, 5):
        raise ValueError(f"Trace Seidel model_dim must be 3, 4, or 5, got {model_dim}")
    return int(model_dim)


def expand_trace_seidel(
    theta_trace: Any,
    model_dim: int | str | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = torch.float32,
) -> torch.Tensor | np.ndarray:
    """Expand public trace-separated Seidel coefficients to backend 6D order.

    This is a parameterization layer only.  It does not change the RDM forward
    model, PSF synthesis, phase convention, RoFT packing, or ring convolution.
    Trace-separated conventions are currently paused for default experiments
    and retained as explicit reproduction helpers.

    Public trace conventions:

    * ``theta_trace5 = [S, C, A, F, D]`` maps to
      ``[S, C, 2A, F-A, D, 0]``. This is the no-arbitrary-defocus
      convention that retains distortion in backend ``W311``.
    * ``theta_trace4 = [S, C, A, F]`` maps to
      ``[S, C, 2A, F-A, 0, 0]``.
    * ``theta_trace3 = [S, C, A]`` has public ``F=0`` and maps to
      ``[S, C, 2A, -A, 0, 0]``.

    The backend order is always
    ``[W040, W131, W222, W220, W311, Wd]`` with defocus last.
    """
    if torch.is_tensor(theta_trace):
        trace = theta_trace.reshape(-1)
        dim = _normalize_trace_model_dim(model_dim, int(trace.numel()))
        if trace.numel() != dim:
            raise ValueError(
                f"Expected theta_trace{dim} to have {dim} values, got {trace.numel()}"
            )
        target_device = device or trace.device
        target_dtype = dtype or trace.dtype
        trace = trace.to(device=target_device, dtype=target_dtype)
        zero = torch.zeros((), dtype=trace.dtype, device=trace.device)
        s, c, a = trace[0], trace[1], trace[2]
        f = trace[3] if dim in (4, 5) else zero
        d = trace[4] if dim == 5 else zero
        return torch.stack((s, c, 2.0 * a, f - a, d, zero))

    arr = np.asarray(theta_trace, dtype=np.float64).reshape(-1)
    dim = _normalize_trace_model_dim(model_dim, int(arr.size))
    if arr.size != dim:
        raise ValueError(f"Expected theta_trace{dim} to have {dim} values, got {arr.size}")
    s, c, a = arr[0], arr[1], arr[2]
    f = arr[3] if dim in (4, 5) else 0.0
    d = arr[4] if dim == 5 else 0.0
    return np.asarray([s, c, 2.0 * a, f - a, d, 0.0], dtype=np.float64)


def compress_trace_seidel(
    theta_backend6: Any,
    model_dim: int | str = 4,
    *,
    atol: float = 1e-8,
) -> torch.Tensor | np.ndarray:
    """Compress backend 6D coefficients into public trace-separated form.

    Compression is intentionally strict for the existing no-distortion trace4
    and trace3 models: distortion and arbitrary defocus must already be absent
    within ``atol``. For the trace5 no-defocus model, only arbitrary defocus
    must be absent; backend ``W311`` is returned as public ``D``. For the 3D
    model, public field curvature ``F = W220 + 0.5 * W222`` must also be
    absent within ``atol``.
    """
    dim = _normalize_trace_model_dim(model_dim)
    if torch.is_tensor(theta_backend6):
        backend = theta_backend6.reshape(-1)
        if backend.numel() != NUM_SEIDEL:
            raise ValueError(f"Expected theta_backend6 to have 6 values, got {backend.numel()}")
        vals = backend.detach().cpu().double().numpy()
        validate_trace_seidel_constraints(vals, model_dim=dim, atol=atol)
        s = backend[0]
        c = backend[1]
        a = 0.5 * backend[2]
        f = backend[3] + 0.5 * backend[2]
        d = backend[4]
        if dim == 5:
            return torch.stack((s, c, a, f, d))
        return torch.stack((s, c, a, f)) if dim == 4 else torch.stack((s, c, a))

    backend = np.asarray(theta_backend6, dtype=np.float64).reshape(-1)
    if backend.size != NUM_SEIDEL:
        raise ValueError(f"Expected theta_backend6 to have 6 values, got {backend.size}")
    validate_trace_seidel_constraints(backend, model_dim=dim, atol=atol)
    s = backend[0]
    c = backend[1]
    a = 0.5 * backend[2]
    f = backend[3] + 0.5 * backend[2]
    d = backend[4]
    if dim == 5:
        return np.asarray([s, c, a, f, d], dtype=np.float64)
    if dim == 3:
        return np.asarray([s, c, a], dtype=np.float64)
    return np.asarray([s, c, a, f], dtype=np.float64)


def validate_trace_seidel_constraints(
    theta_backend6: Any,
    model_dim: int | str = 4,
    *,
    atol: float = 1e-8,
) -> None:
    """Validate trace constraints for the selected public convention."""
    dim = _normalize_trace_model_dim(model_dim)
    if torch.is_tensor(theta_backend6):
        backend = theta_backend6.detach().cpu().double().numpy().reshape(-1)
    else:
        backend = np.asarray(theta_backend6, dtype=np.float64).reshape(-1)
    if backend.size != NUM_SEIDEL:
        raise ValueError(f"Expected theta_backend6 to have 6 values, got {backend.size}")
    if dim in (3, 4) and abs(float(backend[4])) > float(atol):
        raise ValueError(f"W311/distortion must be zero within atol={atol}, got {backend[4]}")
    if abs(float(backend[5])) > float(atol):
        raise ValueError(f"Wd/W020 defocus must be zero within atol={atol}, got {backend[5]}")
    public_f = float(backend[3] + 0.5 * backend[2])
    if dim == 3 and abs(public_f) > float(atol):
        raise ValueError(
            "3D no_field_curvature model requires public F = "
            f"W220 + 0.5*W222 to be zero within atol={atol}, got {public_f}"
        )


def trace_seidel_wavefront(
    theta_trace: Any,
    field_x: Any,
    field_y: Any,
    pupil_x: Any,
    pupil_y: Any,
    *,
    model_dim: int | str | None = None,
) -> Any:
    """Diagnostic/test-only public trace-separated wavefront evaluator.

    This helper must not be used in the production forward path.  Production
    must expand public trace coefficients to backend 6D and then call the
    existing RDM PSF/ring-convolution code.
    """
    if torch.is_tensor(theta_trace) or any(
        torch.is_tensor(v) for v in (field_x, field_y, pupil_x, pupil_y)
    ):
        trace = (
            theta_trace.reshape(-1)
            if torch.is_tensor(theta_trace)
            else torch.as_tensor(theta_trace, dtype=torch.float32).reshape(-1)
        )
        dim = _normalize_trace_model_dim(model_dim, int(trace.numel()))
        if trace.numel() != dim:
            raise ValueError(
                f"Expected theta_trace{dim} to have {dim} values, got {trace.numel()}"
            )
        device = trace.device
        dtype = trace.dtype
        hx = torch.as_tensor(field_x, dtype=dtype, device=device)
        hy = torch.as_tensor(field_y, dtype=dtype, device=device)
        px = torch.as_tensor(pupil_x, dtype=dtype, device=device)
        py = torch.as_tensor(pupil_y, dtype=dtype, device=device)
        rho2 = px * px + py * py
        h2 = hx * hx + hy * hy
        h_dot_rho = hx * px + hy * py
        s, c, a = trace[0], trace[1], trace[2]
        zero = torch.zeros((), dtype=dtype, device=device)
        f = trace[3] if dim in (4, 5) else zero
        d = trace[4] if dim == 5 else zero
        return (
            s * rho2 * rho2
            + c * h_dot_rho * rho2
            + a * (2.0 * h_dot_rho * h_dot_rho - h2 * rho2)
            + f * h2 * rho2
            + d * h2 * h_dot_rho
        )

    trace_np = np.asarray(theta_trace, dtype=np.float64).reshape(-1)
    dim = _normalize_trace_model_dim(model_dim, int(trace_np.size))
    if trace_np.size != dim:
        raise ValueError(f"Expected theta_trace{dim} to have {dim} values, got {trace_np.size}")
    hx = np.asarray(field_x, dtype=np.float64)
    hy = np.asarray(field_y, dtype=np.float64)
    px = np.asarray(pupil_x, dtype=np.float64)
    py = np.asarray(pupil_y, dtype=np.float64)
    rho2 = px * px + py * py
    h2 = hx * hx + hy * hy
    h_dot_rho = hx * px + hy * py
    s, c, a = trace_np[0], trace_np[1], trace_np[2]
    f = trace_np[3] if dim in (4, 5) else 0.0
    d = trace_np[4] if dim == 5 else 0.0
    return (
        s * rho2 * rho2
        + c * h_dot_rho * rho2
        + a * (2.0 * h_dot_rho * h_dot_rho - h2 * rho2)
        + f * h2 * rho2
        + d * h2 * h_dot_rho
    )


# ── system-parameter resolution ─────────────────────────────────────────────

def build_sys_params(
    dim: int,
    sys_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete ``sys_params`` dict for rdmpy from user overrides.

    Mirrors the default-resolution logic in ``rdmpy/calibrate.py:267-275``.
    """
    if dim <= 0:
        raise ValueError("dim must be positive")

    provided = dict(sys_params or {})
    if "samples" in provided and int(provided["samples"]) != dim:
        raise ValueError(
            f"sys_params['samples'] must equal image size ({dim}), "
            f"got {provided['samples']}"
        )

    lamb = float(provided.get("lamb", DEFAULT_LAMBDA))
    na = float(provided.get("NA", DEFAULT_NA))

    if "L" in provided:
        system_length = float(provided["L"])
    else:
        radius_over_z = math.tan(math.asin(na))
        system_length = (dim * lamb) / (4 * radius_over_z)

    resolved: dict[str, Any] = {
        "samples": dim,
        "L": system_length,
        "lamb": lamb,
        "NA": na,
    }
    for k, v in provided.items():
        if k not in resolved:
            resolved[k] = v
    return resolved


# ── RoFT packing ────────────────────────────────────────────────────────────

def polar_psfs_to_roft(
    polar_psfs: torch.Tensor,
    *,
    buffer: int = 2,
) -> torch.Tensor:
    """Convert stacked polar PSFs into the packed RoFT representation.

    rdmpy's ``get_rdm_psfs`` does this in-place (``calibrate.py:331-335``),
    which breaks autograd.  This function produces the same result while
    keeping the computation graph alive.

    Parameters
    ----------
    polar_psfs : (num_radii, num_angles + buffer, num_radii) tensor
        PSFs in polar coordinates, with *buffer* extra angular rows at the end.
    buffer : int
        Number of trailing angular rows to skip before the FFT.

    Returns
    -------
    roft : (num_radii, freq_bins, num_radii) tensor
        Real/imag halves concatenated along dim-1.
    """
    if polar_psfs.ndim != 3:
        raise ValueError(
            f"Expected 3-D polar PSF stack, got shape {tuple(polar_psfs.shape)}"
        )
    if polar_psfs.shape[1] <= buffer:
        raise ValueError(
            "Not enough angular samples for RoFT packing "
            f"(got {polar_psfs.shape[1]} with buffer={buffer})"
        )

    # rfft along the angular axis (dim=1), skipping the trailing buffer rows
    packed_fft = torch.fft.rfft(polar_psfs[:, :-buffer, :], dim=1)
    return torch.cat((packed_fft.real, packed_fft.imag), dim=1)


# ── PSF generation ──────────────────────────────────────────────────────────

def get_reference_ring_psfs(
    seidel_coeffs: Any,
    dim: int,
    sys_params: dict[str, Any] | None = None,
    *,
    patch_size: int = 0,
    verbose: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate ring PSFs via rdmpy's released public API.

    This calls ``rdmpy.get_rdm_psfs(..., model='lri')`` directly.
    The result is *not* autograd-safe because rdmpy packs the RoFT in-place.
    Use :func:`get_trainable_ring_psfs` when you need gradients through the
    coefficients.
    """
    if device is None:
        device = (
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    resolved_device = device
    coeffs = normalize_seidel_coeffs(seidel_coeffs, device=resolved_device)
    params = build_sys_params(dim, sys_params)
    return _get_rdm_psfs(
        coeffs,
        dim=dim,
        model="lri",
        patch_size=patch_size,
        sys_params=params,
        verbose=verbose,
        device=resolved_device,
    )


def _compute_roft_psfs_checkpointed(
    coeffs: torch.Tensor,
    point_list: list[tuple[float, float]],
    params: dict[str, Any],
    *,
    device: torch.device,
    buffer: int = 2,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Chunked + intra-chunk **batched** PSF synthesis with autograd-safe
    RoFT packing.

    Equivalent to ``compute_rdm_psfs(..., polar=True, stack=True, buffer=2)``
    followed by :func:`polar_psfs_to_roft`. Two layers of optimisation
    relative to the reference ring-by-ring loop:

    * Within a chunk of ``chunk_size`` radii the entire pipeline
      (compute_pupil_phase → exp/ifftn → abs²/normalise → shift → polar →
      rfft) is **vectorised across radii**. CUDA fuses the work into a
      handful of large kernels instead of ``chunk_size`` × 8 small launches,
      bringing per-step wall time at 336² from ~5 sec to well under 1 sec.

    * Each chunk is wrapped in ``torch.utils.checkpoint`` — backward
      recomputes one chunk at a time, so peak activation memory scales with
      ``chunk_size`` rather than ``len(point_list)``. Default
      ``chunk_size=64`` keeps memory under ~10 GB at 768² (vs ~45 GB without
      chunking) and well inside the 49 GB Ada budget.

    Returns a tensor of shape ``(n_points, samples*4 + buffer, samples)``
    in RoFT format (real / imag halves concatenated along dim=1).
    """
    samples = params["samples"]
    L = params["L"]
    lamb = params["lamb"]
    radius_over_z = float(np.tan(np.arcsin(params["NA"])))
    dt = L / samples
    k = (2 * np.pi) / lamb
    scale_factor = lamb / radius_over_z
    half_samples = samples / 2.0

    # Pupil-grid constants — built once and shared (no autograd graph cost).
    fx = torch.linspace(-1 / (2 * dt), 1 / (2 * dt), samples, device=device)
    Fx, Fy = torch.meshgrid((fx, fx), indexing="xy")        # (N, N)
    circle = _circ(                                         # (N, N) real
        torch.sqrt(torch.square(Fx) + torch.square(Fy)) * scale_factor,
        radius=1,
    )
    X_pupil = -Fx * scale_factor                            # (N, N)
    Y_pupil = -Fy * scale_factor                            # (N, N)

    # Cartesian-shift grid base — for batched grid_sample below.
    xs = torch.arange(0, samples, device=device).float()
    ys = torch.arange(0, samples, device=device).float()
    gx_base, gy_base = torch.meshgrid(xs, ys, indexing="xy")  # (N, N) each

    def _chunk_batched(coeffs_inner: torch.Tensor,
                        u_vec: torch.Tensor,
                        v_vec: torch.Tensor) -> torch.Tensor:
        """Compute the RoFT-packed PSFs for a chunk of radii in one shot.

        Args
        ----
        coeffs_inner : (6,)
        u_vec, v_vec : (P,) point coordinates in pixel units (matching the
            (point[0], point[1]) tuples produced by the caller).

        Returns
        -------
        roft : (P, samples*4 + 2, samples) real
        """
        P = u_vec.shape[0]

        # ── Batched compute_pupil_phase ──────────────────────────────
        # The original ``compute_pupil_phase`` operates on scalar (u, v).
        # Inside it normalises u, v by half_samples and computes
        # rot_angle = atan2(v_norm, u_norm), obj_rad = sqrt(u_norm² + v_norm²).
        # We reproduce that with broadcasting across the chunk dim.
        u_norm = u_vec / half_samples           # (P,)
        v_norm = -v_vec / half_samples          # (P,) — note original passed v=-point[1]/half
        rot_angle = torch.atan2(v_norm, u_norm)  # (P,)
        obj_rad = torch.sqrt(u_norm * u_norm + v_norm * v_norm)  # (P,)
        cos_r = torch.cos(rot_angle)[:, None, None]     # (P, 1, 1)
        sin_r = torch.sin(rot_angle)[:, None, None]
        Xp = X_pupil[None]                              # (1, N, N)
        Yp = Y_pupil[None]
        X_rot = Xp * cos_r + Yp * sin_r                 # (P, N, N)
        Y_rot = -Xp * sin_r + Yp * cos_r                # (P, N, N)
        pupil_radii = X_rot * X_rot + Y_rot * Y_rot     # (P, N, N)
        obj_rad_b = obj_rad[:, None, None]              # (P, 1, 1)

        c = lamb * coeffs_inner                         # (6,)
        pupil_phase = (
            c[0] * pupil_radii * pupil_radii
            + c[1] * obj_rad_b * pupil_radii * X_rot
            + c[2] * (obj_rad_b * obj_rad_b) * (X_rot * X_rot)
            + c[3] * (obj_rad_b * obj_rad_b) * pupil_radii
            + c[4] * (obj_rad_b * obj_rad_b * obj_rad_b) * X_rot
            + c[5] * pupil_radii
        )                                               # (P, N, N)

        # ── exp + circle mask + ifftn ────────────────────────────────
        circle_b = circle[None]                         # (1, N, N)
        H = circle_b * torch.exp(-1j * k * pupil_phase)  # (P, N, N) complex
        H = torch.where(circle_b < 1e-12, torch.zeros_like(H), H)
        curr = torch.fft.fftshift(
            torch.fft.ifftn(
                torch.fft.ifftshift(H, dim=(-2, -1)),
                dim=(-2, -1),
            ),
            dim=(-2, -1),
        )                                               # (P, N, N) complex
        curr = (curr.real * curr.real + curr.imag * curr.imag)  # |·|² faster than abs²
        norm = curr.sum(dim=(-2, -1), keepdim=True)     # (P, 1, 1)
        curr = curr / norm                              # (P, N, N) real

        # ── Batched shift_torch via grid_sample ──────────────────────
        # Original shift_torch with shift=(-v_val, u_val): subtracts shift[1]
        # from x_grid and shift[0] from y_grid. Here that is x_grid - u_val
        # and y_grid - (-v_val) = y_grid + v_val.
        x_b = gx_base[None] - u_vec[:, None, None]      # (P, N, N)
        y_b = gy_base[None] - (-v_vec[:, None, None])
        gx = 2.0 * (x_b / (samples - 1)) - 1.0
        gy = 2.0 * (y_b / (samples - 1)) - 1.0
        grid = torch.stack([gx, gy], dim=-1)            # (P, N, N, 2)
        shifted = F.grid_sample(
            curr[:, None, :, :].float(),
            grid.float(),
            padding_mode="zeros",
            mode="bilinear",
            align_corners=True,
        )                                               # (P, 1, N, N)
        curr = shifted.squeeze(1)                       # (P, N, N)

        # ── Batched img2polar ────────────────────────────────────────
        polar = _polar_transform.batchimg2polar(
            curr[:, None, :, :], numRadii=samples,
        ).squeeze(1)                                    # (P, samples*4, samples)

        # ── Per-PSF rfft + concat along the polar-angle axis ─────────
        packed_fft = torch.fft.rfft(polar, dim=-2)       # (P, samples*2 + 1, samples) complex
        roft = torch.cat(
            [packed_fft.real, packed_fft.imag], dim=-2,
        )                                               # (P, samples*4 + 2, samples) real
        return roft

    # ── Chunk dispatcher ─────────────────────────────────────────────
    n = len(point_list)
    u_full = torch.tensor([p[0] for p in point_list], device=device, dtype=torch.float32)
    v_full = torch.tensor([p[1] for p in point_list], device=device, dtype=torch.float32)
    if chunk_size <= 0:
        return _chunk_batched(coeffs, u_full, v_full)

    if not torch.is_grad_enabled():
        out = torch.empty(
            (n, samples * 4 + 2, samples),
            device=device,
            dtype=torch.float32,
        )
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_roft = _chunk_batched(coeffs, u_full[start:end], v_full[start:end])
            out[start:end].copy_(chunk_roft)
            del chunk_roft
        return out

    chunks: list[torch.Tensor] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_roft = torch.utils.checkpoint.checkpoint(
            _chunk_batched,
            coeffs,
            u_full[start:end],
            v_full[start:end],
            use_reentrant=False,
        )
        chunks.append(chunk_roft)
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=0)


def get_trainable_ring_psfs(
    seidel_coeffs: Any,
    dim: int,
    sys_params: dict[str, Any] | None = None,
    *,
    patch_size: int = 0,
    verbose: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate ring PSFs with an autograd-safe RoFT packing path.

    The PSF synthesis AND RoFT packing are both done inside a per-iteration
    gradient checkpoint — see :func:`_compute_roft_psfs_checkpointed` for the
    memory rationale.

    Currently only ``patch_size=0`` is supported.
    """
    if patch_size != 0:
        raise NotImplementedError(
            "Autograd-safe PSF generation currently supports patch_size=0 only."
        )

    if device is None:
        device = (
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    resolved_device = device
    coeffs = normalize_seidel_coeffs(seidel_coeffs, device=resolved_device)
    params = build_sys_params(dim, sys_params)

    # Build the same radial point list as the original rdmpy.get_rdm_psfs body
    # (vendored at hybrid_ring_cocoa/_rdm/calibrate.py).
    radii = np.linspace(0, dim / 2, dim, endpoint=False)
    point_list = [(float(r), float(-r)) for r in radii]
    chunk_size = int(os.environ.get("COCOA_RING_PSF_CHUNK_SIZE", "64"))

    return _compute_roft_psfs_checkpointed(
        coeffs,
        point_list,
        params,
        device=resolved_device,
        buffer=2,
        chunk_size=chunk_size,
    )


def get_reference_trace_ring_psfs(
    theta_trace: Any,
    dim: int,
    sys_params: dict[str, Any] | None = None,
    *,
    model_dim: int | str | None = None,
    patch_size: int = 0,
    verbose: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate reference RDM ring PSFs from public trace coefficients.

    This is a thin parameterization wrapper:
    ``theta_trace -> expand_trace_seidel -> get_reference_ring_psfs``.
    It intentionally preserves the existing RDM PSF synthesis path.
    """
    theta_backend6 = expand_trace_seidel(theta_trace, model_dim=model_dim)
    return get_reference_ring_psfs(
        theta_backend6,
        dim,
        sys_params,
        patch_size=patch_size,
        verbose=verbose,
        device=device,
    )


def get_trainable_trace_ring_psfs(
    theta_trace: Any,
    dim: int,
    sys_params: dict[str, Any] | None = None,
    *,
    model_dim: int | str | None = None,
    patch_size: int = 0,
    verbose: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate trainable RDM ring PSFs from public trace coefficients.

    This is a thin parameterization wrapper:
    ``theta_trace -> expand_trace_seidel -> get_trainable_ring_psfs``.
    The trainable backend remains the existing vectorized/chunked RDM path.
    """
    if torch.is_tensor(theta_trace):
        theta_backend6 = expand_trace_seidel(
            theta_trace,
            model_dim=model_dim,
            device=device or theta_trace.device,
            dtype=theta_trace.dtype,
        )
    else:
        theta_backend6 = expand_trace_seidel(theta_trace, model_dim=model_dim)
    return get_trainable_ring_psfs(
        theta_backend6,
        dim,
        sys_params,
        patch_size=patch_size,
        verbose=verbose,
        device=device,
    )
