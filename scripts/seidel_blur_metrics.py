"""Convert Seidel coefficients into blur-relevant scalars comparable to the paper.

Why this exists
---------------
Our sweeps report a *field-weighted, piston-only* wavefront RMS (see
``hybrid_ring_cocoa/evaluation/seidel_operator_evaluator.py::field_weighted_wavefront_rms``
and ``run_seidel_blind_recovery_sweep.py::field_weighted_rms_backend``). The CoCoA
paper instead quotes a single-pupil wavefront RMS over a Zernike basis with
piston / tip-tilt / defocus excluded, and uses that number as a proxy for how
much the image is blurred.

At 0.3-0.4 lambda the RMS-as-blur proxy is well past the Maréchal regime, so this
tool also reports blur metrics derived from the *actual* PSF (Strehl, FWHM, R50),
which remain meaningful at large aberration. Use those to line our cases up with
the paper's 0.15 / 0.19 / 0.31 lambda anchors on the blur axis.

Units
-----
Seidel coefficients are in **waves (lambda)** in this codebase: the PSF builder does
``c = lamb * coeffs`` then ``exp(-1j * (2*pi/lamb) * pupil_phase)``, i.e. the
optical-path lambda cancels and the pupil phase in radians is ``2*pi * W(coeffs)``.
So every RMS below is in waves, directly comparable to the paper's "lambda r.m.s.".

Wavefront model (matches ``_seidel_wavefront``), with rho^2 = x^2 + y^2 and field h:
    W = W040*rho^4 + W131*h*rho^2*x + W222*h^2*x^2 + W220*h^2*rho^2
        + W311*h^3*x + Wd*rho^2

CLI examples
------------
    python scripts/seidel_blur_metrics.py --coeffs 0.05 0.20 0.04 0.02 0 0
    python scripts/seidel_blur_metrics.py --direction coma_dominant --strength 0.40
    python scripts/seidel_blur_metrics.py --direction coma_dominant --strength 0.40 --field-table
    python scripts/seidel_blur_metrics.py --coeffs 0 0 0 0 0 0.15 --na 1.05 --lam 0.55 --paper-anchors
"""

from __future__ import annotations

import argparse
import math
from typing import Sequence

import numpy as np

SEIDEL_COEFF_NAMES = ("W040", "W131", "W222", "W220", "W311", "Wd")

# Mirrors run_seidel_blind_recovery_sweep.py::DIRECTIONS (keep in sync if edited there).
DIRECTIONS: dict[str, np.ndarray] = {
    "balanced": np.asarray([0.30, -0.10, 0.10, 0.03, 0.00, 0.00], dtype=np.float64),
    "coma_dominant": np.asarray([0.05, 0.20, 0.04, 0.02, 0.00, 0.00], dtype=np.float64),
    "astig_field": np.asarray([0.08, 0.04, 0.32, -0.06, 0.00, 0.00], dtype=np.float64),
    "pure_distortion": np.asarray([0.0, 0.0, 0.0, 0.0, 0.04, 0.00], dtype=np.float64),
    "coma_distortion_mixed": np.asarray([0.0, -0.10, 0.0, 0.0, 0.04, 0.00], dtype=np.float64),
    "balanced_with_D": np.asarray([0.30, -0.10, 0.10, 0.03, 0.04, 0.00], dtype=np.float64),
}

# Diffraction-limited intensity-PSF FWHM for an incoherent circular pupil.
AIRY_FWHM_OVER_LAM_NA = 0.514


# --------------------------------------------------------------------------- #
# Wavefront and pupil helpers
# --------------------------------------------------------------------------- #
def seidel_wavefront(coeffs: Sequence[float], x: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    """Seidel wavefront in waves; identical algebra to ``_seidel_wavefront``."""
    c = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    rho2 = x * x + y * y
    return (
        c[0] * rho2 * rho2
        + c[1] * h * rho2 * x
        + c[2] * h * h * x * x
        + c[3] * h * h * rho2
        + c[4] * h * h * h * x
        + c[5] * rho2
    )


def _pupil_grid(pupil_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1 = np.linspace(-1.0, 1.0, int(pupil_samples), dtype=np.float64)
    x, y = np.meshgrid(x1, x1, indexing="xy")
    mask = (x * x + y * y) <= 1.0
    return x, y, mask


def _project_out(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the least-squares projection of ``values`` onto ``basis`` columns.

    The ``errstate`` guard silences a spurious divide/overflow RuntimeWarning that
    macOS Accelerate-BLAS raises from ``@``; the result is finite and correct.
    """
    coef, *_ = np.linalg.lstsq(basis, values, rcond=None)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return values - basis @ coef


def rms_over_pupil(
    coeffs: Sequence[float],
    h: float,
    *,
    remove: str = "piston",
    pupil_samples: int = 151,
) -> float:
    """Single-field wavefront RMS (waves) over the unit pupil.

    ``remove`` selects which non-blurring modes are projected out first:
      - "piston":  subtract the mean only (this codebase's native convention)
      - "ptd":     subtract piston + tip/tilt + defocus (the paper's convention)
    """
    x, y, mask = _pupil_grid(pupil_samples)
    w = seidel_wavefront(coeffs, x, y, float(h))[mask]
    xm, ym = x[mask], y[mask]
    if remove == "piston":
        w = w - float(np.mean(w))
    elif remove == "ptd":
        basis = np.column_stack([np.ones_like(xm), xm, ym, xm * xm + ym * ym])
        w = _project_out(w, basis)
    else:
        raise ValueError(f"unknown remove={remove!r}")
    return math.sqrt(float(np.mean(w * w)))


def field_weighted_rms(
    coeffs: Sequence[float],
    *,
    remove: str = "piston",
    field_samples: int = 41,
    pupil_samples: int = 151,
) -> float:
    """Field-weighted RMS (weight = h, h0 dropped). Matches the project diagnostic
    when ``remove='piston'``."""
    hs = np.linspace(0.0, 1.0, int(field_samples), dtype=np.float64)
    weights = hs.copy()
    weights[0] = 0.0
    per = np.array([rms_over_pupil(coeffs, h, remove=remove, pupil_samples=pupil_samples) for h in hs])
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return float(per[-1])
    return float(np.sum(per * weights) / denom)


# --------------------------------------------------------------------------- #
# PSF-based blur metrics (valid even far past the Maréchal regime)
# --------------------------------------------------------------------------- #
def _effective_phase(coeffs, x, y, h, mask, *, keep_defocus: bool) -> np.ndarray:
    """Pupil phase (radians) with piston+tilt removed; defocus optional.

    Tilt only shifts the PSF and defocus is degenerate with refocus in a stack, so
    we strip the modes that should not count as blur before forming the PSF."""
    w = seidel_wavefront(coeffs, x, y, float(h))
    xm, ym = x[mask], y[mask]
    cols = [np.ones_like(xm), xm, ym]
    if not keep_defocus:
        cols.append(xm * xm + ym * ym)
    basis = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(basis, w[mask], rcond=None)
    w = w.copy()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        w[mask] = w[mask] - basis @ coef
    return 2.0 * np.pi * w


def strehl(coeffs: Sequence[float], h: float, *, keep_defocus: bool = True, pupil_samples: int = 151) -> float:
    """Exact aperture-average Strehl S = |<exp(i*phi)>|^2 over the pupil."""
    x, y, mask = _pupil_grid(pupil_samples)
    phi = _effective_phase(coeffs, x, y, float(h), mask, keep_defocus=keep_defocus)[mask]
    return float(abs(np.mean(np.exp(1j * phi))) ** 2)


def _radial_profile(img: np.ndarray, center: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(img.shape)
    r = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    r_int = r.astype(np.int64)
    tbin = np.bincount(r_int.ravel(), img.ravel())
    nbin = np.bincount(r_int.ravel())
    profile = tbin / np.maximum(nbin, 1)
    return np.arange(profile.size, dtype=np.float64), profile


def _psf(coeffs, h, *, keep_defocus, grid=512, pupil_radius=64) -> np.ndarray:
    lin = (np.arange(grid) - grid / 2.0) / float(pupil_radius)
    x, y = np.meshgrid(lin, lin, indexing="xy")
    mask = (x * x + y * y) <= 1.0
    phi = _effective_phase(coeffs, x, y, float(h), mask, keep_defocus=keep_defocus)
    field = np.zeros((grid, grid), dtype=np.complex128)
    field[mask] = np.exp(1j * phi[mask])
    psf = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    return psf


def psf_blur_metrics(
    coeffs: Sequence[float],
    h: float,
    *,
    keep_defocus: bool = True,
    grid: int = 512,
    pupil_radius: int = 64,
) -> dict[str, float]:
    """FWHM and R50 (enclosed-energy radius) of the aberrated PSF, as multiples of
    the diffraction-limited PSF on the same grid."""
    out: dict[str, float] = {}
    perfect = _psf([0, 0, 0, 0, 0, 0], 0.0, keep_defocus=True, grid=grid, pupil_radius=pupil_radius)
    aberr = _psf(coeffs, h, keep_defocus=keep_defocus, grid=grid, pupil_radius=pupil_radius)

    def fwhm_pix(img: np.ndarray) -> float:
        c = np.unravel_index(int(np.argmax(img)), img.shape)
        radii, prof = _radial_profile(img, (float(c[0]), float(c[1])))
        half = prof[0] / 2.0
        below = np.where(prof <= half)[0]
        if below.size == 0:
            return float(radii[-1]) * 2.0
        i = int(below[0])
        if i == 0:
            return 0.0
        # linear interpolation between i-1 and i for the half-max crossing
        r0, r1, p0, p1 = radii[i - 1], radii[i], prof[i - 1], prof[i]
        r_half = r0 + (p0 - half) / max(p0 - p1, 1e-12) * (r1 - r0)
        return float(r_half) * 2.0

    def r50_pix(img: np.ndarray) -> float:
        c = np.unravel_index(int(np.argmax(img)), img.shape)
        radii, prof = _radial_profile(img, (float(c[0]), float(c[1])))
        ring_energy = prof * (2.0 * np.pi * np.maximum(radii, 0.5))  # ~energy per radial bin
        cum = np.cumsum(ring_energy)
        cum /= cum[-1]
        idx = int(np.searchsorted(cum, 0.5))
        return float(radii[min(idx, radii.size - 1)])

    f_perfect = fwhm_pix(perfect)
    r_perfect = r50_pix(perfect)
    out["fwhm_ratio"] = fwhm_pix(aberr) / max(f_perfect, 1e-12)
    out["r50_ratio"] = r50_pix(aberr) / max(r_perfect, 1e-12)
    return out


# --------------------------------------------------------------------------- #
# Direction + strength scaling (reproduces the sweep's ground-truth coeffs)
# --------------------------------------------------------------------------- #
def scaled_direction(direction: str, strength: float) -> np.ndarray:
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}; choose from {sorted(DIRECTIONS)}")
    base = DIRECTIONS[direction].astype(np.float64)
    base_rms = field_weighted_rms(base, remove="piston")
    if base_rms <= 1e-12:
        return base.copy()
    return base * (float(strength) / base_rms)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def metrics_at_field(coeffs: Sequence[float], h: float, *, keep_defocus: bool = True) -> dict[str, float]:
    psf = psf_blur_metrics(coeffs, h, keep_defocus=keep_defocus)
    return {
        "paper_rms": rms_over_pupil(coeffs, h, remove="ptd"),
        "native_rms_h": rms_over_pupil(coeffs, h, remove="piston"),
        "strehl": strehl(coeffs, h, keep_defocus=keep_defocus),
        "fwhm_ratio": psf["fwhm_ratio"],
        "r50_ratio": psf["r50_ratio"],
    }


def report(coeffs: Sequence[float], *, field: float, keep_defocus: bool, na, lam, field_table: bool) -> str:
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    lines: list[str] = []
    names = "  ".join(f"{n}={v:+.4f}" for n, v in zip(SEIDEL_COEFF_NAMES, coeffs))
    lines.append(f"Seidel coeffs (waves):  {names}")
    lines.append(f"Field h = {field:.2f}   (0 = on-axis center, 1 = field edge / full aberration)")
    lines.append(f"Defocus treated as blur in PSF: {keep_defocus}  (our experiment is 2D single-plane)")
    lines.append("")

    fw_native = field_weighted_rms(coeffs, remove="piston")
    fw_paper = field_weighted_rms(coeffs, remove="ptd")
    m = metrics_at_field(coeffs, field, keep_defocus=keep_defocus)

    lines.append("Aberration magnitude (wavefront RMS, units of lambda)")
    lines.append(f"  paper-convention RMS @h   (piston+tilt+defocus removed) : {m['paper_rms']:.4f} lambda")
    lines.append(f"  native RMS @h             (piston only)                 : {m['native_rms_h']:.4f} lambda")
    lines.append(f"  native field-weighted RMS (piston only, == sweep knob)  : {fw_native:.4f} lambda")
    lines.append(f"  paper  field-weighted RMS (piston+tilt+defocus removed) : {fw_paper:.4f} lambda")
    lines.append("")

    lines.append("Blur metrics @h  (piston+tilt removed; PSF from actual pupil)")
    lines.append(f"  Strehl ratio              : {m['strehl']:.4f}")
    lines.append(f"  PSF FWHM / diffraction    : {m['fwhm_ratio']:.2f}x")
    lines.append(f"  PSF R50  / diffraction    : {m['r50_ratio']:.2f}x")
    if na and lam:
        dl_fwhm = AIRY_FWHM_OVER_LAM_NA * lam / na  # microns if lam in microns
        lines.append(
            f"  absolute FWHM             : {m['fwhm_ratio'] * dl_fwhm * 1000:.0f} nm"
            f"   (diffraction limit {dl_fwhm * 1000:.0f} nm, lam={lam} um, NA={na})"
        )
    lines.append("")
    lines.append(
        "  Read: Strehl<<1 and FWHM>>1 mean heavy blur. RMS alone stops tracking blur"
    )
    lines.append(
        "  past ~0.1 lambda, so compare Strehl/FWHM, not RMS, in the 0.3-0.4 lambda regime."
    )

    if field_table:
        lines.append("")
        lines.append("Per-field variation (shows non-uniform blur across the FOV):")
        lines.append("   h     paperRMS  nativeRMS  Strehl  FWHMx  R50x")
        for h in np.linspace(0.0, 1.0, 5):
            mm = metrics_at_field(coeffs, float(h), keep_defocus=keep_defocus)
            lines.append(
                f"  {h:.2f}   {mm['paper_rms']:7.4f}  {mm['native_rms_h']:8.4f}  "
                f"{mm['strehl']:6.3f}  {mm['fwhm_ratio']:5.2f}  {mm['r50_ratio']:5.2f}"
            )
    return "\n".join(lines)


def paper_anchor_table() -> str:
    """Exact Strehl for a single astigmatism mode at the paper's reference RMS values.

    Illustrative anchor only: beyond ~0.2 lambda the exact Strehl is mode-dependent,
    so other Zernike mixes differ by a few percent. The Maréchal column is the
    small-aberration approximation exp(-(2*pi*sigma)^2) and is invalid past ~0.15.
    """
    x, y, mask = _pupil_grid(201)
    xm, ym = x[mask], y[mask]
    z_astig = np.sqrt(6.0) * (xm * xm - ym * ym)  # ANSI vertical astigmatism, unit RMS on the disk
    z_astig -= z_astig.mean()
    z_astig /= np.sqrt(np.mean(z_astig * z_astig))
    lines = ["Paper RMS anchors (lambda) -> Strehl"]
    lines.append("  RMS     Strehl(exact,astig)  Strehl(Marechal approx)   note")
    notes = {0.072: "Marechal diff-limited", 0.075: "Rayleigh limit",
             0.15: "Fig.3 single mode", 0.19: "Fig.4 low-order cutoff",
             0.31: "Fig.4 strongest", 0.40: "our example"}
    for sigma in (0.072, 0.075, 0.15, 0.19, 0.31, 0.40):
        phi = 2.0 * np.pi * sigma * z_astig
        s_exact = float(abs(np.mean(np.exp(1j * phi))) ** 2)
        s_mar = math.exp(-((2.0 * math.pi * sigma) ** 2))
        lines.append(f"  {sigma:.3f}   {s_exact:18.4f}  {s_mar:22.4f}   {notes.get(sigma, '')}")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--coeffs", nargs=6, type=float, metavar=SEIDEL_COEFF_NAMES,
                     help="six Seidel coefficients in waves: W040 W131 W222 W220 W311 Wd")
    src.add_argument("--direction", choices=sorted(DIRECTIONS), help="named sweep direction (use with --strength)")
    src.add_argument("--base", nargs=6, type=float, metavar=SEIDEL_COEFF_NAMES,
                     help="raw (unscaled) 6-vector from any sweep; scaled to --strength after --fix")
    p.add_argument("--fix", nargs="*", type=int, default=None, metavar="IDX",
                   help="indices zeroed before scaling --base (4D model = '--fix 4 5'; 5D = '--fix 5')")
    p.add_argument("--strength", type=float, help="target field-weighted RMS (waves) for --direction/--base")
    p.add_argument("--field", type=float, default=1.0, help="field height h in [0,1] (default 1.0 = edge)")
    p.add_argument("--no-defocus-blur", action="store_true",
                   help="also remove defocus before the PSF (mimic a refocused 3D stack)")
    p.add_argument("--field-table", action="store_true", help="print metrics across field heights")
    p.add_argument("--na", type=float, default=None, help="numerical aperture for absolute FWHM")
    p.add_argument("--lam", type=float, default=None, help="wavelength in microns for absolute FWHM")
    p.add_argument("--paper-anchors", action="store_true", help="print paper RMS->Strehl reference table")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.direction is not None:
        if args.strength is None:
            raise SystemExit("--direction requires --strength")
        coeffs = scaled_direction(args.direction, args.strength)
        print(f"# direction={args.direction}  strength={args.strength}")
    elif args.base is not None:
        if args.strength is None:
            raise SystemExit("--base requires --strength")
        base = np.asarray(args.base, dtype=np.float64)
        if args.fix:
            base[list(args.fix)] = 0.0
        base_rms = field_weighted_rms(base, remove="piston")
        coeffs = base * (args.strength / base_rms) if base_rms > 1e-12 else base
        print(f"# base={args.base}  fix={args.fix or []}  strength={args.strength}")
    else:
        coeffs = np.asarray(args.coeffs, dtype=np.float64)
    print(report(
        coeffs,
        field=args.field,
        keep_defocus=not args.no_defocus_blur,
        na=args.na,
        lam=args.lam,
        field_table=args.field_table,
    ))
    if args.paper_anchors:
        print()
        print(paper_anchor_table())


if __name__ == "__main__":
    main()
