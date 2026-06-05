#!/usr/bin/env python3
"""Render paper-style wavefront figures (GT / Aligned) using the ``jet`` colormap.

This reproduces the layout of the codex ``*_wavefront_paper_style_*.png`` cards
but with classic matplotlib ``jet`` instead of ``turbo``, a transparent pupil
exterior, an anti-aliased circular edge, and a slim shared "(Wave)" colorbar, so
the colour and feel match the reference paper figure (e.g. the NeAT / DWS
panels).

The real per-candidate Seidel coefficients are used unchanged -- no fabricated
structure. GT comes from ``seidel_gt`` and Aligned from
``aligned_seidel_physical`` in
``outputs/.../seidel_physical_similarity_cards/candidate_index.csv``.

Usage:
    python3 scripts/render_wavefront_paper_style.py                  # top_operator, h=1.0 & 0.5
    python3 scripts/render_wavefront_paper_style.py --all            # every candidate row
    python3 scripts/render_wavefront_paper_style.py --kind mid_operator
    python3 scripts/render_wavefront_paper_style.py --cmap turbo     # side-by-side comparison
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import (  # noqa: E402
    _seidel_wavefront,
    apply_seidel_transform,
    coerce_seidel_vector,
)

CARDS_DIR = (
    REPO_ROOT
    / "outputs"
    / "cocoa_like_2d_mechanism"
    / "seidel_physical_similarity_cards"
)
CANDIDATE_CSV = CARDS_DIR / "candidate_index.csv"


def parse_vec(text: str) -> np.ndarray:
    """Parse a JSON-ish ``[...]`` coefficient string into a length-6 vector."""
    return np.asarray(ast.literal_eval(text), dtype=np.float64).reshape(6)


def load_rows(csv_path: Path = CANDIDATE_CSV) -> list[dict]:
    with open(csv_path, newline="") as fh:
        return list(csv.DictReader(fh))


def disc_wavefront(theta, h: float, n: int) -> np.ndarray:
    """Seidel wavefront sampled on the unit disc; NaN outside the pupil."""
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    X, Y = np.meshgrid(x, x, indexing="xy")
    mask = (X * X + Y * Y) <= 1.0
    W = _seidel_wavefront(coerce_seidel_vector(theta), X, Y, float(h))
    return np.where(mask, W, np.nan)


def h_tag(h: float) -> str:
    if abs(h - 1.0) < 1e-9:
        return "1"
    if abs(h - 0.5) < 1e-9:
        return "05"
    return ("%g" % h).replace(".", "p")


def out_path_for(row: dict, h: float, cmap_name: str) -> Path:
    base = Path(row["card_png"]).name.replace("__physical_similarity_card.png", "")
    return CARDS_DIR / f"{base}__wavefront_paper_style_h{h_tag(h)}_{cmap_name}.png"


def render(
    row: dict,
    h: float,
    *,
    cmap_name: str = "jet",
    n: int = 1000,
    dpi: int = 200,
    model_label: str = "size512 6D",
) -> Path:
    theta_gt = parse_vec(row["seidel_gt"])
    theta_aligned = parse_vec(row["aligned_seidel_physical"])
    transform = row.get("best_physical_transform", "twin")
    op = float(row["operator_error_calibrated"])

    # Cross-check: the aligned coeffs should equal the physical (e.g. twin)
    # sign-transform of the raw recovered coeffs -- this is the equivalence the
    # evaluator gated on. Warn (don't fail) if the CSV is inconsistent.
    if row.get("raw_seidel_final"):
        chk = apply_seidel_transform(parse_vec(row["raw_seidel_final"]), transform)
        diff = float(np.max(np.abs(chk - theta_aligned)))
        if diff > 1e-5:
            print(
                f"  [warn] aligned vs apply_seidel_transform(raw,'{transform}') "
                f"max|diff|={diff:.2e}"
            )

    W_gt = disc_wavefront(theta_gt, h, n)
    W_al = disc_wavefront(theta_aligned, h, n)

    # Shared 0 -> vmax scale across both panels (like the reference colorbar).
    vmin = 0.0
    vmax = float(np.nanmax([np.nanmax(W_gt), np.nanmax(W_al)]))
    assert np.isfinite(vmax) and vmax > 0.0, f"bad vmax={vmax!r}"

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(alpha=0.0)  # transparent outside the pupil

    fig = plt.figure(figsize=(3.6, 6.2))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.03, right=0.76, top=0.88, bottom=0.04, hspace=0.16)

    im = None
    for idx, (label, W) in enumerate((("GT", W_gt), ("Aligned", W_al))):
        ax = fig.add_subplot(2, 1, idx + 1)
        im = ax.imshow(
            W,
            extent=[-1, 1, -1, 1],
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
        # Perfectly round, anti-aliased pupil edge (no pixel stair-stepping).
        im.set_clip_path(Circle((0, 0), 1.0, transform=ax.transData))
        ax.set_xlim(-1.06, 1.06)
        ax.set_ylim(-1.06, 1.06)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(label, fontsize=15, pad=4)

    fig.suptitle(
        f"{model_label} | h={h:.1f}\noperator={op:.4f}, best_phys={transform}",
        fontsize=11,
    )

    # Slim shared vertical colorbar, endpoints only, "(Wave)" label.
    cax = fig.add_axes([0.80, 0.30, 0.045, 0.40])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(["0", f"{vmax:.2f}"])
    cbar.ax.tick_params(labelsize=12, length=0)
    cbar.set_label("(Wave)", rotation=270, labelpad=16, fontsize=13)
    cbar.outline.set_visible(False)

    out = out_path_for(row, h, cmap_name)
    # bbox_inches="tight" keeps the rotated "(Wave)" label from being clipped.
    fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  saved {out.name}  (vmax={vmax:.3f})")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Render paper-style jet wavefront figures from candidate_index.csv"
    )
    ap.add_argument("--csv", type=Path, default=CANDIDATE_CSV)
    ap.add_argument(
        "--kind",
        default="top_operator",
        help="candidate 'kind' column value to render (default: top_operator)",
    )
    ap.add_argument("--all", action="store_true", help="render every candidate row")
    ap.add_argument("--cmap", default="jet", help="matplotlib colormap (default: jet)")
    ap.add_argument(
        "--h",
        type=float,
        nargs="+",
        default=[1.0, 0.5],
        help="field coordinate(s) to render (default: 1.0 0.5)",
    )
    ap.add_argument("--n", type=int, default=1000, help="pupil grid samples per axis")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)

    rows = load_rows(args.csv)
    if args.all:
        selected = rows
    else:
        selected = [r for r in rows if r.get("kind") == args.kind] or rows[:1]

    for row in selected:
        print(f"[{row.get('kind', '?')}] {row.get('image', '?')} / {row.get('direction', '?')}")
        for h in args.h:
            render(row, h, cmap_name=args.cmap, n=args.n, dpi=args.dpi)


if __name__ == "__main__":
    main()
