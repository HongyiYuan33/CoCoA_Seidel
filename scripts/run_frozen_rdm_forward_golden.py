"""Frozen RDM forward-model golden regression.

This script is intentionally tests/diagnostics only.  It does not implement a
new forward model.  Every production check calls the existing path:

    theta -> get_rdm_psfs / trainable equivalent -> ring_convolve

The reduced trace-separated API is paused for default experiments and tested
only as a parameterization wrapper for explicit reproduction:

    theta_trace -> expand_trace_seidel -> existing backend6 RDM path
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_ring_cocoa.evaluation import (  # noqa: E402
    OperatorProbeConfig,
)
from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import (  # noqa: E402
    RingOperatorProbeEvaluator,
    evaluate_trace_seidel_recovery,
)
from hybrid_ring_cocoa.optics.ring_forward import (  # noqa: E402
    blur_ring,
    blur_ring_trace,
    blur_ring_with_psfs,
)
from hybrid_ring_cocoa.optics.seidel_psf import (  # noqa: E402
    expand_trace_seidel,
    get_reference_ring_psfs,
    get_reference_trace_ring_psfs,
    get_trainable_ring_psfs,
    get_trainable_trace_ring_psfs,
)
from hybrid_ring_cocoa.training.data import load_baboon_gt  # noqa: E402


SEED = 1729
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "golden_forward_regression"
DEFAULT_DIMS = (64, 128)
OPTIONAL_DIM = 256
SYS_PARAM_FIXTURES: dict[str, dict[str, float]] = {
    "default_like": {"lamb": 0.55e-6, "NA": 0.5},
    "sweep_like": {"lamb": 0.55e-6, "NA": 0.45},
}
TOLERANCES: dict[str, float] = {
    "reference_operator_rtol": 1e-6,
    "trace_wrapper_rtol": 1e-6,
    "trainable_psf_rtol": 1e-4,
    "trainable_operator_rtol": 1e-4,
    "identity_operator_rtol": 1e-8,
}


@dataclass(frozen=True)
class ProbeRecord:
    group: str
    name: str
    image: torch.Tensor


def set_deterministic_seeds(seed: int = SEED) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def relative_l2(a: Any, b: Any, *, eps: float = 1e-12) -> float:
    a_t = torch.as_tensor(a).detach().cpu().double()
    b_t = torch.as_tensor(b).detach().cpu().double()
    num = torch.linalg.vector_norm(a_t - b_t)
    den = torch.clamp(torch.linalg.vector_norm(b_t), min=float(eps))
    return float((num / den).item())


def l2_normalize_probe(probe: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    probe = probe.detach().clone().float()
    norm = torch.linalg.vector_norm(probe)
    if float(norm) <= eps:
        return probe
    return probe / norm


def backend_theta_fixtures() -> dict[str, np.ndarray]:
    trace5 = np.asarray([0.30, -0.10, 0.05, 0.08, 0.04], dtype=np.float64)
    trace4 = np.asarray([0.30, -0.10, 0.05, 0.08], dtype=np.float64)
    trace3 = np.asarray([0.30, -0.10, 0.05], dtype=np.float64)
    return {
        "zero": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "trace5_no_defocus_with_distortion_expanded": np.asarray(
            expand_trace_seidel(trace5, model_dim=5),
            dtype=np.float64,
        ),
        "trace4_normal_expanded": np.asarray(
            expand_trace_seidel(trace4, model_dim=4),
            dtype=np.float64,
        ),
        "trace3_normal_expanded": np.asarray(
            expand_trace_seidel(trace3, model_dim=3),
            dtype=np.float64,
        ),
        "legacy_backend6_with_defocus": np.asarray(
            [0.20, -0.05, 0.08, 0.04, 0.00, 0.06],
            dtype=np.float64,
        ),
        "legacy_backend6_with_distortion": np.asarray(
            [0.15, 0.04, 0.05, 0.03, 0.07, 0.02],
            dtype=np.float64,
        ),
    }


def trace_theta_fixtures() -> dict[str, tuple[int, np.ndarray]]:
    return {
        "trace5_no_defocus_with_distortion": (
            5,
            np.asarray([0.30, -0.10, 0.05, 0.08, 0.04], dtype=np.float64),
        ),
        "trace4_normal": (4, np.asarray([0.30, -0.10, 0.05, 0.08], dtype=np.float64)),
        "trace3_normal": (3, np.asarray([0.30, -0.10, 0.05], dtype=np.float64)),
    }


def build_probe_records(
    dim: int,
    *,
    include_natural: bool = True,
    seed: int = SEED,
    device: torch.device | None = None,
) -> tuple[list[ProbeRecord], list[dict[str, str]]]:
    if device is None:
        device = torch.device("cpu")
    probes: list[ProbeRecord] = []
    skipped: list[dict[str, str]] = []

    def add(group: str, name: str, probe: torch.Tensor) -> None:
        probes.append(ProbeRecord(group, name, l2_normalize_probe(probe.to(device))))

    # Delta-grid probes: center, quarter points, and safe near-edge positions.
    delta_positions = [
        (dim // 2, dim // 2),
        (dim // 4, dim // 4),
        (dim // 4, 3 * dim // 4),
        (3 * dim // 4, dim // 4),
        (max(1, dim // 8), min(dim - 2, 7 * dim // 8)),
    ]
    for idx, (row, col) in enumerate(delta_positions):
        probe = torch.zeros((dim, dim), dtype=torch.float32, device=device)
        probe[int(row), int(col)] = 1.0
        add("delta_grid", f"delta_{idx}_r{row}_c{col}", probe)

    coords = torch.linspace(-1.0, 1.0, dim, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    rr = torch.sqrt(xx * xx + yy * yy)

    for center, width in ((0.25, 0.08), (0.55, 0.10), (0.85, 0.11)):
        probe = torch.exp(-0.5 * ((rr - center) / width) ** 2)
        add("radial_basis", f"radial_gaussian_c{center:.2f}", probe)

    x_idx = torch.arange(dim, dtype=torch.float32, device=device) / float(dim)
    y_idx = torch.arange(dim, dtype=torch.float32, device=device) / float(dim)
    fy, fx = torch.meshgrid(y_idx, x_idx, indexing="ij")
    for kx, ky in ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2)):
        phase = 2.0 * np.pi * (float(kx) * fx + float(ky) * fy)
        add("fourier", f"cos_{kx}_{ky}", torch.cos(phase))
        add("fourier", f"sin_{kx}_{ky}", torch.sin(phase))

    rng = np.random.default_rng(int(seed) + int(dim))
    rademacher = rng.choice([-1.0, 1.0], size=(dim, dim)).astype(np.float32)
    gaussian = rng.standard_normal((dim, dim)).astype(np.float32)
    add("random", "rademacher_seeded", torch.as_tensor(rademacher, device=device))
    add("random", "gaussian_seeded", torch.as_tensor(gaussian, device=device))

    if include_natural:
        try:
            natural = load_baboon_gt(dim, device=device)
            add("natural_image", "baboon", natural)
        except Exception as exc:  # pragma: no cover - depends on local asset availability
            skipped.append({"group": "natural_image", "reason": str(exc)})

    return probes, skipped


def limit_probes_per_group(
    probes: list[ProbeRecord],
    max_probes_per_group: int | None,
) -> list[ProbeRecord]:
    if max_probes_per_group is None or int(max_probes_per_group) <= 0:
        return probes
    counts: dict[str, int] = {}
    out: list[ProbeRecord] = []
    limit = int(max_probes_per_group)
    for probe in probes:
        count = counts.get(probe.group, 0)
        if count < limit:
            out.append(probe)
            counts[probe.group] = count + 1
    return out


def _base_row(
    *,
    check: str,
    dim: int,
    sys_params_name: str,
    theta_fixture: str,
    probe: ProbeRecord | None,
    error: float,
    tolerance: float,
    likely_drift: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "check": check,
        "dim": int(dim),
        "sys_params_name": sys_params_name,
        "theta_fixture": theta_fixture,
        "probe_group": probe.group if probe is not None else "operator_probe_set",
        "probe_name": probe.name if probe is not None else "deterministic_operator_probes",
        "relative_error": float(error),
        "tolerance": float(tolerance),
        "pass": bool(error <= tolerance),
        "likely_drift": likely_drift,
    }
    if extra:
        row.update(extra)
    return row


def _probe_output_rows(
    *,
    check: str,
    dim: int,
    sys_params_name: str,
    theta_fixture: str,
    probes: Iterable[ProbeRecord],
    lhs_psfs: torch.Tensor | None = None,
    rhs_psfs: torch.Tensor | None = None,
    lhs_fn: Any | None = None,
    rhs_fn: Any | None = None,
    tolerance: float,
    likely_drift: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        if lhs_psfs is not None:
            lhs = blur_ring_with_psfs(probe.image, lhs_psfs, patch_size=0)
        else:
            lhs = lhs_fn(probe.image)
        if rhs_psfs is not None:
            rhs = blur_ring_with_psfs(probe.image, rhs_psfs, patch_size=0)
        else:
            rhs = rhs_fn(probe.image)
        err = relative_l2(lhs, rhs)
        rows.append(
            _base_row(
                check=check,
                dim=dim,
                sys_params_name=sys_params_name,
                theta_fixture=theta_fixture,
                probe=probe,
                error=err,
                tolerance=tolerance,
                likely_drift=likely_drift,
            )
        )
    return rows


def operator_probe_config() -> OperatorProbeConfig:
    return OperatorProbeConfig(
        delta_grid_size=3,
        radial_basis_count=3,
        fourier_frequencies=((1, 0), (0, 1), (1, 1), (2, 0), (0, 2)),
        random_count=2,
        random_seed=SEED,
        diagnostic_psf_points=((0.0, 0.0), (0.25, -0.25), (0.5, -0.5)),
        wavefront_field_samples=9,
        wavefront_pupil_samples=41,
        twin_invariance_tol=1e-7,
    )


def run_golden_suite(
    *,
    dims: Iterable[int] = DEFAULT_DIMS,
    include_natural: bool = True,
    include_trainable: bool = True,
    tolerances: dict[str, float] | None = None,
    max_probes_per_group: int | None = None,
    direct_blur_max_probes_per_group: int | None = 1,
    device: torch.device | None = None,
) -> dict[str, Any]:
    set_deterministic_seeds(SEED)
    if device is None:
        device = torch.device("cpu")
    tol = dict(TOLERANCES)
    if tolerances:
        tol.update({str(k): float(v) for k, v in tolerances.items()})

    backend_fixtures = backend_theta_fixtures()
    trace_fixtures = trace_theta_fixtures()
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for dim in [int(d) for d in dims]:
        probes, skipped_for_dim = build_probe_records(
            dim,
            include_natural=include_natural,
            device=device,
        )
        probes = limit_probes_per_group(probes, max_probes_per_group)
        direct_blur_probes = limit_probes_per_group(
            probes,
            direct_blur_max_probes_per_group,
        )
        for item in skipped_for_dim:
            skipped.append({"dim": str(dim), **item})

        for sys_name, sys_params in SYS_PARAM_FIXTURES.items():
            for theta_name, theta_backend6 in backend_fixtures.items():
                psfs_ref = get_reference_ring_psfs(
                    theta_backend6,
                    dim,
                    sys_params,
                    patch_size=0,
                    device=device,
                )

                # Protect that the direct reference path and explicit PSF path
                # remain behaviorally identical. This intentionally calls
                # blur_ring, even though it recomputes PSFs.
                rows.extend(
                    _probe_output_rows(
                        check="reference_backend_consistency",
                        dim=dim,
                        sys_params_name=sys_name,
                        theta_fixture=theta_name,
                        probes=direct_blur_probes,
                        lhs_psfs=psfs_ref,
                        rhs_fn=lambda probe, theta=theta_backend6, sp=sys_params: blur_ring(
                            probe,
                            theta,
                            sp,
                            patch_size=0,
                        ),
                        tolerance=tol["reference_operator_rtol"],
                        likely_drift=(
                            "reference PSF synthesis / theta order / defocus-last / "
                            "ring_convolve / normalization drift"
                        ),
                    )
                )

                if include_trainable:
                    psfs_train = get_trainable_ring_psfs(
                        torch.as_tensor(theta_backend6, dtype=torch.float32, device=device),
                        dim,
                        sys_params,
                        patch_size=0,
                        device=device,
                    )
                    psf_err = relative_l2(psfs_train, psfs_ref)
                    rows.append(
                        _base_row(
                            check="backend_reference_vs_trainable_psf",
                            dim=dim,
                            sys_params_name=sys_name,
                            theta_fixture=theta_name,
                            probe=None,
                            error=psf_err,
                            tolerance=tol["trainable_psf_rtol"],
                            likely_drift=(
                                "trainable/reference PSF mismatch: PSF synthesis / "
                                "RoFT packing / interpolation / normalization drift"
                            ),
                        )
                    )
                    rows.extend(
                        _probe_output_rows(
                            check="backend_reference_vs_trainable_operator",
                            dim=dim,
                            sys_params_name=sys_name,
                            theta_fixture=theta_name,
                            probes=probes,
                            lhs_psfs=psfs_train,
                            rhs_psfs=psfs_ref,
                            tolerance=tol["trainable_operator_rtol"],
                            likely_drift=(
                                "trainable/reference operator mismatch: RoFT packing / "
                                "ring_convolve / interpolation / normalization drift"
                            ),
                        )
                    )

                evaluator = RingOperatorProbeEvaluator(
                    dim=dim,
                    sys_params=sys_params,
                    probe_config=operator_probe_config(),
                    device=device,
                )
                identity_err = evaluator.distance(theta_backend6, theta_backend6)
                rows.append(
                    _base_row(
                        check="backend_operator_identity",
                        dim=dim,
                        sys_params_name=sys_name,
                        theta_fixture=theta_name,
                        probe=None,
                        error=identity_err,
                        tolerance=tol["identity_operator_rtol"],
                        likely_drift="exact operator deterministic probe identity drift",
                    )
                )

            for trace_name, (model_dim, theta_trace) in trace_fixtures.items():
                theta_backend6 = np.asarray(
                    expand_trace_seidel(theta_trace, model_dim=model_dim),
                    dtype=np.float64,
                )
                rows.extend(
                    _probe_output_rows(
                        check="trace_wrapper_equivalence",
                        dim=dim,
                        sys_params_name=sys_name,
                        theta_fixture=trace_name,
                        probes=probes,
                        lhs_fn=lambda probe, theta=theta_trace, md=model_dim, sp=sys_params: blur_ring_trace(
                            probe,
                            theta,
                            sp,
                            model_dim=md,
                            patch_size=0,
                        ),
                        rhs_fn=lambda probe, theta=theta_backend6, sp=sys_params: blur_ring(
                            probe,
                            theta,
                            sp,
                            patch_size=0,
                        ),
                        tolerance=tol["trace_wrapper_rtol"],
                        likely_drift="trace expansion issue or wrapper changed production RDM forward path",
                    )
                )

                if include_trainable:
                    psfs_ref_trace = get_reference_trace_ring_psfs(
                        theta_trace,
                        dim,
                        sys_params,
                        model_dim=model_dim,
                        patch_size=0,
                        device=device,
                    )
                    psfs_train_trace = get_trainable_trace_ring_psfs(
                        torch.as_tensor(theta_trace, dtype=torch.float32, device=device),
                        dim,
                        sys_params,
                        model_dim=model_dim,
                        patch_size=0,
                        device=device,
                    )
                    psf_err = relative_l2(psfs_train_trace, psfs_ref_trace)
                    rows.append(
                        _base_row(
                            check="trace_reference_vs_trainable_psf",
                            dim=dim,
                            sys_params_name=sys_name,
                            theta_fixture=trace_name,
                            probe=None,
                            error=psf_err,
                            tolerance=tol["trainable_psf_rtol"],
                            likely_drift=(
                                "trace-expanded trainable/reference PSF mismatch: "
                                "trace expansion / RoFT packing / normalization drift"
                            ),
                        )
                    )
                    rows.extend(
                        _probe_output_rows(
                            check="trace_reference_vs_trainable_operator",
                            dim=dim,
                            sys_params_name=sys_name,
                            theta_fixture=trace_name,
                            probes=probes,
                            lhs_psfs=psfs_train_trace,
                            rhs_psfs=psfs_ref_trace,
                            tolerance=tol["trainable_operator_rtol"],
                            likely_drift=(
                                "trace-expanded trainable/reference operator mismatch: "
                                "trace expansion / ring_convolve / interpolation drift"
                            ),
                        )
                    )

                metrics = evaluate_trace_seidel_recovery(
                    theta_trace,
                    theta_trace,
                    dim=dim,
                    sys_params=sys_params,
                    model_dim=model_dim,
                    probe_config=operator_probe_config(),
                    dataset_twin_invariance_pass=False,
                )
                for key in (
                    "operator_error_strict",
                    "operator_error_phys_equiv",
                    "operator_error_coord_diagnostic",
                ):
                    rows.append(
                        _base_row(
                            check=f"trace_operator_identity_{key}",
                            dim=dim,
                            sys_params_name=sys_name,
                            theta_fixture=trace_name,
                            probe=None,
                            error=float(metrics[key]),
                            tolerance=tol["identity_operator_rtol"],
                            likely_drift="trace exact-operator metric identity drift",
                        )
                    )

    failed = [row for row in rows if not row["pass"]]
    grouped: dict[str, dict[str, float | int]] = {}
    for row in rows:
        check = str(row["check"])
        item = grouped.setdefault(
            check,
            {"count": 0, "failed": 0, "max_error": 0.0, "mean_error": 0.0},
        )
        item["count"] = int(item["count"]) + 1
        item["failed"] = int(item["failed"]) + (0 if row["pass"] else 1)
        item["max_error"] = max(float(item["max_error"]), float(row["relative_error"]))
        item["mean_error"] = float(item["mean_error"]) + float(row["relative_error"])
    for item in grouped.values():
        if int(item["count"]) > 0:
            item["mean_error"] = float(item["mean_error"]) / int(item["count"])

    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit_hash(),
        "seed": SEED,
        "dims": [int(d) for d in dims],
        "sys_params": SYS_PARAM_FIXTURES,
        "backend_theta_fixtures": {
            name: [float(v) for v in theta] for name, theta in backend_fixtures.items()
        },
        "trace_theta_fixtures": {
            name: {"model_dim": int(model_dim), "theta": [float(v) for v in theta]}
            for name, (model_dim, theta) in trace_fixtures.items()
        },
        "tolerances": tol,
        "probe_groups": sorted({row["probe_group"] for row in rows}),
        "max_probes_per_group": max_probes_per_group,
        "direct_blur_max_probes_per_group": direct_blur_max_probes_per_group,
        "skipped_probes": skipped,
        "num_rows": len(rows),
        "num_failed": len(failed),
        "pass": not failed,
        "summary_by_check": grouped,
        "rows": rows,
    }


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def write_summary_artifacts(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(summary["rows"])
    json_payload = {key: value for key, value in summary.items() if key != "rows"}
    json_payload["rows"] = rows
    (output_dir / "golden_summary.json").write_text(json.dumps(json_payload, indent=2))

    fieldnames = [
        "check",
        "dim",
        "sys_params_name",
        "theta_fixture",
        "probe_group",
        "probe_name",
        "relative_error",
        "tolerance",
        "pass",
        "likely_drift",
    ]
    with (output_dir / "golden_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def format_failure(row: dict[str, Any]) -> str:
    return (
        f"{row['check']} failed: dim={row['dim']} sys={row['sys_params_name']} "
        f"theta={row['theta_fixture']} probe={row['probe_group']}/{row['probe_name']} "
        f"relative_error={row['relative_error']:.6g} tolerance={row['tolerance']:.6g}; "
        f"likely drift: {row['likely_drift']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dim", action="append", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the diagnostic run, e.g. cpu or cuda:0.",
    )
    parser.add_argument("--include-dim256", action="store_true")
    parser.add_argument("--skip-natural", action="store_true")
    parser.add_argument("--skip-trainable", action="store_true")
    parser.add_argument(
        "--max-probes-per-group",
        type=int,
        default=None,
        help="Optional speed knob. Omit for the full deterministic probe set.",
    )
    parser.add_argument(
        "--direct-blur-max-probes-per-group",
        type=int,
        default=1,
        help=(
            "Limit the expensive reference-backend check that calls blur_ring "
            "directly and recomputes PSFs. Use 0 to cover every probe."
        ),
    )
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    dims = tuple(args.dim) if args.dim else DEFAULT_DIMS
    if args.include_dim256 and OPTIONAL_DIM not in dims:
        dims = tuple(list(dims) + [OPTIONAL_DIM])

    summary = run_golden_suite(
        dims=dims,
        include_natural=not args.skip_natural,
        include_trainable=not args.skip_trainable,
        max_probes_per_group=args.max_probes_per_group,
        direct_blur_max_probes_per_group=args.direct_blur_max_probes_per_group,
        device=torch.device(args.device),
    )
    write_summary_artifacts(summary, args.output_dir)
    print(
        f"[golden] pass={summary['pass']} rows={summary['num_rows']} "
        f"failed={summary['num_failed']} output={args.output_dir}",
        flush=True,
    )
    if summary["num_failed"]:
        for row in summary["rows"]:
            if not row["pass"]:
                print(format_failure(row), flush=True)
    if summary["num_failed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
