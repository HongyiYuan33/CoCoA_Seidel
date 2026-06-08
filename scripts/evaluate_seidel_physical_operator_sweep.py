"""Augment Seidel sweep CSVs with gauge-aware operator recovery metrics.

This is the primary gauge-aware full operator evaluator for Seidel recovery
sweeps. In addition to calibrated/strict exact ring-convolution operator
metrics, it reports physical-canonical and gauge-canonical transforms,
operator errors, recovered RMS ratios, sign agreement, and twin gating columns.
Strict-only metrics from lightweight blind sweeps should not be treated as the
final physical-equivalence score.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_ring_cocoa.evaluation import (  # noqa: E402
    OperatorProbeConfig,
    check_dataset_twin_invariance,
    evaluate_seidel_recovery,
    validate_hardcoded_transform_wavefronts,
)
from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import (  # noqa: E402
    check_trace_dataset_twin_invariance,
    evaluate_trace_seidel_recovery,
)


DEFAULT_SYS_PARAMS = {"NA": 0.45, "lamb": 0.55e-6}
CLASSICAL_FIXED_INDICES = {
    "classical4d": [4, 5],
    "classical5d": [5],
    "classical6d": [],
    "backend6": [],
}
TRACE_THETA_DIMS = {"trace5": 5, "trace4": 4, "trace3": 3}


def parse_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict, tuple, int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def parse_float_vector(value: Any) -> list[float]:
    parsed = parse_jsonish(value)
    if parsed is None:
        raise ValueError("Missing Seidel vector")
    return [float(item) for item in parsed]


def parse_fixed_indices(value: Any, fallback: Sequence[int] | None = None) -> list[int]:
    if value is None or str(value).strip() == "":
        return [int(idx) for idx in (fallback or [])]
    parsed = parse_jsonish(value)
    if parsed is None:
        return [int(idx) for idx in (fallback or [])]
    return [int(idx) for idx in parsed]


def parse_boolish(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    if text in {"auto", ""}:
        return None
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, dict, tuple)):
                    value = json.dumps(value, separators=(",", ":"))
                out[key] = value
            writer.writerow(out)


def make_probe_config(args: argparse.Namespace) -> OperatorProbeConfig:
    frequencies = []
    for spec in args.fourier_frequency:
        try:
            kx_text, ky_text = spec.split(",", maxsplit=1)
            frequencies.append((int(kx_text), int(ky_text)))
        except ValueError as exc:
            raise ValueError(
                f"Fourier frequency must be 'kx,ky', got {spec!r}"
            ) from exc

    group_weights = {
        "delta_grid": float(args.delta_weight),
        "radial_basis": float(args.radial_weight),
        "fourier": float(args.fourier_weight),
        "random": float(args.random_weight),
        "full_delta": float(args.full_delta_weight),
    }
    return OperatorProbeConfig(
        delta_grid_size=int(args.delta_grid_size),
        radial_basis_count=int(args.radial_basis_count),
        fourier_frequencies=tuple(frequencies),
        random_count=int(args.random_count),
        random_seed=int(args.random_seed),
        group_weights=group_weights,
        full_delta_basis=bool(args.full_delta_basis),
        twin_invariance_tol=float(args.twin_invariance_tol),
        wavefront_field_samples=int(args.wavefront_field_samples),
        wavefront_pupil_samples=int(args.wavefront_pupil_samples),
    )


def infer_fixed_indices(row: dict[str, Any], fallback: Sequence[int]) -> list[int]:
    fixed = parse_fixed_indices(row.get("fixed_seidel_indices"), fallback=fallback)
    if fixed:
        return fixed
    no_w311 = parse_boolish(row.get("no_w311_no_defocus"))
    no_defocus = parse_boolish(row.get("no_defocus"))
    inferred: list[int] = []
    if no_w311:
        inferred.extend([4, 5])
    elif no_defocus:
        inferred.append(5)
    return sorted(set(inferred))


def infer_dataset_fixed_indices(rows: Sequence[dict[str, Any]], fallback: Sequence[int]) -> list[int]:
    if fallback:
        return [int(idx) for idx in fallback]
    inferred_sets = {tuple(infer_fixed_indices(row, [])) for row in rows}
    if not inferred_sets:
        return []
    if len(inferred_sets) == 1:
        return list(next(iter(inferred_sets)))
    raise ValueError(
        "CSV contains multiple fixed-index sets. Pass --fixed-indices explicitly "
        "or evaluate each sweep dimensionality separately."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--sys-params-json", default=json.dumps(DEFAULT_SYS_PARAMS))
    parser.add_argument(
        "--theta-convention",
        choices=["backend6", "classical4d", "classical5d", "classical6d"],
        default="backend6",
        help=(
            "How to parse seidel_gt/seidel_final vectors. Classical backend "
            "choices are classical4d (fixed W311,Wd), classical5d (fixed Wd), "
            "classical6d/backend6 (full [W040,W131,W222,W220,W311,Wd]). "
            "Trace-separated CLI modes are paused."
        ),
    )
    parser.add_argument("--fixed-indices", nargs="*", type=int, default=[])
    parser.add_argument(
        "--dataset-twin-invariance-pass",
        default="auto",
        help="'auto', 'true', or 'false'. Auto runs the dataset-level random-theta gate.",
    )
    parser.add_argument("--dataset-twin-samples", type=int, default=8)
    parser.add_argument("--dataset-twin-seed", type=int, default=314159)
    parser.add_argument("--dataset-twin-theta-scale", type=float, default=0.15)
    parser.add_argument("--twin-invariance-tol", type=float, default=1e-5)
    parser.add_argument("--delta-grid-size", type=int, default=3)
    parser.add_argument("--radial-basis-count", type=int, default=4)
    parser.add_argument("--fourier-frequency", action="append", default=["1,0", "0,1", "1,1", "2,0", "0,2"])
    parser.add_argument("--random-count", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument("--full-delta-basis", action="store_true")
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--radial-weight", type=float, default=1.0)
    parser.add_argument("--fourier-weight", type=float, default=1.0)
    parser.add_argument("--random-weight", type=float, default=1.0)
    parser.add_argument("--full-delta-weight", type=float, default=1.0)
    parser.add_argument("--wavefront-field-samples", type=int, default=41)
    parser.add_argument("--wavefront-pupil-samples", type=int, default=151)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--row-start", type=int, default=0, help="0-based inclusive input row start.")
    parser.add_argument("--row-end", type=int, default=None, help="0-based exclusive input row end.")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in the output CSV.")
    args = parser.parse_args()

    probe_config = make_probe_config(args)
    sys_params = parse_jsonish(args.sys_params_json) or DEFAULT_SYS_PARAMS
    transform_validation = validate_hardcoded_transform_wavefronts()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = list(csv.DictReader(args.input_csv.open()))
    indexed_rows = list(enumerate(all_rows))
    if args.row_start:
        indexed_rows = [(idx, row) for idx, row in indexed_rows if idx >= int(args.row_start)]
    if args.row_end is not None:
        indexed_rows = [(idx, row) for idx, row in indexed_rows if idx < int(args.row_end)]
    if args.limit is not None:
        indexed_rows = indexed_rows[: int(args.limit)]
    rows = [row for _, row in indexed_rows]
    if args.theta_convention in CLASSICAL_FIXED_INDICES:
        fallback_fixed = (
            [int(idx) for idx in args.fixed_indices]
            if args.fixed_indices
            else list(CLASSICAL_FIXED_INDICES[args.theta_convention])
        )
        dataset_fixed_indices = infer_dataset_fixed_indices(rows, fallback_fixed)
    else:
        dataset_fixed_indices = []

    requested_dataset_gate = parse_boolish(args.dataset_twin_invariance_pass)
    dataset_gate_report: dict[str, Any] | None = None
    if requested_dataset_gate is None:
        if args.theta_convention in CLASSICAL_FIXED_INDICES:
            dataset_gate_report = check_dataset_twin_invariance(
                dim=int(args.dim),
                sys_params=sys_params,
                fixed_indices=dataset_fixed_indices,
                probe_config=probe_config,
                num_samples=int(args.dataset_twin_samples),
                random_seed=int(args.dataset_twin_seed),
                theta_scale=float(args.dataset_twin_theta_scale),
            )
        else:
            model_dim = TRACE_THETA_DIMS[args.theta_convention]
            dataset_gate_report = check_trace_dataset_twin_invariance(
                dim=int(args.dim),
                sys_params=sys_params,
                model_dim=model_dim,
                probe_config=probe_config,
                num_samples=int(args.dataset_twin_samples),
                random_seed=int(args.dataset_twin_seed),
                theta_scale=float(args.dataset_twin_theta_scale),
            )
        dataset_twin_pass = bool(dataset_gate_report["dataset_twin_invariance_pass"])
    else:
        dataset_twin_pass = bool(requested_dataset_gate)

    output_csv = args.output_dir / "seidel_physical_operator_metrics.csv"
    augmented: list[dict[str, Any]] = []
    completed_source_indices: set[int] = set()
    if args.resume and output_csv.is_file():
        augmented = list(csv.DictReader(output_csv.open()))
        for existing in augmented:
            if existing.get("_source_row_index") not in (None, ""):
                completed_source_indices.add(int(existing["_source_row_index"]))

    for local_idx, (source_idx, row) in enumerate(indexed_rows, start=1):
        if source_idx in completed_source_indices:
            print(
                f"[{local_idx}/{len(indexed_rows)}] skip row={source_idx} "
                f"{row.get('image','?')} {row.get('candidate_id','?')}",
                flush=True,
            )
            continue
        theta_gt = parse_float_vector(row["seidel_gt"])
        theta_hat = parse_float_vector(row["seidel_final"])
        if args.theta_convention in CLASSICAL_FIXED_INDICES:
            fixed = infer_fixed_indices(row, args.fixed_indices)
            if not fixed and args.theta_convention != "backend6":
                fixed = list(CLASSICAL_FIXED_INDICES[args.theta_convention])
            metrics = evaluate_seidel_recovery(
                theta_gt=theta_gt,
                theta_hat=theta_hat,
                dim=int(args.dim),
                sys_params=sys_params,
                fixed_indices=fixed,
                probe_config=probe_config,
                dataset_twin_invariance_pass=dataset_twin_pass,
            )
        else:
            model_dim = TRACE_THETA_DIMS[args.theta_convention]
            metrics = evaluate_trace_seidel_recovery(
                theta_trace_gt=theta_gt,
                theta_trace_hat=theta_hat,
                dim=int(args.dim),
                sys_params=sys_params,
                model_dim=model_dim,
                probe_config=probe_config,
                dataset_twin_invariance_pass=dataset_twin_pass,
            )
        merged = dict(row)
        merged["_source_row_index"] = int(source_idx)
        merged.update(metrics)
        augmented.append(merged)
        print(
            f"[{local_idx}/{len(indexed_rows)}] row={source_idx} "
            f"{row.get('image','?')} {row.get('candidate_id','?')} "
            f"operator_calibrated={metrics['operator_error_calibrated']:.6g} "
            f"phys={metrics['operator_error_phys_equiv']:.6g} "
            f"coord={metrics['operator_error_coord_diagnostic']:.6g}",
            flush=True,
        )
        if args.resume:
            write_csv(augmented, output_csv)

    write_csv(augmented, output_csv)
    summary = {
        "input_csv": str(args.input_csv),
        "output_csv": str(output_csv),
        "num_rows": len(augmented),
        "input_num_rows": len(all_rows),
        "selected_num_rows": len(indexed_rows),
        "row_start": int(args.row_start),
        "row_end": None if args.row_end is None else int(args.row_end),
        "dim": int(args.dim),
        "sys_params": sys_params,
        "theta_convention": args.theta_convention,
        "fixed_indices_fallback": [int(idx) for idx in args.fixed_indices],
        "fixed_indices_dataset_gate": [int(idx) for idx in dataset_fixed_indices],
        "dataset_twin_invariance_pass": bool(dataset_twin_pass),
        "dataset_twin_invariance_report": dataset_gate_report,
        "transform_wavefront_validation": transform_validation,
        "probe_config_hash": probe_config.stable_hash(),
        "probe_config": probe_config.to_dict(),
        "probe_group_weights": probe_config.resolved_group_weights(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Seidel Gauge-Aware Physical-Operator Evaluation",
                "",
                f"- Input CSV: `{args.input_csv}`",
                f"- Output CSV: `{output_csv}`",
                f"- Rows: {len(augmented)}",
                f"- Probe config hash: `{probe_config.stable_hash()}`",
                f"- Dataset twin invariance pass: `{dataset_twin_pass}`",
                f"- Transform wavefront validation pass: `{transform_validation['pass']}`",
                "",
                "The primary metric is the exact ring-convolution forward operator "
                "evaluated on deterministic probes. PSF and complex OTF stack metrics "
                "are diagnostic only.",
                "",
                "This primary evaluator reports calibrated/strict, physical-canonical, "
                "and gauge-canonical operator errors plus best transforms, sign "
                "agreement, recovered/GT RMS ratios, and twin gating columns. Sign "
                "tables should use `canonical_sign_source=gauge` by default, while raw "
                "and physical-canonical columns remain available for control checks.",
                "",
                "Gauge v1 uses the existing hard-coded discrete operator transforms: "
                "`identity -> I`, `x_reflection -> mirror_x`, `y_reflection -> I`, "
                "`rot180 -> mirror_x`, `phase_conjugate_twin -> twin`, and "
                "`phase_conjugate_twin_mirror -> twin_mirror`. Continuous "
                "image-space distortion warp, per-field recentering, and per-field "
                "refocus are not enabled in the current forward model.",
                "",
                "Classical backend conventions are the default analysis path:",
                "",
                "- `classical4d`: active backend `[W040,W131,W222,W220]`, "
                "`fixed_seidel_indices=[4,5]`.",
                "- `classical5d`: active backend `[W040,W131,W222,W220,W311]`, "
                "`fixed_seidel_indices=[5]`.",
                "- `classical6d` / `backend6`: full backend "
                "`[W040,W131,W222,W220,W311,Wd]`.",
                "",
                "Trace-separated conventions are paused and kept for explicit "
                "reproduction only:",
                "",
                "- `trace5`: public `[S,C,A,F,D]`, backend `[S,C,2A,F-A,D,0]`, "
                "`fixed_seidel_indices=[5]`, `no_defocus=true`, "
                "`no_w311_no_defocus=false`, `distortion_forward_model=frozen_backend_W311`, "
                "`distortion_warp=false`, `per_field_recenter=false`.",
                "- `trace4`: public `[S,C,A,F]`, backend `[S,C,2A,F-A,0,0]`, "
                "`fixed_seidel_indices=[4,5]`; intentionally misspecified for "
                "D-containing ground truth.",
                "- `trace3`: public `[S,C,A]`, backend `[S,C,2A,-A,0,0]`, "
                "`fixed_seidel_indices=[4,5]`, public `F=0`; intentionally "
                "misspecified when `F != 0` or `D != 0`.",
                "",
            ]
        )
    )
    print(f"[done] wrote {output_csv}")


if __name__ == "__main__":
    main()
