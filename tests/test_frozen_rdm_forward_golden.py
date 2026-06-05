from __future__ import annotations

import os

import pytest
import torch

from scripts.run_frozen_rdm_forward_golden import (
    DEFAULT_DIMS,
    OPTIONAL_DIM,
    format_failure,
    run_golden_suite,
    write_summary_artifacts,
)


@pytest.fixture(scope="session")
def golden_summary():
    dims = list(DEFAULT_DIMS)
    if os.environ.get("RUN_GOLDEN_DIM256") == "1":
        dims.append(OPTIONAL_DIM)
    device = torch.device(os.environ.get("GOLDEN_DEVICE", "cpu"))
    return run_golden_suite(
        dims=tuple(dims),
        include_natural=True,
        include_trainable=True,
        max_probes_per_group=1,
        device=device,
    )


def test_frozen_rdm_forward_golden_suite_passes(golden_summary):
    failures = [row for row in golden_summary["rows"] if not row["pass"]]
    assert not failures, "\n".join(format_failure(row) for row in failures[:20])
    assert set(DEFAULT_DIMS).issubset(set(golden_summary["dims"]))
    assert golden_summary["num_rows"] > 0


def test_golden_suite_covers_required_checks(golden_summary):
    checks = {row["check"] for row in golden_summary["rows"]}
    required = {
        "reference_backend_consistency",
        "trace_wrapper_equivalence",
        "backend_reference_vs_trainable_psf",
        "backend_reference_vs_trainable_operator",
        "trace_reference_vs_trainable_psf",
        "trace_reference_vs_trainable_operator",
        "backend_operator_identity",
        "trace_operator_identity_operator_error_strict",
        "trace_operator_identity_operator_error_phys_equiv",
        "trace_operator_identity_operator_error_coord_diagnostic",
    }
    assert required.issubset(checks)


def test_golden_suite_covers_required_fixtures_and_probe_groups(golden_summary):
    theta_names = {row["theta_fixture"] for row in golden_summary["rows"]}
    assert {
        "zero",
        "trace5_no_defocus_with_distortion_expanded",
        "trace4_normal_expanded",
        "trace3_normal_expanded",
        "legacy_backend6_with_defocus",
        "legacy_backend6_with_distortion",
        "trace5_no_defocus_with_distortion",
        "trace4_normal",
        "trace3_normal",
    }.issubset(theta_names)

    probe_groups = {row["probe_group"] for row in golden_summary["rows"]}
    assert {"delta_grid", "radial_basis", "fourier", "random"}.issubset(probe_groups)
    if "natural_image" not in probe_groups:
        pytest.skip(f"natural image probe unavailable: {golden_summary['skipped_probes']}")


def test_golden_artifact_writer_outputs_json_and_csv(golden_summary, tmp_path):
    write_summary_artifacts(golden_summary, tmp_path)
    json_path = tmp_path / "golden_summary.json"
    csv_path = tmp_path / "golden_metrics.csv"
    assert json_path.is_file()
    assert csv_path.is_file()
    assert "psfs" not in json_path.read_text().lower()
    assert "relative_error" in csv_path.read_text().splitlines()[0]
