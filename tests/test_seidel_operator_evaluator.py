from __future__ import annotations

import math

import numpy as np
import torch

from hybrid_ring_cocoa.evaluation import (
    OperatorProbeConfig,
    apply_seidel_transform,
    check_dataset_twin_invariance,
    evaluate_seidel_recovery,
    validate_hardcoded_transform_wavefronts,
)
from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import (
    _otf_complex_error,
    build_operator_probe_groups,
    coefficient_residuals,
)


SYS_PARAMS = {"NA": 0.45, "lamb": 0.55e-6}


def tiny_probe_config(**overrides) -> OperatorProbeConfig:
    values = {
        "delta_grid_size": 1,
        "radial_basis_count": 0,
        "fourier_frequencies": (),
        "random_count": 0,
        "diagnostic_psf_points": ((0.0, 0.0),),
        "wavefront_field_samples": 7,
        "wavefront_pupil_samples": 31,
        "twin_invariance_tol": 1e-7,
    }
    values.update(overrides)
    return OperatorProbeConfig(**values)


def test_hardcoded_sign_transforms_match_wavefront_definitions():
    validation = validate_hardcoded_transform_wavefronts(grid_size=31)
    assert validation["pass"]
    assert max(validation["max_errors"].values()) < 1e-12


def test_fixed_indices_stay_zero_after_5d_and_4d_transforms():
    theta = np.asarray([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])

    five_d = apply_seidel_transform(theta, "twin_mirror", fixed_indices=[5])
    assert five_d[5] == 0.0
    np.testing.assert_allclose(five_d[:5], [-1.0, 2.0, -3.0, 4.0, -5.0])

    compact_four_d = np.asarray([1.0, -2.0, 3.0, -4.0])
    four_d = apply_seidel_transform(compact_four_d, "mirror_x", fixed_indices=[4, 5])
    np.testing.assert_allclose(four_d, [1.0, 2.0, 3.0, -4.0, 0.0, 0.0])


def test_identity_recovery_reports_zero_operator_and_probe_metadata():
    theta = np.asarray([0.02, -0.015, 0.012, 0.006, 0.004, -0.003])
    config = tiny_probe_config()

    metrics = evaluate_seidel_recovery(
        theta,
        theta,
        dim=16,
        sys_params=SYS_PARAMS,
        fixed_indices=[],
        probe_config=config,
        dataset_twin_invariance_pass=False,
    )

    assert metrics["operator_error_calibrated"] < 1e-12
    assert metrics["operator_error_phys_equiv"] < 1e-12
    assert metrics["operator_error_coord_diagnostic"] < 1e-12
    assert metrics["best_physical_transform"] == "I"
    assert metrics["best_coordinate_diagnostic_transform"] == "I"
    assert metrics["probe_config_hash"] == config.stable_hash()
    assert metrics["probe_group_weights"]["delta_grid"] == 1.0
    assert metrics["aligned_seidel_physical"] == metrics["aligned_seidel_coord_diagnostic"]


def test_global_sign_maps_to_twin_mirror_coordinate_diagnostic():
    theta_gt = np.asarray([0.02, -0.015, 0.012, 0.006, 0.004, -0.003])
    theta_hat = -theta_gt

    metrics = evaluate_seidel_recovery(
        theta_gt,
        theta_hat,
        dim=16,
        sys_params=SYS_PARAMS,
        fixed_indices=[],
        probe_config=tiny_probe_config(),
        dataset_twin_invariance_pass=False,
    )

    assert metrics["best_coordinate_diagnostic_transform"] == "twin_mirror"
    assert metrics["operator_error_coord_diagnostic"] < 1e-12
    assert metrics["physical_transform_set"] == ["I"]
    np.testing.assert_allclose(metrics["aligned_seidel_coord_diagnostic"], theta_gt)


def test_mirror_x_stays_coordinate_diagnostic_not_physical_equivalence():
    theta_gt = np.asarray([0.018, -0.03, 0.007, 0.011, -0.019, 0.004])
    theta_hat = apply_seidel_transform(theta_gt, "mirror_x")

    metrics = evaluate_seidel_recovery(
        theta_gt,
        theta_hat,
        dim=16,
        sys_params=SYS_PARAMS,
        fixed_indices=[],
        probe_config=tiny_probe_config(),
        dataset_twin_invariance_pass=False,
    )

    assert metrics["best_coordinate_diagnostic_transform"] == "mirror_x"
    assert metrics["operator_error_coord_diagnostic"] < 1e-12
    assert metrics["best_physical_transform"] == "I"


def test_twin_gate_requires_dataset_gt_and_hat_invariance():
    config = tiny_probe_config()
    theta_zero = np.zeros(6)
    metrics_allowed = evaluate_seidel_recovery(
        theta_zero,
        theta_zero,
        dim=16,
        sys_params=SYS_PARAMS,
        fixed_indices=[],
        probe_config=config,
        dataset_twin_invariance_pass=True,
    )
    assert metrics_allowed["twin_allowed_for_sample"]
    assert metrics_allowed["physical_transform_set"] == ["I", "twin"]

    theta_gt = np.asarray([0.02, -0.015, 0.012, 0.006, 0.004, -0.003])
    theta_hat = apply_seidel_transform(theta_gt, "twin")
    metrics_blocked = evaluate_seidel_recovery(
        theta_gt,
        theta_hat,
        dim=16,
        sys_params=SYS_PARAMS,
        fixed_indices=[],
        probe_config=config,
        dataset_twin_invariance_pass=False,
    )
    assert not metrics_blocked["twin_allowed_for_sample"]
    assert metrics_blocked["physical_transform_set"] == ["I"]
    assert "twin_invariance_error_hat" in metrics_blocked
    assert "twin_invariance_pass_hat" in metrics_blocked


def test_dataset_level_twin_gate_is_deterministic_and_records_probe_config():
    config = tiny_probe_config(random_seed=99)
    first = check_dataset_twin_invariance(
        dim=16,
        sys_params=SYS_PARAMS,
        fixed_indices=[5],
        probe_config=config,
        num_samples=2,
        random_seed=123,
        theta_scale=0.02,
    )
    second = check_dataset_twin_invariance(
        dim=16,
        sys_params=SYS_PARAMS,
        fixed_indices=[5],
        probe_config=config,
        num_samples=2,
        random_seed=123,
        theta_scale=0.02,
    )

    assert first["dataset_twin_invariance_errors"] == second["dataset_twin_invariance_errors"]
    assert first["probe_config_hash"] == config.stable_hash()
    assert first["probe_group_weights"]["delta_grid"] == 1.0


def test_epsilon_safe_coefficient_diagnostics_are_finite_for_zero_gt():
    absolute, relative = coefficient_residuals(np.zeros(6), np.full(6, 1e-15), eps=1e-12)
    assert absolute > 0.0
    assert math.isfinite(relative)
    assert relative < 0.01


def test_probe_config_hash_and_full_delta_probe_count_are_stable():
    base = tiny_probe_config()
    changed = tiny_probe_config(delta_grid_size=2)
    assert base.stable_hash() != changed.stable_hash()

    full_delta = OperatorProbeConfig(full_delta_basis=True)
    groups = build_operator_probe_groups(4, full_delta, device=torch.device("cpu"))
    assert set(groups) == {"full_delta"}
    assert len(groups["full_delta"]) == 16


def test_complex_otf_diagnostic_is_zero_for_identical_psf_stack():
    theta = np.asarray([0.02, -0.015, 0.012, 0.006, 0.004, -0.003])
    error = _otf_complex_error(
        theta,
        theta,
        dim=16,
        sys_params=SYS_PARAMS,
        config=tiny_probe_config(),
        device=torch.device("cpu"),
    )
    assert error < 1e-12


def test_no_recenter_or_refocus_hides_w311_or_defocus_residuals():
    theta_gt = np.asarray([0.0, 0.0, 0.0, 0.0, 0.035, -0.025])
    theta_hat = np.zeros(6)

    metrics = evaluate_seidel_recovery(
        theta_gt,
        theta_hat,
        dim=16,
        sys_params=SYS_PARAMS,
        fixed_indices=[],
        probe_config=tiny_probe_config(),
        dataset_twin_invariance_pass=False,
    )

    assert metrics["wavefront_error_calibrated"] > 0.0
    assert metrics["operator_error_calibrated"] > 0.0
    assert metrics["aligned_wavefront_error_physical"] > 0.0
