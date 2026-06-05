from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hybrid_ring_cocoa.evaluation import OperatorProbeConfig
from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import (
    _otf_complex_error,
    apply_trace_transform,
    evaluate_trace_seidel_recovery,
)
from hybrid_ring_cocoa.optics.ring_forward import (
    blur_ring,
    blur_ring_trace,
    blur_ring_with_psfs,
)
from hybrid_ring_cocoa.optics.seidel_psf import (
    TRACE5_COEFF_NAMES,
    compress_trace_seidel,
    expand_trace_seidel,
    get_reference_trace_ring_psfs,
    get_trainable_trace_ring_psfs,
    trace_seidel_wavefront,
)


SYS_PARAMS = {"NA": 0.45, "lamb": 0.55e-6}


def tiny_probe_config(**overrides) -> OperatorProbeConfig:
    values = {
        "delta_grid_size": 1,
        "radial_basis_count": 1,
        "fourier_frequencies": ((1, 0),),
        "random_count": 1,
        "random_seed": 123,
        "diagnostic_psf_points": ((0.0, 0.0), (0.5, -0.5)),
        "wavefront_field_samples": 7,
        "wavefront_pupil_samples": 31,
        "twin_invariance_tol": 1e-7,
    }
    values.update(overrides)
    return OperatorProbeConfig(**values)


def backend_wavefront(theta_backend6, field_x, field_y, pupil_x, pupil_y):
    theta = np.asarray(theta_backend6, dtype=np.float64)
    hx = np.asarray(field_x, dtype=np.float64)
    hy = np.asarray(field_y, dtype=np.float64)
    px = np.asarray(pupil_x, dtype=np.float64)
    py = np.asarray(pupil_y, dtype=np.float64)
    h2 = hx * hx + hy * hy
    h = np.sqrt(h2)
    rho2 = px * px + py * py
    h_dot_rho = hx * px + hy * py
    x_rot = np.divide(h_dot_rho, h, out=np.zeros_like(h_dot_rho), where=h > 0)
    return (
        theta[0] * rho2 * rho2
        + theta[1] * h * rho2 * x_rot
        + theta[2] * h2 * x_rot * x_rot
        + theta[3] * h2 * rho2
        + theta[4] * h2 * h * x_rot
        + theta[5] * rho2
    )


def test_expand_compress_trace_round_trips_and_constraints():
    assert TRACE5_COEFF_NAMES == ("S", "C", "A", "F", "D")
    theta5 = np.asarray([0.30, -0.10, 0.05, 0.08, 0.04])
    backend5 = expand_trace_seidel(theta5, model_dim=5)
    np.testing.assert_allclose(backend5, [0.30, -0.10, 0.10, 0.03, 0.04, 0.0])
    np.testing.assert_allclose(compress_trace_seidel(backend5, model_dim=5), theta5)
    np.testing.assert_allclose(expand_trace_seidel(theta5, model_dim="trace5_no_defocus"), backend5)
    np.testing.assert_allclose(expand_trace_seidel(theta5, model_dim="no_defocus_5d"), backend5)

    theta4 = np.asarray([0.1, -0.03, 0.02, 0.07])
    backend4 = expand_trace_seidel(theta4, model_dim=4)
    np.testing.assert_allclose(backend4, [0.1, -0.03, 0.04, 0.05, 0.0, 0.0])
    np.testing.assert_allclose(compress_trace_seidel(backend4, model_dim=4), theta4)

    theta3 = np.asarray([0.1, -0.03, 0.02])
    backend3 = expand_trace_seidel(theta3, model_dim=3)
    np.testing.assert_allclose(backend3, [0.1, -0.03, 0.04, -0.02, 0.0, 0.0])
    np.testing.assert_allclose(compress_trace_seidel(backend3, model_dim=3), theta3)

    with pytest.raises(ValueError, match="W311"):
        compress_trace_seidel([0, 0, 0, 0, 1e-4, 0], model_dim=4, atol=1e-8)
    np.testing.assert_allclose(
        compress_trace_seidel([0, 0, 0, 0, 1e-4, 0], model_dim=5, atol=1e-8),
        [0, 0, 0, 0, 1e-4],
    )
    with pytest.raises(ValueError, match="Wd"):
        compress_trace_seidel([0, 0, 0, 0, 0, 1e-4], model_dim=4, atol=1e-8)
    with pytest.raises(ValueError, match="Wd"):
        compress_trace_seidel([0, 0, 0, 0, 1e-4, 1e-4], model_dim=5, atol=1e-8)
    with pytest.raises(ValueError, match="public F"):
        compress_trace_seidel([0, 0, 0.1, 0.0, 0.0, 0.0], model_dim=3, atol=1e-8)


def test_public_vs_backend_wavefront_equivalence_random_points():
    rng = np.random.default_rng(7)
    for model_dim in (3, 4, 5):
        for _ in range(50):
            theta = rng.normal(0.0, 0.1, size=model_dim)
            h = rng.uniform(0.0, 1.0)
            psi = rng.uniform(-math.pi, math.pi)
            rho = rng.uniform(0.0, 1.0)
            phi = rng.uniform(-math.pi, math.pi)
            hx, hy = h * math.cos(psi), h * math.sin(psi)
            px, py = rho * math.cos(phi), rho * math.sin(phi)
            public = trace_seidel_wavefront(theta, hx, hy, px, py, model_dim=model_dim)
            backend = backend_wavefront(
                expand_trace_seidel(theta, model_dim=model_dim),
                hx,
                hy,
                px,
                py,
            )
            np.testing.assert_allclose(public, backend, atol=1e-12, rtol=1e-12)


def test_factor_of_two_public_astigmatism_convention():
    rng = np.random.default_rng(11)
    for _ in range(40):
        a = rng.normal()
        h = rng.uniform(0.05, 1.0)
        rho = rng.uniform(0.05, 1.0)
        psi = rng.uniform(-math.pi, math.pi)
        phi = rng.uniform(-math.pi, math.pi)
        hx, hy = h * math.cos(psi), h * math.sin(psi)
        px, py = rho * math.cos(phi), rho * math.sin(phi)
        expected = a * h * h * rho * rho * math.cos(2.0 * (phi - psi))
        got = trace_seidel_wavefront([0.0, 0.0, a, 0.0], hx, hy, px, py, model_dim=4)
        np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_pure_astigmatism_has_no_scalar_trace_leakage():
    a = 0.37
    h = 0.8
    rho = 0.9
    angles = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    values = trace_seidel_wavefront(
        [0.0, 0.0, a, 0.0],
        h,
        0.0,
        rho * np.cos(angles),
        rho * np.sin(angles),
        model_dim=4,
    )
    assert abs(float(np.mean(values))) < 1e-14

    classical_direct = backend_wavefront(
        [0.0, 0.0, 2.0 * a, 0.0, 0.0, 0.0],
        h,
        0.0,
        rho * np.cos(angles),
        rho * np.sin(angles),
    )
    assert float(np.mean(classical_direct)) > 0.1


def test_term_isolation_and_opposite_field_parity():
    px = np.asarray([0.2, -0.4, 0.7])
    py = np.asarray([0.6, 0.3, -0.1])
    hx, hy = 0.5, -0.25
    even_theta = [0.1, 0.0, -0.04, 0.03]
    odd_theta = [0.0, 0.08, 0.0, 0.0]

    even_plus = trace_seidel_wavefront(even_theta, hx, hy, px, py, model_dim=4)
    even_minus = trace_seidel_wavefront(even_theta, -hx, -hy, px, py, model_dim=4)
    odd_plus = trace_seidel_wavefront(odd_theta, hx, hy, px, py, model_dim=4)
    odd_minus = trace_seidel_wavefront(odd_theta, -hx, -hy, px, py, model_dim=4)

    np.testing.assert_allclose(even_plus - even_minus, 0.0, atol=1e-12)
    np.testing.assert_allclose(odd_plus + odd_minus, 0.0, atol=1e-12)

    s_center = trace_seidel_wavefront([0.1, 0.2, 0.3, 0.4], 0.0, 0.0, px, py, model_dim=4)
    s_only = trace_seidel_wavefront([0.1, 0.0, 0.0, 0.0], 0.0, 0.0, px, py, model_dim=4)
    np.testing.assert_allclose(s_center, s_only, atol=1e-12)


def test_reference_and_trainable_psfs_and_operator_match_for_trace_models():
    dim = 16
    device = torch.device("cpu")
    probes = [
        torch.eye(dim, dtype=torch.float32, device=device),
        torch.arange(dim * dim, dtype=torch.float32, device=device).reshape(dim, dim) / (dim * dim),
    ]
    cases = [
        (5, np.asarray([0.02, -0.015, 0.01, 0.006, 0.012])),
        (4, np.asarray([0.02, -0.015, 0.01, 0.006])),
        (3, np.asarray([0.02, -0.015, 0.01])),
    ]
    for model_dim, theta in cases:
        ref = get_reference_trace_ring_psfs(
            theta,
            dim,
            SYS_PARAMS,
            model_dim=model_dim,
            device=device,
        )
        trainable = get_trainable_trace_ring_psfs(
            torch.tensor(theta, dtype=torch.float32, device=device),
            dim,
            SYS_PARAMS,
            model_dim=model_dim,
            device=device,
        )
        rel = torch.linalg.vector_norm(ref - trainable) / torch.clamp(
            torch.linalg.vector_norm(ref), min=1e-12
        )
        assert float(rel) < 1e-5
        for probe in probes:
            out_ref = blur_ring_with_psfs(probe, ref)
            out_trainable = blur_ring_with_psfs(probe, trainable)
            torch.testing.assert_close(out_trainable, out_ref, rtol=1e-5, atol=1e-6)


def test_trace_wrapper_matches_backend_forward_golden_outputs():
    dim = 16
    image = torch.linspace(0.0, 1.0, dim * dim, dtype=torch.float32).reshape(dim, dim)
    theta5 = np.asarray([0.30, -0.10, 0.05, 0.08, 0.04])
    backend5 = expand_trace_seidel(theta5, model_dim=5)
    np.testing.assert_allclose(backend5, [0.30, -0.10, 0.10, 0.03, 0.04, 0.0])
    direct5 = blur_ring(image, backend5, SYS_PARAMS)
    wrapped5 = blur_ring_trace(image, theta5, SYS_PARAMS, model_dim=5)
    torch.testing.assert_close(wrapped5, direct5, rtol=0.0, atol=0.0)

    theta4 = np.asarray([0.02, -0.01, 0.008, 0.004])
    backend = expand_trace_seidel(theta4, model_dim=4)
    direct = blur_ring(image, backend, SYS_PARAMS)
    wrapped = blur_ring_trace(image, theta4, SYS_PARAMS, model_dim=4)
    torch.testing.assert_close(wrapped, direct, rtol=0.0, atol=0.0)


def test_trace_identity_no_refocus_and_no_recenter_operator_metrics():
    config = tiny_probe_config()
    theta4 = np.asarray([0.02, -0.015, 0.01, 0.006])
    identity = evaluate_trace_seidel_recovery(
        theta4,
        theta4,
        dim=16,
        sys_params=SYS_PARAMS,
        model_dim=4,
        probe_config=config,
        dataset_twin_invariance_pass=False,
    )
    assert identity["operator_error_strict"] < 1e-12
    assert identity["physical_transform_set"] == ["I"]
    assert identity["eta_1"] == "not_applicable"

    theta5 = np.asarray([0.02, -0.015, 0.01, 0.006, 0.012])
    identity5 = evaluate_trace_seidel_recovery(
        theta5,
        theta5,
        dim=16,
        sys_params=SYS_PARAMS,
        model_dim=5,
        probe_config=config,
        dataset_twin_invariance_pass=False,
    )
    assert identity5["operator_error_strict"] < 1e-12
    assert identity5["theta_convention"] == "trace5"
    assert identity5["fixed_seidel_indices"] == [5]
    assert identity5["no_defocus"] is True
    assert identity5["no_w311_no_defocus"] is False
    assert identity5["theta_backend6_hat"][4] != 0.0
    assert identity5["theta_backend6_hat"][5] == 0.0

    no_refocus = evaluate_trace_seidel_recovery(
        [0.0, 0.0, 0.0, 0.03],
        [0.0, 0.0, 0.0, 0.0],
        dim=16,
        sys_params=SYS_PARAMS,
        model_dim=4,
        probe_config=config,
        dataset_twin_invariance_pass=False,
    )
    assert no_refocus["operator_error_strict"] > 0.0

    no_recenter = evaluate_trace_seidel_recovery(
        [0.0, 0.03, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        dim=16,
        sys_params=SYS_PARAMS,
        model_dim=4,
        probe_config=config,
        dataset_twin_invariance_pass=False,
    )
    assert no_recenter["operator_error_strict"] > 0.0

    wrong_distortion = evaluate_trace_seidel_recovery(
        [0.0, 0.0, 0.0, 0.0, 0.03],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        dim=16,
        sys_params=SYS_PARAMS,
        model_dim=5,
        probe_config=config,
        dataset_twin_invariance_pass=False,
    )
    assert wrong_distortion["operator_error_strict"] > 0.0

    theta3 = np.asarray([0.02, -0.015, 0.01])
    identity3 = evaluate_trace_seidel_recovery(
        theta3,
        theta3,
        dim=16,
        sys_params=SYS_PARAMS,
        model_dim=3,
        probe_config=config,
        dataset_twin_invariance_pass=False,
    )
    assert identity3["operator_error_strict"] < 1e-12


def test_complex_otf_trace_diagnostic_detects_coma_sign_change():
    config = tiny_probe_config()
    gt = expand_trace_seidel([0.0, 0.03, 0.0, 0.0], model_dim=4)
    cand = expand_trace_seidel([0.0, -0.03, 0.0, 0.0], model_dim=4)
    error = _otf_complex_error(
        gt,
        cand,
        dim=16,
        sys_params=SYS_PARAMS,
        config=config,
        device=torch.device("cpu"),
    )
    assert error > 1e-6


def test_analytic_mirror_and_twin_public_conventions():
    theta4 = np.asarray([0.03, -0.02, 0.015, 0.006])
    x = np.linspace(-0.8, 0.8, 9)
    y = np.linspace(-0.7, 0.7, 9)
    px, py = np.meshgrid(x, y, indexing="xy")
    hx, hy = 0.4, 0.0
    base = trace_seidel_wavefront(theta4, hx, hy, px, py, model_dim=4)

    mirror = trace_seidel_wavefront(
        apply_trace_transform(theta4, "mirror_x", model_dim=4),
        hx,
        hy,
        px,
        py,
        model_dim=4,
    )
    mirror_expected = trace_seidel_wavefront(theta4, hx, hy, -px, py, model_dim=4)
    np.testing.assert_allclose(mirror, mirror_expected, atol=1e-12)

    twin = trace_seidel_wavefront(
        apply_trace_transform(theta4, "twin", model_dim=4),
        hx,
        hy,
        px,
        py,
        model_dim=4,
    )
    twin_expected = -trace_seidel_wavefront(theta4, hx, hy, -px, -py, model_dim=4)
    np.testing.assert_allclose(twin, twin_expected, atol=1e-12)

    twin_mirror = trace_seidel_wavefront(
        apply_trace_transform(theta4, "twin_mirror", model_dim=4),
        hx,
        hy,
        px,
        py,
        model_dim=4,
    )
    np.testing.assert_allclose(twin_mirror, -base, atol=1e-12)

    theta5 = np.asarray([0.03, -0.02, 0.015, 0.006, 0.011])
    expected_signs = {
        "I": [1, 1, 1, 1, 1],
        "mirror_x": [1, -1, 1, 1, -1],
        "twin": [-1, 1, -1, -1, 1],
        "twin_mirror": [-1, -1, -1, -1, -1],
    }
    for transform, signs in expected_signs.items():
        np.testing.assert_allclose(
            apply_trace_transform(theta5, transform, model_dim=5),
            theta5 * np.asarray(signs),
        )

    base5 = trace_seidel_wavefront(theta5, hx, hy, px, py, model_dim=5)
    mirror5 = trace_seidel_wavefront(
        apply_trace_transform(theta5, "mirror_x", model_dim=5),
        hx,
        hy,
        px,
        py,
        model_dim=5,
    )
    mirror5_expected = trace_seidel_wavefront(theta5, hx, hy, -px, py, model_dim=5)
    np.testing.assert_allclose(mirror5, mirror5_expected, atol=1e-12)

    twin5 = trace_seidel_wavefront(
        apply_trace_transform(theta5, "twin", model_dim=5),
        hx,
        hy,
        px,
        py,
        model_dim=5,
    )
    twin5_expected = -trace_seidel_wavefront(theta5, hx, hy, -px, -py, model_dim=5)
    np.testing.assert_allclose(twin5, twin5_expected, atol=1e-12)

    twin_mirror5 = trace_seidel_wavefront(
        apply_trace_transform(theta5, "twin_mirror", model_dim=5),
        hx,
        hy,
        px,
        py,
        model_dim=5,
    )
    np.testing.assert_allclose(twin_mirror5, -base5, atol=1e-12)


def test_mirror_and_twin_operator_transform_policy_for_trace():
    config = tiny_probe_config()
    theta4 = np.asarray([0.02, -0.015, 0.01, 0.006])
    mirrored = apply_trace_transform(theta4, "mirror_x", model_dim=4)
    metrics = evaluate_trace_seidel_recovery(
        theta4,
        mirrored,
        dim=16,
        sys_params=SYS_PARAMS,
        model_dim=4,
        probe_config=config,
        dataset_twin_invariance_pass=False,
    )
    assert metrics["best_coordinate_diagnostic_transform"] == "mirror_x"
    assert metrics["operator_error_coord_diagnostic"] < 1e-12
    assert metrics["physical_transform_set"] == ["I"]

    twin = apply_trace_transform(theta4, "twin", model_dim=4)
    blocked = evaluate_trace_seidel_recovery(
        theta4,
        twin,
        dim=16,
        sys_params=SYS_PARAMS,
        model_dim=4,
        probe_config=config,
        dataset_twin_invariance_pass=False,
    )
    assert blocked["physical_transform_set"] == ["I"]

    theta5 = np.asarray([0.02, -0.015, 0.01, 0.006, 0.012])
    mirrored5 = apply_trace_transform(theta5, "mirror_x", model_dim=5)
    metrics5 = evaluate_trace_seidel_recovery(
        theta5,
        mirrored5,
        dim=16,
        sys_params=SYS_PARAMS,
        model_dim=5,
        probe_config=config,
        dataset_twin_invariance_pass=False,
    )
    assert metrics5["best_coordinate_diagnostic_transform"] == "mirror_x"
    assert metrics5["operator_error_coord_diagnostic"] < 1e-12
    assert metrics5["physical_transform_set"] == ["I"]

    twin5 = apply_trace_transform(theta5, "twin", model_dim=5)
    blocked5 = evaluate_trace_seidel_recovery(
        theta5,
        twin5,
        dim=16,
        sys_params=SYS_PARAMS,
        model_dim=5,
        probe_config=config,
        dataset_twin_invariance_pass=False,
    )
    assert blocked5["physical_transform_set"] == ["I"]


def test_trace5_pure_terms_and_coma_distortion_isolation():
    h = 0.6
    rho = 0.7
    hx, hy = h, 0.0
    px, py = rho, 0.0

    s = trace_seidel_wavefront([0.2, 0, 0, 0, 0], hx, hy, px, py, model_dim=5)
    assert s == pytest.approx(0.2 * rho**4)
    s_minus = trace_seidel_wavefront([0.2, 0, 0, 0, 0], -hx, -hy, px, py, model_dim=5)
    assert s_minus == pytest.approx(s)

    c = trace_seidel_wavefront([0, 0.2, 0, 0, 0], hx, hy, px, py, model_dim=5)
    assert c == pytest.approx(0.2 * h * rho**3)
    c_minus = trace_seidel_wavefront([0, 0.2, 0, 0, 0], -hx, -hy, px, py, model_dim=5)
    assert c_minus == pytest.approx(-c)

    a = trace_seidel_wavefront([0, 0, 0.2, 0, 0], hx, hy, px, py, model_dim=5)
    assert a == pytest.approx(0.2 * h**2 * rho**2)
    angles = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    a_ring = trace_seidel_wavefront(
        [0, 0, 0.2, 0, 0],
        hx,
        hy,
        rho * np.cos(angles),
        rho * np.sin(angles),
        model_dim=5,
    )
    assert abs(float(np.mean(a_ring))) < 1e-14

    f = trace_seidel_wavefront([0, 0, 0, 0.2, 0], hx, hy, px, py, model_dim=5)
    assert f == pytest.approx(0.2 * h**2 * rho**2)
    f_minus = trace_seidel_wavefront([0, 0, 0, 0.2, 0], -hx, -hy, px, py, model_dim=5)
    assert f_minus == pytest.approx(f)

    d = trace_seidel_wavefront([0, 0, 0, 0, 0.2], hx, hy, px, py, model_dim=5)
    assert d == pytest.approx(0.2 * h**3 * rho)
    d_minus = trace_seidel_wavefront([0, 0, 0, 0, 0.2], -hx, -hy, px, py, model_dim=5)
    assert d_minus == pytest.approx(-d)

    c_h2 = trace_seidel_wavefront([0, 0.2, 0, 0, 0], 2 * h, 0.0, px, py, model_dim=5)
    d_h2 = trace_seidel_wavefront([0, 0, 0, 0, 0.2], 2 * h, 0.0, px, py, model_dim=5)
    assert c_h2 / c == pytest.approx(2.0)
    assert d_h2 / d == pytest.approx(8.0)

    c_r2 = trace_seidel_wavefront([0, 0.2, 0, 0, 0], hx, hy, 2 * px, py, model_dim=5)
    d_r2 = trace_seidel_wavefront([0, 0, 0, 0, 0.2], hx, hy, 2 * px, py, model_dim=5)
    assert c_r2 / c == pytest.approx(8.0)
    assert d_r2 / d == pytest.approx(2.0)


def test_synthetic_ablation_full_symmetry_generalizes_to_heldout_fields():
    theta = np.asarray([0.1, -0.04, 0.03, 0.02])
    train_fields = [
        (0.35, 0.0),
        (0.35, math.pi / 2),
        (0.7, math.pi),
        (0.7, 3 * math.pi / 2),
    ]
    heldout = [(0.55, math.pi / 4), (0.9, 3 * math.pi / 4)]

    def local_proxy(h, psi):
        return {
            "free_local_psf": np.asarray([theta[0], theta[1] * h, theta[2] * h * h, theta[3] * h * h]),
            "local_zernike": np.asarray(
                [
                    theta[0],
                    theta[1] * h * np.exp(1j * psi),
                    theta[2] * h * h * np.exp(2j * psi),
                    theta[3] * h * h,
                ],
                dtype=complex,
            ),
        }

    proxies = [local_proxy(h, psi)["local_zernike"] for h, psi in train_fields]
    hs = np.asarray([h for h, _ in train_fields])
    psis = np.asarray([psi for _, psi in train_fields])
    c_hat = np.sum(hs * np.real([z[1] * np.exp(-1j * p) for z, p in zip(proxies, psis)])) / np.sum(hs * hs)
    a_hat = np.sum(hs**2 * np.real([z[2] * np.exp(-2j * p) for z, p in zip(proxies, psis)])) / np.sum(hs**4)
    f_hat = np.sum(hs**2 * np.real([z[3] for z in proxies])) / np.sum(hs**4)
    s_hat = np.mean([z[0].real for z in proxies])
    full_symmetry = np.asarray([s_hat, c_hat, a_hat, f_hat])
    np.testing.assert_allclose(full_symmetry, theta, atol=1e-12)

    # A field-scaling-only model that ignores spin orientation cannot predict
    # complex held-out coma/astigmatism axes.
    field_scaling_only_coma = np.mean([z[1] / h for z, h in zip(proxies, hs)])
    spin_errors = []
    for h, psi in heldout:
        true_coma = theta[1] * h * np.exp(1j * psi)
        bad_coma = field_scaling_only_coma * h
        good_coma = full_symmetry[1] * h * np.exp(1j * psi)
        spin_errors.append(abs(bad_coma - true_coma) - abs(good_coma - true_coma))
    assert min(spin_errors) > 1e-6
