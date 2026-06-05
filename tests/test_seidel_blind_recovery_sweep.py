from __future__ import annotations

import json

import pytest

import scripts.run_cocoa_like_2d_mechanism as cocoa
from scripts.run_seidel_blind_recovery_sweep import (
    build_cases,
    grouped_summary,
    metrics_path,
    parse_args,
    run_sweep,
    sanity_gate,
    scaled_backend_gt,
    scaled_trace_gt,
    theta_for_model_target,
)


def test_sanity_matrix_has_expected_classical_cases():
    args = parse_args(
        [
            "--stage",
            "sanity",
            "--dry-run",
            "--dim",
            "256",
            "--images",
            "Test_figure_1",
            "dendrites_dense",
            "--models",
            "backend6",
            "classical4d",
            "classical5d",
            "classical6d",
            "--directions",
            "balanced",
            "coma_dominant",
            "astig_field",
        ]
    )
    cases = build_cases(args)
    assert len(cases) == 24
    assert {case.model_name for case in cases} == {"backend6", "classical4d", "classical5d", "classical6d"}
    assert {case.image for case in cases} == {"Test_figure_1", "dendrites_dense"}
    assert {case.strength for case in cases} == {0.06}


def test_medium_matrix_has_expected_216_cases():
    args = parse_args(["--stage", "medium", "--dry-run", "--dim", "256"])
    cases = build_cases(args)
    assert len(cases) == 216
    assert {case.model_name for case in cases} == {"classical4d", "classical5d", "classical6d"}
    assert {case.seed for case in cases} == {0, 1, 2}
    assert {case.strength for case in cases} == {0.04, 0.06, 0.08, 0.10}
    assert {case.direction for case in cases} == {"balanced", "coma_dominant", "astig_field"}


def test_shards_are_disjoint_and_cover_full_matrix():
    base = ["--stage", "medium", "--dim", "256"]
    full = build_cases(parse_args(base))
    shard0 = build_cases(parse_args([*base, "--num-shards", "2", "--shard-index", "0"]))
    shard1 = build_cases(parse_args([*base, "--num-shards", "2", "--shard-index", "1"]))
    ids0 = {case.case_id for case in shard0}
    ids1 = {case.case_id for case in shard1}
    assert ids0.isdisjoint(ids1)
    assert ids0 | ids1 == {case.case_id for case in full}


def test_reduced_targets_drop_only_their_missing_trace5_terms():
    theta_trace5 = [0.3, -0.1, 0.05, 0.08, 0.04]
    theta_backend6 = [0.3, -0.1, 0.1, 0.03, 0.04, 0.0]
    target5 = theta_for_model_target(theta_backend6, "trace5", theta_trace5_gt=theta_trace5)
    assert target5.tolist() == pytest.approx(theta_trace5)
    target4 = theta_for_model_target(theta_backend6, "trace4", theta_trace5_gt=theta_trace5)
    assert target4.tolist() == pytest.approx([0.3, -0.1, 0.05, 0.08])
    target = theta_for_model_target(theta_backend6, "trace3", theta_trace5_gt=theta_trace5)
    assert target.tolist() == pytest.approx([0.3, -0.1, 0.05])


def test_classical_gt_scaling_is_backend6_not_trace5():
    backend_gt = scaled_backend_gt("balanced", 0.06)
    trace_gt = scaled_trace_gt("balanced", 0.06)
    assert len(backend_gt) == 6
    assert len(trace_gt) == 5
    assert backend_gt[4] == pytest.approx(0.0)
    assert backend_gt[5] == pytest.approx(0.0)

    target4 = theta_for_model_target(backend_gt, "classical4d")
    assert len(target4) == 6
    assert target4[4] == pytest.approx(0.0)
    assert target4[5] == pytest.approx(0.0)


def test_cocoa_cli_rejects_trace_separated_conventions():
    with pytest.raises(SystemExit):
        cocoa.parse_args(["--seidel-convention", "trace5"])


def test_cocoa_cli_defaults_to_classical_backend_family():
    args = cocoa.parse_args([])
    assert args.seidel_convention == "classical6d"
    assert args.nerf_depth == 6
    assert args.nerf_width == 128
    assert args.nerf_skips == (2, 4, 6)
    assert cocoa.trace_model_dim("classical4d") is None
    assert cocoa.trace_model_dim("classical5d") is None
    assert cocoa.trace_model_dim("classical6d") is None
    assert cocoa.convention_metadata("classical4d")["fixed_seidel_indices"] == [4, 5]
    assert cocoa.convention_metadata("classical5d")["fixed_seidel_indices"] == [5]
    assert cocoa.convention_metadata("classical6d")["fixed_seidel_indices"] == []


def test_cocoa_cli_parses_mlp_capacity_knobs():
    args = cocoa.parse_args(
        ["--nerf-depth", "4", "--nerf-width", "64", "--nerf-skips", "2"]
    )
    assert args.nerf_depth == 4
    assert args.nerf_width == 64
    assert args.nerf_skips == (2,)

    args = cocoa.parse_args(["--nerf-skips", "none"])
    assert args.nerf_skips == ()

    with pytest.raises(SystemExit):
        cocoa.parse_args(["--nerf-skips", "0"])


def test_blind_sweep_cli_parses_mlp_capacity_knobs():
    args = parse_args(
        ["--nerf-depth", "3", "--nerf-width", "32", "--nerf-skips", "none"]
    )
    assert args.nerf_depth == 3
    assert args.nerf_width == 32
    assert args.nerf_skips == ()


def fake_metric(case_id: str, model: str, image: str, op: float) -> dict:
    return {
        "case_id": case_id,
        "stage": "sanity",
        "status": "success",
        "image": image,
        "model_name": model,
        "direction": "balanced",
        "strength": 0.06,
        "seed": 0,
        "dim": 256,
        "loss_decreased": True,
        "operator_error_initial": 0.2,
        "operator_error_strict": op,
        "operator_error_reduction": 1.0 - op / 0.2,
        "operator_error_improved": op < 0.2,
        "wavefront_error_strict": op,
        "W311_hat": 0.01 if model == "backend6" else 0.0,
        "Wd_hat": 0.02 if model == "backend6" else 0.0,
        "gauge_leakage_l2": 0.03 if model == "backend6" else 0.0,
    }


def test_sanity_gate_checks_classical_models_and_backend_gauge():
    rows = []
    for image in ("Test_figure_1", "dendrites_dense"):
        rows.append(fake_metric(f"{image}_backend6", "backend6", image, 0.04))
        rows.append(fake_metric(f"{image}_classical4d", "classical4d", image, 0.035))
        rows.append(fake_metric(f"{image}_classical5d", "classical5d", image, 0.03))
        rows.append(fake_metric(f"{image}_classical6d", "classical6d", image, 0.025))
    gate = sanity_gate(rows)
    assert gate["pass"] is True
    assert gate["checks"]["classical4d_operator_improved_all"] is True
    assert gate["checks"]["classical5d_operator_improved_all"] is True
    assert gate["checks"]["classical6d_operator_improved_all"] is True
    assert gate["checks"]["backend6_gauge_diagnostics_present"] is True
    assert gate["classical4d_mean_operator_error"] == pytest.approx(0.035)


def test_aggregate_only_rebuilds_outputs_from_markers(tmp_path):
    args = parse_args(["--output-root", str(tmp_path), "--stage", "sanity", "--aggregate-only"])
    case = build_cases(parse_args(["--output-root", str(tmp_path), "--stage", "sanity", "--limit", "1"]))[0]
    path = metrics_path(tmp_path, case)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(fake_metric(case.case_id, case.model_name, case.image, 0.04)))
    rows = run_sweep(args)
    assert len(rows) == 1
    assert (tmp_path / "blind_recovery_results.csv").is_file()
    assert (tmp_path / "blind_recovery_summary.json").is_file()
    assert (tmp_path / "plots" / "operator_error_by_model.png").is_file()


def test_grouped_summary_reports_backend_gauge_leakage():
    rows = [
        fake_metric("a", "backend6", "Test_figure_1", 0.04),
        fake_metric("b", "classical4d", "Test_figure_1", 0.03),
    ]
    summary = grouped_summary(rows)
    by_model = {row["model_name"]: row for row in summary}
    assert by_model["backend6"]["mean_gauge_leakage_l2"] == 0.03
