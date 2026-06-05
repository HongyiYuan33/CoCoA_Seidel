from __future__ import annotations

import json

import pytest

from scripts.run_seidel_symmetry_ablation_sweep import (
    build_cases,
    grouped_summary,
    metrics_path,
    parse_args,
    recover_case,
    run_sweep,
)


def test_ablation_dry_run_case_matrix_is_explicit():
    args = parse_args(
        [
            "--dry-run",
            "--dim",
            "16",
            "--num-seeds",
            "1",
            "--models",
            "backend6",
            "classical4d",
            "classical5d",
            "classical6d",
            "--directions",
            "balanced",
            "--strengths",
            "0.06",
            "--image",
            "baboon",
        ]
    )
    cases = build_cases(args)
    assert len(cases) == 4
    assert {case.model_name for case in cases} == {"backend6", "classical4d", "classical5d", "classical6d"}
    assert all(case.case_id for case in cases)


def test_ablation_default_models_are_classical_backend_family():
    args = parse_args(["--dry-run", "--dim", "16", "--num-seeds", "1"])
    cases = build_cases(args)
    assert {case.model_name for case in cases} == {"classical4d", "classical5d", "classical6d"}


def test_default_classical_case_uses_backend_gt_not_trace_gt():
    args = parse_args(
        [
            "--dim",
            "16",
            "--num-seeds",
            "1",
            "--models",
            "classical4d",
            "--directions",
            "balanced",
            "--strengths",
            "0.06",
            "--image",
            "baboon",
        ]
    )
    row = recover_case(build_cases(args)[0])
    assert row["gt_convention"] == "classical_backend6"
    assert json.loads(row["theta_gt_trace5"]) == "not_applicable"
    assert len(json.loads(row["theta_backend6_gt"])) == 6
    assert row["theta_convention"] == "classical4d"
    assert row["eta_1"] == "not_applicable"


def test_case_shards_are_disjoint_and_cover_matrix():
    base_args = [
        "--dim",
        "16",
        "--num-seeds",
        "2",
            "--models",
            "backend6",
            "classical4d",
        "--directions",
        "balanced",
        "coma_dominant",
        "--strengths",
        "0.04",
        "0.06",
        "--image",
        "baboon",
    ]
    full = build_cases(parse_args(base_args))
    shard0 = build_cases(parse_args([*base_args, "--num-shards", "2", "--shard-index", "0"]))
    shard1 = build_cases(parse_args([*base_args, "--num-shards", "2", "--shard-index", "1"]))
    ids0 = {case.case_id for case in shard0}
    ids1 = {case.case_id for case in shard1}
    assert ids0.isdisjoint(ids1)
    assert ids0 | ids1 == {case.case_id for case in full}


def test_recover_case_serializes_conventions_and_metrics():
    args = parse_args(
        [
            "--dim",
            "16",
            "--num-seeds",
            "1",
            "--models",
            "classical5d",
            "--directions",
            "balanced",
            "--strengths",
            "0.06",
            "--image",
            "baboon",
        ]
    )
    row = recover_case(build_cases(args)[0])
    assert row["gt_convention"] == "classical_backend6"
    assert row["theta_convention"] == "classical5d"
    assert len(json.loads(row["theta_gt"])) == 6
    assert json.loads(row["theta_gt_trace5"]) == "not_applicable"
    assert len(json.loads(row["theta_backend6_gt"])) == 6
    assert row["fixed_seidel_indices"] == [5]
    assert row["no_defocus"] is True
    assert row["no_w311_no_defocus"] is False
    assert "operator_error_strict" in row
    assert "heldout_operator_error_strict" in row
    assert "generalization_gap" in row
    assert row["result_type"] == "native_exact_rdm"
    assert row["eta_1"] == "not_applicable"


def test_d_containing_ablation_marks_classical4d_misspecified():
    args = parse_args(
        [
            "--dim",
            "16",
            "--num-seeds",
            "1",
            "--models",
            "classical4d",
            "classical5d",
            "classical6d",
            "--directions",
            "pure_distortion",
            "--strengths",
            "0.06",
            "--image",
            "baboon",
        ]
    )
    rows = {row["model_name"]: row for row in [recover_case(case) for case in build_cases(args)]}
    assert rows["classical4d"]["misspecified_gt"] is True
    assert rows["classical5d"]["misspecified_gt"] is False
    assert rows["classical6d"]["misspecified_gt"] is False
    assert rows["classical5d"]["distortion_forward_model"] == "backend_W311"


def test_proxy_trace_ablation_cli_modes_are_paused():
    with pytest.raises(SystemExit):
        parse_args(["--models", "field_scaling_only"])


def test_run_sweep_resume_and_summary_outputs(tmp_path):
    args = parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--dim",
            "16",
            "--num-seeds",
            "1",
            "--models",
            "classical4d",
            "--directions",
            "balanced",
            "--strengths",
            "0.06",
            "--image",
            "baboon",
            "--resume",
        ]
    )
    rows_first = run_sweep(args)
    assert len(rows_first) == 1
    case = build_cases(args)[0]
    path = metrics_path(tmp_path, case)
    assert path.is_file()
    marker = json.loads(path.read_text())
    marker["resume_marker"] = True
    path.write_text(json.dumps(marker, indent=2))
    rows_second = run_sweep(args)
    assert rows_second[0]["resume_marker"] is True
    assert (tmp_path / "ablation_results.csv").is_file()
    assert (tmp_path / "ablation_summary.json").is_file()
    assert (tmp_path / "plots" / "heldout_operator_error_by_model.png").is_file()


def test_parallel_shard_mode_can_skip_and_rebuild_aggregate(tmp_path):
    base_args = [
        "--output-root",
        str(tmp_path),
        "--dim",
        "16",
        "--num-seeds",
        "1",
        "--models",
        "classical4d",
        "--directions",
        "balanced",
        "--strengths",
        "0.06",
        "--image",
        "baboon",
        "--resume",
    ]
    rows = run_sweep(parse_args([*base_args, "--skip-aggregate", "--no-plots"]))
    assert len(rows) == 1
    assert not (tmp_path / "ablation_results.csv").exists()

    rebuilt = run_sweep(parse_args([*base_args, "--aggregate-only"]))
    assert len(rebuilt) == 1
    assert (tmp_path / "ablation_results.csv").is_file()
    assert (tmp_path / "ablation_summary.json").is_file()
    assert (tmp_path / "plots" / "heldout_operator_error_by_model.png").is_file()


def test_grouped_summary_computes_model_rollups():
    rows = [
        {
            "model_name": "classical4d",
            "operator_error_strict": 0.01,
            "heldout_operator_error_strict": 0.02,
            "generalization_gap": 0.01,
            "parameter_count": 4,
        },
        {
            "model_name": "classical4d",
            "operator_error_strict": 0.03,
            "heldout_operator_error_strict": 0.05,
            "generalization_gap": 0.02,
            "parameter_count": 4,
        },
        {
            "model_name": "backend6",
            "operator_error_strict": 0.20,
            "heldout_operator_error_strict": 0.25,
            "generalization_gap": 0.05,
            "parameter_count": 6,
        },
    ]
    summary = grouped_summary(rows, failure_threshold=0.1)
    by_model = {row["model_name"]: row for row in summary}
    assert by_model["classical4d"]["mean_parameter_count"] == 4.0
    assert by_model["backend6"]["failure_rate"] == 1.0
