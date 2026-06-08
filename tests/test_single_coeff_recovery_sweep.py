from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.run_cocoa_like_2d_mechanism as cocoa
import scripts.run_cocoa_like_seidel_accuracy_sweep as sweep


def test_single_coeff_candidates_are_one_hot_and_fixed_except_active():
    args = sweep.parse_args(
        [
            "--stage",
            "stage1",
            "--candidate-mode",
            "single_coeff",
            "--seidel-convention",
            "classical6d",
            "--coefficients",
            "W040",
            "W131",
            "W222",
            "W220",
            "W311",
            "Wd",
            "--coefficient-values",
            "0.1",
            "0.2",
            "0.4",
            "-0.1",
            "-0.2",
            "-0.4",
        ]
    )
    candidates = sweep.make_candidates(
        args.directions,
        args.strengths,
        seidel_convention=args.seidel_convention,
        candidate_mode=args.candidate_mode,
        coefficients=args.coefficients,
        coefficient_values=args.coefficient_values,
    )

    assert len(candidates) == 36
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    assert "W040__coef0p1" in by_id
    assert "W040__coefm0p1" in by_id

    w040_neg = by_id["W040__coefm0p1"]
    np.testing.assert_allclose(w040_neg.seidel, [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert w040_neg.active_seidel_index == 0
    assert w040_neg.fixed_seidel_indices == (1, 2, 3, 4, 5)

    wd_pos = by_id["Wd__coef0p4"]
    np.testing.assert_allclose(wd_pos.seidel, [0.0, 0.0, 0.0, 0.0, 0.0, 0.4])
    assert wd_pos.active_seidel_index == 5
    assert wd_pos.fixed_seidel_indices == (0, 1, 2, 3, 4)


def test_single_coeff_requires_classical6d():
    with pytest.raises(SystemExit):
        sweep.parse_args(
            [
                "--candidate-mode",
                "single_coeff",
                "--seidel-convention",
                "classical5d",
            ]
        )


def test_augment_metrics_records_active_error_and_leakage():
    candidate = sweep.make_single_coeff_candidates(
        coefficients=["W222"],
        coefficient_values=[0.2],
        seidel_convention="classical6d",
    )[0]
    metrics = {
        "seidel_gt": candidate.seidel.tolist(),
        "seidel_final": [0.01, 0.0, 0.18, 0.0, -0.03, 0.0],
        "gt_hf_ratio": 1.0,
        "measurement_hf_ratio": 0.5,
        "config": {},
    }

    out = sweep.augment_metrics(
        metrics,
        stage="stage1",
        image="Test_figure_1",
        candidate=candidate,
        seed=0,
        seidel_convention="classical6d",
    )

    assert out["active_seidel_index"] == 2
    assert out["active_seidel_name"] == "W222"
    assert out["active_seidel_value"] == pytest.approx(0.2)
    assert out["active_seidel_final"] == pytest.approx(0.18)
    assert out["active_seidel_error"] == pytest.approx(-0.02)
    assert out["non_active_seidel_l2_leakage"] == pytest.approx(np.hypot(0.01, -0.03))
    assert out["fixed_seidel_indices"] == [0, 1, 3, 4, 5]


def test_run_one_mode_fixed_override_is_written_to_metrics(tmp_path, monkeypatch):
    sharp = torch.full((8, 8), 0.5, dtype=torch.float32)
    meas = torch.full((8, 8), 0.4, dtype=torch.float32)
    gt = torch.tensor([0.0, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    def fake_pretrain(*args, **kwargs):
        return [0.25]

    def fake_train(*args, pretrain_history, **kwargs):
        return cocoa.CocoaLikeResult(
            sharp_final=sharp.clone(),
            seidel_final=torch.tensor([0.0, 0.09, 0.0, 0.0, 0.0, 0.0]),
            measurement_pred=meas.clone(),
            loss_history=[0.2],
            ssim_history=[0.2],
            rsd_history=[0.0],
            tv_history=[0.0],
            anchor_history=[0.0],
            seidel_rms_floor_history=[0.0],
            seidel_wavefront_rms_history=[0.0],
            pretrain_history=pretrain_history,
            elapsed_s=0.0,
        )

    monkeypatch.setattr(cocoa, "pretrain_cocoa_like", fake_pretrain)
    monkeypatch.setattr(cocoa, "train_cocoa_like", fake_train)
    monkeypatch.setattr(cocoa, "save_mode_figures", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        image="Test_figure_1",
        size=8,
        seed=0,
        max_val=40.0,
        nerf_beta=1.0,
        output_mode="softplus",
        nerf_depth=1,
        nerf_width=4,
        nerf_skips=(),
        fourier_num_angles=1,
        fourier_num_octaves=1,
        pretrain_iter=1,
        num_iter=1,
        lr_obj=1e-3,
        lr_seidel=1e-3,
        pretrain_scalar=1.0,
        rsd_weight=0.0,
        tv_weight=0.0,
        defocus_anchor_weight=1.0,
        defocus_index=5,
        scheduler=None,
        eta_min_ratio=0.1,
        seidel_rms_floor_weight=0.0,
        seidel_rms_floor_alpha=0.8,
        seidel_rms_floor_target=None,
        seidel_rms_floor_field_samples=3,
        seidel_rms_floor_pupil_samples=5,
        seidel_convention="classical6d",
        gt_preset="custom",
        gt_label="W131__coef0p1",
        gt_source="custom",
        fixed_seidel_indices_override=[0, 2, 3, 4, 5],
        verbose=False,
    )

    _, metrics = cocoa.run_one_mode(
        args,
        mode="joint",
        sharp_gt=sharp,
        meas_gt=meas,
        gt_vec=gt,
        gt_np=gt.numpy(),
        root_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert metrics["fixed_seidel_indices"] == [0, 2, 3, 4, 5]
    assert metrics["no_defocus"] is True
    assert metrics["no_w311_no_defocus"] is True
    assert (tmp_path / "joint" / "metrics.json").is_file()
