from __future__ import annotations

import numpy as np

import hybrid_ring_cocoa.evaluation as evaluation
from hybrid_ring_cocoa.evaluation import (
    OperatorProbeConfig,
    evaluate_seidel_recovery,
)
from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import (
    TRACE5_TRANSFORM_SIGNS,
    apply_trace_transform,
    check_trace_dataset_twin_invariance,
    evaluate_trace_seidel_recovery,
)


def test_evaluation_package_exports_classical_public_api_only():
    assert OperatorProbeConfig is not None
    assert evaluate_seidel_recovery is not None

    paused_trace_names = {
        "TRACE5_TRANSFORM_SIGNS",
        "TRACE4_TRANSFORM_SIGNS",
        "TRACE3_TRANSFORM_SIGNS",
        "apply_trace_transform",
        "check_trace_dataset_twin_invariance",
        "evaluate_trace_seidel_recovery",
    }
    for name in paused_trace_names:
        assert not hasattr(evaluation, name)


def test_trace_reproduction_helpers_are_internal_module_only():
    assert evaluate_trace_seidel_recovery is not None
    assert check_trace_dataset_twin_invariance is not None
    assert TRACE5_TRANSFORM_SIGNS["mirror_x"] == (1.0, -1.0, 1.0, 1.0, -1.0)

    theta5 = np.asarray([0.30, -0.10, 0.05, 0.08, 0.04], dtype=np.float64)
    transformed = apply_trace_transform(theta5, "twin", model_dim=5)
    np.testing.assert_allclose(transformed, [-0.30, -0.10, -0.05, -0.08, 0.04])
