# Recent Seidel Evaluator Artifacts

This workspace is now the main local home for the recent Seidel physical-operator
evaluator and the latest completed evaluation outputs.

## Evaluator Code

- `hybrid_ring_cocoa/evaluation/seidel_operator_evaluator.py`
- `hybrid_ring_cocoa/evaluation/__init__.py`
- `scripts/evaluate_seidel_physical_operator_sweep.py`
- `tests/test_seidel_operator_evaluator.py`

## Completed Evaluator Outputs

Summary directory:

- `outputs/cocoa_like_2d_mechanism/seidel_operator_eval_summary/`

Included sweeps:

- `seidel_operator_eval_size256_6d`
- `seidel_operator_eval_size256_5d`
- `seidel_operator_eval_size256_4d`
- `seidel_operator_eval_size512_6d`
- `seidel_operator_eval_size128_6d`

Top-level summary files:

- `outputs/cocoa_like_2d_mechanism/seidel_operator_eval_summary/seidel_operator_eval_summary.md`
- `outputs/cocoa_like_2d_mechanism/seidel_operator_eval_summary/sweep_summary.csv`
- `outputs/cocoa_like_2d_mechanism/seidel_operator_eval_summary/direction_summary.csv`
- `outputs/cocoa_like_2d_mechanism/seidel_operator_eval_summary/rms_summary.csv`
- `outputs/cocoa_like_2d_mechanism/seidel_operator_eval_summary/transform_counts.csv`

## Recent Visualization Artifacts

- `outputs/cocoa_like_2d_mechanism/seidel_sweep_size256_5d_viz/`
- `outputs/cocoa_like_2d_mechanism/seidel_sweep_size256_4d_viz/`

## Size256 5D Data Snapshot

The size256 5D/no-defocus sweep snapshot used for visualization and evaluator
input is also staged as:

- `outputs/cocoa_like_2d_mechanism/seidel_recovery_sweep_stage1_size256_fullbudget_nodefocus_gpu0/stage1_metrics.csv`

## Verification

Local verification in this workspace:

- `python3 -m py_compile hybrid_ring_cocoa/evaluation/seidel_operator_evaluator.py scripts/evaluate_seidel_physical_operator_sweep.py tests/test_seidel_operator_evaluator.py`
- `python3 -m pytest tests/test_seidel_operator_evaluator.py -q`

Result: `11 passed`.
