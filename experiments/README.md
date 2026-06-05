# Organized Experiment Index

This directory is a structured index for the experiment results in this
workspace. It does not move or rename the original result directories under
`outputs/` or `notebooks/`; each entry here is a symlink with a clearer,
standardized name.

Keeping the original paths intact protects old scripts, notebooks, logs, and
README references while still giving the project a clean experiment catalog.

## Naming Convention

Symlink names use double underscores to separate stable fields:

```text
<family>__<model_or_scope>__<size_or_data>__<stage_or_budget>__<device_or_date>__<status>
```

Common status tags:

- `complete`: result directory appears to contain completed outputs.
- `partial`: result directory has only a small subset or exploratory outputs.
- `stage1_complete_stage2_paused`: stage1 metrics exist, later stages are
  intentionally paused or empty.
- `index`: sequence/index directory rather than a full experiment result.
- `smoke`: diagnostic or regression smoke result.

## Layout

- `01_baseline_cocoa_like_2d/`: original CoCoA-like 2D mechanism runs and
  moderate reproduction sweeps.
- `02_seidel_recovery_sweeps/`: Seidel recoverability sweeps, split into full
  grids, no-defocus/no-W311 variants, tuned size512 mini runs, and sequence
  indices.
- `03_object_prior_sweeps/`: object prior parameter grids and learning-rate
  sweeps.
- `04_operator_evaluation/`: physical-operator evaluator outputs, trace5
  post-hoc checks, and calibrated operator-error visualizations.
- `05_ablation_and_blind_recovery/`: symmetry ablations and blind-recovery
  sanity sweeps.
- `06_regression_and_smoke/`: frozen RDM golden regressions and API smoke
  outputs.
- `07_real_measurement_reconstructions/`: real-measurement reconstruction runs.
- `08_reports_and_visualizations/`: notebooks, Seidel sweep reports, and
  physical-similarity contact sheets/cards.

## Manifest

See `00_catalog/EXPERIMENT_MANIFEST.md` for the full mapping from standardized
names back to original paths.
