# Experiment Manifest

This manifest maps the organized symlink names in `experiments/` back to the
original directories. Original paths are preserved.

## 01 Baseline CoCoA-Like 2D

| Organized entry | Original path | Notes |
|---|---|---|
| `01_baseline_cocoa_like_2d/baseline_cocoa2d__ucla_fluor256__pre400_joint1000__seed0__complete` | `outputs/cocoa_like_2d_mechanism/cocoa_like2d_fluor256_ucla_pre400_joint1000_seed0` | Main README baseline run. |
| `01_baseline_cocoa_like_2d/cocoa2d_sweep__moderate__pre400_joint1000__20260524__complete` | `outputs/cocoa_like_2d_mechanism/moderate_sweep_20260524_pre400_joint1000` | Moderate reproduction sweep. |

## 02 Seidel Recovery Sweeps

### Full Grid

| Organized entry | Original path | Notes |
|---|---|---|
| `02_seidel_recovery_sweeps/01_full_grid/seidel_recovery__backend6__size128_256__all_stages__20260525__complete` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_sweep_20260525_full` | Full all-stage sweep with stage2/stage3 outputs. |
| `02_seidel_recovery_sweeps/01_full_grid/seidel_recovery__backend6__size256__stage1_fastbudget__gpu1__complete` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_sweep_stage1_size256_fastbudget_gpu1` | Stage1 size256 fast-budget grid. |
| `02_seidel_recovery_sweeps/01_full_grid/seidel_recovery__backend6__size256__stage1_fullbudget__gpu1__complete` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_sweep_stage1_size256_fullbudget_gpu1` | Stage1 size256 full-budget grid. |
| `02_seidel_recovery_sweeps/01_full_grid/seidel_recovery__backend6__size512__stage1_fullbudget__gpu1__complete` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_sweep_stage1_size512_fullbudget_gpu1` | Stage1 size512 full-budget grid. |

### No Defocus / No W311

| Organized entry | Original path | Notes |
|---|---|---|
| `02_seidel_recovery_sweeps/02_no_defocus_or_no_w311/seidel_recovery__classical5d__size256__no_defocus__stage1_fullbudget__gpu0__complete` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_sweep_stage1_size256_fullbudget_nodefocus_gpu0` | 5D no-defocus size256 stage1 grid. |
| `02_seidel_recovery_sweeps/02_no_defocus_or_no_w311/seidel_recovery__classical5d__size512__no_defocus__stage1_fullbudget__gpu1__complete` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_sweep_stage1_size512_fullbudget_nodefocus_gpu1` | 5D no-defocus size512 stage1 grid. |
| `02_seidel_recovery_sweeps/02_no_defocus_or_no_w311/seidel_recovery__classical4d__size256__no_w311_no_defocus__stage1_fullbudget__gpu1__complete` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_sweep_stage1_size256_fullbudget_no_w311_nodefocus_gpu1` | 4D no-W311/no-defocus size256 grid. |

### Tuned Mini Size512

| Organized entry | Original path | Notes |
|---|---|---|
| `02_seidel_recovery_sweeps/03_tuned_mini_size512/seidel_recovery_mini__classical4d__size512__rms006_012__fourier_gpu2__complete` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_mini_size512_4d_tuned_rms006_012_fourier_gpu2` | 4D tuned mini run on Fourier GPU2. |
| `02_seidel_recovery_sweeps/03_tuned_mini_size512/seidel_recovery_mini__classical5d__size512__rms006_012__caml_gpu0__complete` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_mini_size512_5d_tuned_rms006_012_caml_gpu0` | 5D tuned mini run. |
| `02_seidel_recovery_sweeps/03_tuned_mini_size512/seidel_recovery_mini__classical5d__size512__rms020__caml_gpu0__complete` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_mini_size512_5d_tuned_rms020_caml_gpu0` | 5D tuned rms0.20 mini run. |
| `02_seidel_recovery_sweeps/03_tuned_mini_size512/seidel_recovery_mini__classical5d__size512__extra_dirs_rms006_012_020__caml_gpu0__complete` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_mini_size512_5d_tuned_extra_dirs_rms006_012_020_caml_gpu0` | 5D tuned mini run with extra directions. |
| `02_seidel_recovery_sweeps/03_tuned_mini_size512/seidel_recovery_mini__backend6__size512__rms006_012__caml_gpu1__complete` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_mini_size512_6d_tuned_rms006_012_caml_gpu1` | 6D/backend6 tuned mini run. |
| `02_seidel_recovery_sweeps/03_tuned_mini_size512/seidel_recovery_mini__backend6__size512__rms020__caml_gpu1__complete` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_mini_size512_6d_tuned_rms020_caml_gpu1` | 6D/backend6 tuned rms0.20 mini run. |
| `02_seidel_recovery_sweeps/03_tuned_mini_size512/seidel_recovery_mini__backend6__size512__extra_dirs_rms006_012_020__caml_gpu1__complete` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_mini_size512_6d_tuned_extra_dirs_rms006_012_020_caml_gpu1` | 6D/backend6 tuned mini run with extra directions. |

### Sequences

| Organized entry | Original path | Notes |
|---|---|---|
| `02_seidel_recovery_sweeps/04_sequences/seidel_recovery_sequence__size256__stage1__gpu1__index` | `outputs/cocoa_like_2d_mechanism/seidel_recovery_sweep_stage1_size256_gpu1_sequence` | Size256 sequence comparison/index. |
| `02_seidel_recovery_sweeps/04_sequences/seidel_recovery_sequence__size512__gpu_sequence__index` | `outputs/cocoa_like_2d_mechanism/size512_sweeps_gpu_sequence` | Size512 sequence/index directory. |

## 03 Object Prior Sweeps

### Parameter Grids

| Organized entry | Original path | Notes |
|---|---|---|
| `03_object_prior_sweeps/01_param_grid/object_prior_grid__size128_256__softplus_sigmoid__20260525__complete` | `outputs/cocoa_like_2d_mechanism/object_prior_param_sweep_20260525_full` | Full object-prior parameter grid with stage1/2/3. |
| `03_object_prior_sweeps/01_param_grid/object_prior_grid__size512__focused_seidelmetric__stage1_complete_stage2_paused` | `outputs/cocoa_like_2d_mechanism/object_prior_param_sweep_size512_focused_seidelmetric` | Size512 focused grid; stage1 metrics exist, later stages paused/empty. |

### Learning Rate Sweeps

| Organized entry | Original path | Notes |
|---|---|---|
| `03_object_prior_sweeps/02_learning_rate/object_prior_lr__size512__beta5_rsd1em3_lower_lrs__gpu01__partial` | `outputs/cocoa_like_2d_mechanism/object_prior_lr_sweep_size512_beta5_rsd1em3_lower_lrs_cocoa06_iksung_dendrites_gpu01` | Small LR sweep subset. |
| `03_object_prior_sweeps/02_learning_rate/object_prior_lr__size512__beta7p5_rsd1em3_lro0p01_lrs5em4__gpu01__partial` | `outputs/cocoa_like_2d_mechanism/object_prior_lr_sweep_size512_beta7p5_rsd1em3_lro0p01_lrs5em4_cocoa06_iksung_dendrites_gpu01` | Small LR sweep subset. |
| `03_object_prior_sweeps/02_learning_rate/object_prior_lr__size512__beta7p5_rsd1em3_lro0p01_lrs5em4__gpu01__fullbudget_partial` | `outputs/cocoa_like_2d_mechanism/object_prior_lr_sweep_size512_beta7p5_rsd1em3_lro0p01_lrs5em4_fullbudget_cocoa06_iksung_dendrites_gpu01` | Full-budget LR subset. |

## 04 Operator Evaluation

| Organized entry | Original path | Notes |
|---|---|---|
| `04_operator_evaluation/01_physical_operator_metrics/operator_eval__physical_equiv__size128_256_512__all_models__complete` | `outputs/cocoa_like_2d_mechanism/seidel_operator_eval_summary` | Physical-operator evaluator summary; five sweeps complete. |
| `04_operator_evaluation/02_trace5_posthoc_analysis/operator_eval__trace5_posthoc__backend6_variants__complete` | `outputs/evaluator_trace5_analysis` | Trace5/backend6 post-hoc evaluator outputs. |
| `04_operator_evaluation/03_calibrated_error_visualization/operator_error_viz__calibrated__size256__complete` | `outputs/cocoa_like_2d_mechanism/operator_error_calibrated_size256_viz` | Size256 calibrated operator-error visualization. |
| `04_operator_evaluation/03_calibrated_error_visualization/operator_error_viz__calibrated_6d__size128_256_512__complete` | `outputs/cocoa_like_2d_mechanism/operator_error_calibrated_6d_size128_256_512_viz` | 6D calibrated operator-error visualization across sizes. |

## 05 Ablation And Blind Recovery

### Symmetry Ablation

| Organized entry | Original path | Notes |
|---|---|---|
| `05_ablation_and_blind_recovery/01_symmetry_ablation/symmetry_ablation__classical__smoke__gpu1__complete` | `outputs/seidel_symmetry_ablation_gpu1_smoke` | Classical symmetry-ablation smoke run. |
| `05_ablation_and_blind_recovery/01_symmetry_ablation/symmetry_ablation__classical__dim256__gpu01__complete` | `outputs/seidel_symmetry_ablation_gpu01_dim256` | Classical dim256 ablation. |
| `05_ablation_and_blind_recovery/01_symmetry_ablation/symmetry_ablation__classical__full__gpu01__complete` | `outputs/seidel_symmetry_ablation_gpu01_full` | Classical full ablation. |
| `05_ablation_and_blind_recovery/01_symmetry_ablation/symmetry_ablation__trace5_D__full__gpu1__complete` | `outputs/seidel_symmetry_ablation_gpu1_trace5_D_full` | Trace5-D full ablation. |
| `05_ablation_and_blind_recovery/01_symmetry_ablation/symmetry_ablation__trace5__dim256_full__gpu0__complete` | `outputs/seidel_symmetry_ablation_gpu0_dim256_trace5_full` | Trace5 dim256/full ablation. |

### Blind Recovery

| Organized entry | Original path | Notes |
|---|---|---|
| `05_ablation_and_blind_recovery/02_blind_recovery/blind_recovery__classical__dim256__sanity__complete` | `outputs/seidel_blind_recovery_dim256_sanity` | Classical blind-recovery sanity run. |
| `05_ablation_and_blind_recovery/02_blind_recovery/blind_recovery__trace5__dim256__sanity__gpu1__complete` | `outputs/seidel_blind_recovery_dim256_trace5_sanity_gpu1` | Trace5 blind-recovery sanity run. |

## 06 Regression And Smoke

| Organized entry | Original path | Notes |
|---|---|---|
| `06_regression_and_smoke/golden_regression__frozen_rdm_forward__gpu0__smoke` | `outputs/golden_forward_regression_gpu0` | Frozen RDM forward golden smoke run. |
| `06_regression_and_smoke/golden_regression__frozen_rdm_forward__gpu0__full` | `outputs/golden_forward_regression_gpu0_full` | Frozen RDM forward full golden regression. |
| `06_regression_and_smoke/smoke__trace4_reduced_api__complete` | `outputs/cocoa_like_2d_mechanism/smoke_trace4_reduced_api` | Trace4 reduced API smoke run. |

## 07 Real Measurement Reconstructions

| Organized entry | Original path | Notes |
|---|---|---|
| `07_real_measurement_reconstructions/real_measurement__classical4d__size512__paper_style__caml_gpu0__complete` | `outputs/cocoa_like_2d_mechanism/real_measurement_resolution_target_paper_style_size512_4d_tuned_caml_gpu0` | Real-measurement reconstruction, 4D tuned. |
| `07_real_measurement_reconstructions/real_measurement__classical5d__size512__paper_style__caml_gpu1__complete` | `outputs/cocoa_like_2d_mechanism/real_measurement_resolution_target_paper_style_size512_5d_tuned_caml_gpu1` | Real-measurement reconstruction, 5D tuned. |
| `07_real_measurement_reconstructions/real_measurement__backend6__size512__paper_style__caml_gpu0__complete` | `outputs/cocoa_like_2d_mechanism/real_measurement_resolution_target_paper_style_size512_6d_tuned_caml_gpu0` | Real-measurement reconstruction, 6D/backend6 tuned. |

## 08 Reports And Visualizations

### Notebooks

| Organized entry | Original path | Notes |
|---|---|---|
| `08_reports_and_visualizations/01_notebooks/notebook_report__seidel_sweep__cn__executed` | `notebooks/49_seidel_sweep_report_cn` | Chinese Seidel sweep report notebook. |
| `08_reports_and_visualizations/01_notebooks/notebook_report__object_prior_operator__cn__executed` | `notebooks/50_object_prior_operator_report_cn` | Chinese object-prior/operator report notebook. |

### Seidel Sweep Reports

| Organized entry | Original path | Notes |
|---|---|---|
| `08_reports_and_visualizations/02_seidel_sweep_reports/seidel_sweep_viz__size256__classical4d__complete` | `outputs/cocoa_like_2d_mechanism/seidel_sweep_size256_4d_viz` | Size256 4D Seidel sweep visualization. |
| `08_reports_and_visualizations/02_seidel_sweep_reports/seidel_sweep_viz__size256__classical5d__complete` | `outputs/cocoa_like_2d_mechanism/seidel_sweep_size256_5d_viz` | Size256 5D Seidel sweep visualization. |

### Physical Similarity Cards

| Organized entry | Original path | Notes |
|---|---|---|
| `08_reports_and_visualizations/03_physical_similarity_cards/similarity_cards__seidel_physical__candidate_cards__complete` | `outputs/cocoa_like_2d_mechanism/seidel_physical_similarity_cards` | Physical-similarity cards and candidate index. |
| `08_reports_and_visualizations/03_physical_similarity_cards/similarity_cards__tuned_prior_seidel_physical__contact_sheets__complete` | `outputs/cocoa_like_2d_mechanism/tuned_prior_seidel_physical_similarity` | Tuned-prior physical-similarity contact sheets. |
