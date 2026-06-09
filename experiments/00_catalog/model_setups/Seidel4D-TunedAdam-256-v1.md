# Seidel4D-TunedAdam-256-v1

Search aliases: `S4D-TunedAdam256`, `capacity4d-baseline-adam`, `4d-consistent-adam-baseline`, `tunedprior-4d-adam-256`.

## Purpose

Reusable 4D Seidel recovery baseline that gave the clean Adam coefficient
recovery in the 2026-06-07 capacity sweep.

This setup should be treated as a specific preset, not as a generic
`classical4d` run.  Its important property is that the generated ground-truth
Seidel vectors are first made consistent with the 4D model: backend indices
`W311` and `Wd` are fixed to zero before wavefront-RMS scaling.

## Canonical Run

- Run root: `outputs/cocoa_like_2d_mechanism/capacity4d_dirrms_tunedprior_size256_four_images_20260607__baseline`
- Capacity sweep prefix: `capacity4d_dirrms_tunedprior_size256_four_images_20260607`
- Comparison where it was reused as Adam baseline: `outputs/cocoa_like_2d_mechanism/seidelopt_sgd4d_tunedprior_size256_four_images_pre400_joint1000_20260608_adam_vs_sgd`
- RCP comparison: `outputs/cocoa_like_2d_mechanism/seidelopt_sgd4d_tunedprior_size256_four_images_pre400_joint1000_20260608_adam_vs_sgd_rcp_compare`

## Model And Optimizer

- Seidel convention: `classical4d`
- Fixed backend Seidel indices: `[4, 5]`, corresponding to `W311=0` and `Wd=0`
- Seidel optimizer: Adam
- Object optimizer: Adam
- Object MLP: depth `6`, width `128`, skips `2,4,6`
- Fourier encoding: `60` angles, `7` octaves
- Output mode: `softplus`
- `nerf_beta`: `5`
- `max_val`: `20`
- Image size: `256`

## Training Hyperparameters

- Stage: `stage1`
- Modes: joint only
- Pretrain iterations: `400`
- Joint iterations: `1000`
- `pretrain_scalar`: `5`
- `lr_obj`: `0.005`
- `lr_seidel`: `0.01`
- `rsd_weight`: `1e-3`
- `tv_weight`: `0`
- Scheduler: `cosine`
- `eta_min_ratio`: `0.04`
- `defocus_anchor_weight`: `1.0`
- `seidel_rms_floor_weight`: `0`

## Dataset / Candidate Grid

- Images: `Test_figure_1`, `Iksung_beads`, `dendrites`, `dendrites_dense`
- Directions: `cocoa_signed`, `signed_balanced`
- Target field-weighted wavefront RMS values: `0.06`, `0.20`, `0.40`
- Candidate generation: apply the `classical4d` fixed-index mask first, then
  scale the remaining 4D vector to the target field-weighted wavefront RMS.

## Why This Preset Is Different From Older 4D Runs

Some earlier 4D/no-defocus runs used measurements whose GT vectors still had a
nonzero `Wd` component, while the recovery model fixed `Wd=0`.  Those runs are
model-mismatched: 4D recovery cannot exactly represent the forward measurement.

`Seidel4D-TunedAdam-256-v1` avoids that mismatch.  The GT lives in the same
4D subspace that the optimizer is allowed to recover.  It also uses the tuned
object-prior setting (`max_val=20`, `nerf_beta=5`, `rsd_weight=1e-3`), which is
more restrictive than older defaults such as `max_val=40`, `nerf_beta=1`, and
`rsd_weight=5e-4`.

## Canonical Stage1 Command

```bash
PYTHONPATH=. python scripts/run_cocoa_like_seidel_accuracy_sweep.py \
  --run-name <run_name> \
  --stage stage1 \
  --images Test_figure_1 Iksung_beads dendrites dendrites_dense \
  --directions cocoa_signed signed_balanced \
  --strengths 0.06 0.20 0.40 \
  --seidel-convention classical4d \
  --stage1-size 256 \
  --stage1-pretrain-iter 400 \
  --stage1-num-iter 1000 \
  --pretrain-scalar 5 \
  --lr-obj 0.005 \
  --lr-seidel 0.01 \
  --seidel-optimizer adam \
  --rsd-weight 1e-3 \
  --tv-weight 0 \
  --max-val 20 \
  --nerf-beta 5 \
  --output-mode softplus \
  --scheduler cosine \
  --eta-min-ratio 0.04 \
  --nerf-depth 6 \
  --nerf-width 128 \
  --nerf-skips 2,4,6 \
  --fourier-num-angles 60 \
  --fourier-num-octaves 7 \
  --case-subprocess
```

For two-GPU sharding, add:

```bash
--num-shards 2 --shard-index 0
```

on physical GPU0 and:

```bash
--num-shards 2 --shard-index 1
```

on physical GPU1.

## Evaluation Command

```bash
PYTHONPATH=. python scripts/evaluate_seidel_physical_operator_sweep.py \
  outputs/cocoa_like_2d_mechanism/<run_name>/stage1_metrics.csv \
  outputs/cocoa_like_2d_mechanism/<run_name>/stage1_operator_eval_dim256 \
  --dim 256 \
  --theta-convention classical4d \
  --dataset-twin-invariance-pass true \
  --resume
```

## Reuse Notes

- Use this preset when the scientific question needs a strong, internally
  consistent 4D Adam baseline.
- Do not compare it directly against older 4D runs unless the older run also
  used 4D-consistent GT generation and the same object-prior hyperparameters.
- If changing only the Seidel optimizer, keep every field above fixed and
  change only `--seidel-optimizer`.
