# CoCoA-like 2D Seidel Experiment

这个文件夹是从 `/Users/hongyimac/Desktop/Neural_RIng_AO_Codex` 单独拎出来的实验包，用来复现实验：

- 2D input/output 保持不变。
- Seidel ring forward model 保持不变。
- object/loss 换成 CoCoA-like 机制：Softplus 正值输出、`max_val=40`、`5x measurement` pretrain、RSD contrast prior、默认不加 xy-TV。

## 里面有什么

- `scripts/run_cocoa_like_2d_mechanism.py`：主实验脚本。
- `run_cocoa_like_2d_mechanism.py`：根目录 wrapper，可以直接运行。
- `hybrid_ring_cocoa/`：实验所需的本地 package 代码和数据副本。
- `outputs/cocoa_like_2d_mechanism/cocoa_like2d_fluor256_ucla_pre400_joint1000_seed0/`：已经跑完的结果。

## 复现这次结果

```bash
cd /Users/hongyimac/Desktop/CoCoA_like_2D_Seidel_Experiment
python run_cocoa_like_2d_mechanism.py \
  --image fluorescence \
  --size 256 \
  --modes joint frozen \
  --pretrain-iter 400 \
  --num-iter 1000 \
  --run-name cocoa_like2d_fluor256_ucla_pre400_joint1000_seed0_rerun \
  --verbose
```

## 快速 smoke test

```bash
cd /Users/hongyimac/Desktop/CoCoA_like_2D_Seidel_Experiment
python run_cocoa_like_2d_mechanism.py \
  --image fluorescence \
  --size 64 \
  --modes joint \
  --pretrain-iter 2 \
  --num-iter 2 \
  --run-name smoke_local
```

## Seidel conventions

当前默认路线回到 classical backend 参数化，不再默认使用 Trace-Separated Seidel。默认/推荐模型是：

- `classical4d`: active backend theta `[W040, W131, W222, W220]`，`fixed_seidel_indices = [4, 5]`
- `classical5d`: active backend theta `[W040, W131, W222, W220, W311]`，`fixed_seidel_indices = [5]`
- `classical6d` / `backend6`: full backend theta `[W040, W131, W222, W220, W311, Wd]`

Trace-Separated Seidel 暂时标记为 paused / explicit reproduction only。代码入口仍保留，方便复现已经跑过的 ablation 和 trace5 实验，但不会作为默认矩阵运行：

- `trace5`: public theta `[S, C, A, F, D]`，backend theta `[S, C, 2A, F-A, D, 0]`，`fixed_seidel_indices = [5]`
- `trace4`: public theta `[S, C, A, F]`，backend theta `[S, C, 2A, F-A, 0, 0]`，`fixed_seidel_indices = [4, 5]`
- `trace3`: public theta `[S, C, A]`，backend theta `[S, C, 2A, -A, 0, 0]`，`fixed_seidel_indices = [4, 5]`

所有路径仍然只使用 frozen RDM backend 的 Seidel slot；没有额外的 image-space distortion warp、centroid correction、per-field recentering 或 per-field refocus。`trace4`/`trace3` 在含有非零 `D/W311` 的 ground-truth cases 中仍按设计是 misspecified；当 `F != 0` 时，`trace3` 也按设计是 misspecified。

## Operator metrics

`scripts/run_seidel_blind_recovery_sweep.py` 可以直接从 `RingOperatorProbeEvaluator` 报告 strict/calibrated operator metrics，用于轻量 blind sweep 诊断。

`scripts/evaluate_seidel_physical_operator_sweep.py` 是 post-hoc full physical-equivalence evaluator，会报告 `operator_error_calibrated`、`operator_error_phys_equiv`、`operator_error_coord_diagnostic`、`best_physical_transform`、`best_coordinate_diagnostic_transform` 和 twin gating columns。strict-only metrics 不应被解释为最终 physical-equivalence score。

## 这次已跑出的结论

`frozen GT Seidel` 下 object 很 sharp：`HF=0.3800`，`SSIM=0.9416`。

`joint` 下 object 也明显比 measurement sharp：`HF=0.2519`，但 Seidel 没恢复到 GT，`Seidel L2=0.6948`，所以会出现一些伪结构。

这说明 CoCoA-like object 机制能解决很多 object blur，但 blind joint 的 Seidel/object ambiguity 仍然存在。
