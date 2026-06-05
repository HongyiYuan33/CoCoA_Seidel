#!/usr/bin/env python3
"""Build a Chinese Jupyter notebook for the size512 object-prior sweep.

The notebook is intentionally data-driven: all rankings are recomputed from the
current CSVs, and galleries point to the existing per-case summary images.
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(
    "/Users/hongyimac/Desktop/Neural_RIng_AO_Codex/notebooks/"
    "50_object_prior_operator_report_cn/object_prior_operator_report_cn.ipynb"
)


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


cells = [
    md(
        r"""
# Object-prior Sweep 报告：Operator 与 Object 恢复

这个 notebook 总结 `object_prior_param_sweep_size512_focused_seidelmetric` 的 Stage1 结果，分成两部分：

1. **只从 `operator_error_calibrated` 角度分析**：只问恢复出来的 Seidel 放进 physical forward operator 后是否接近 GT。
2. **同时考虑 `operator_error_calibrated` 和恢复 object 效果**：在 operator 好的基础上，同时看 object 的 SSIM、NRMSE、HF ratio、RSD ratio。

每个展示 case 都配一个四连图：`measurement / recon / forward / heldout_gt`，并在标题里写清楚 image、Seidel profile、object-prior 参数和关键指标。

> 注意：这里刻意不使用 `relative_wavefront_error` 作为排序标准。`relative_wavefront_error` 是 coefficient/WF 逐项接近；本报告的核心是 physical operator 是否等价，以及 object 是否可用。
"""
    ),
    code(
        r"""
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Markdown

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 180)

RUN_NAME = "object_prior_param_sweep_size512_focused_seidelmetric"

ROOT_CANDIDATES = [
    Path("/Users/hongyimac/Desktop/Neural_RIng_AO_Codex"),
    Path("/Users/hongyimac/Desktop/CoCoA_like_2D_Seidel_Experiment"),
]

PROJECT_ROOT = None
RUN_DIR = None
for root in ROOT_CANDIDATES:
    candidate = root / "outputs/cocoa_like_2d_mechanism" / RUN_NAME
    if candidate.exists():
        PROJECT_ROOT = root
        RUN_DIR = candidate
        break

if RUN_DIR is None:
    raise FileNotFoundError("Cannot find object-prior sweep outputs in known local paths.")

STAGE1_CSV = RUN_DIR / "stage1_metrics.csv"
EVAL_DIR = RUN_DIR / "stage1_seidel_physical_operator_eval_dim512"
EVAL_CSV = EVAL_DIR / "seidel_physical_operator_metrics.csv"

stage1 = pd.read_csv(STAGE1_CSV)
df = pd.read_csv(EVAL_CSV)

print("Project root:", PROJECT_ROOT)
print("Run dir:", RUN_DIR)
print("Stage1 rows:", len(stage1))
print("Evaluator rows:", len(df))
"""
    ),
    code(
        r"""
def rsd_human(x):
    x = float(x)
    if abs(x) < 1e-12:
        return "0"
    if abs(x - 5e-4) < 1e-12:
        return "5e-4"
    if abs(x - 1e-3) < 1e-12:
        return "1e-3"
    if abs(x - 2e-3) < 1e-12:
        return "2e-3"
    return f"{x:g}"

def rsd_dir_token(x):
    x = float(x)
    if abs(x) < 1e-12:
        return "0"
    if abs(x - 5e-4) < 1e-12:
        return "5em4"
    if abs(x - 1e-3) < 1e-12:
        return "1em3"
    if abs(x - 2e-3) < 1e-12:
        return "2em3"
    return f"{x:g}".replace("e-", "em").replace(".", "p")

def param_human(row):
    return (
        f"{row['output_mode']} | max_val={int(row['max_val'])}, "
        f"rsd={rsd_human(row['rsd_weight'])}, beta={int(row['nerf_beta'])}, "
        f"tv={row.get('tv_weight', 0):g}"
    )

def param_label(row):
    return (
        f"{row['output_mode']}__max{int(row['max_val'])}"
        f"__rsd{rsd_human(row['rsd_weight'])}__beta{int(row['nerf_beta'])}"
    )

def param_dir_label(row):
    return (
        f"{row['output_mode']}__max{int(row['max_val'])}"
        f"__rsd{rsd_dir_token(row['rsd_weight'])}__beta{int(row['nerf_beta'])}"
    )

def case_dir(row):
    return RUN_DIR / "stage1" / f"{row['image']}__{row['profile_id']}__{param_dir_label(row)}"

def summary_image_path(row):
    return case_dir(row) / "summary_comparison.png"

OP = "operator_error_calibrated"
OBJECT_COLS = [
    "ssim_recon_gain_vs_gt",
    "nrmse_recon_gain_vs_gt",
    "recon_gain_hf_to_gt_hf",
    "recon_gain_rsd_to_gt_rsd",
]

df = df.copy()
df["param_label"] = df.apply(param_label, axis=1)
df["param_pretty"] = df.apply(param_human, axis=1)
df["summary_png"] = df.apply(summary_image_path, axis=1)
df["hf_abs_err"] = (df["recon_gain_hf_to_gt_hf"] - 1).abs()
df["rsd_abs_err"] = (df["recon_gain_rsd_to_gt_rsd"] - 1).abs()

missing_imgs = [p for p in df["summary_png"] if not Path(p).exists()]
print("Missing summary images:", len(missing_imgs))
if missing_imgs[:3]:
    print(missing_imgs[:3])
"""
    ),
    md(
        r"""
## 指标说明

| 指标 | 越好方向 | 这里怎么解释 |
|---|---:|---|
| `operator_error_calibrated` | 越低越好 | 恢复 Seidel 作为 ring-convolution physical forward operator 是否接近 GT |
| `ssim_recon_gain_vs_gt` | 越高越好 | 恢复 object 和 GT object 的结构相似度 |
| `nrmse_recon_gain_vs_gt` | 越低越好 | 恢复 object 和 GT object 的归一化误差 |
| `recon_gain_hf_to_gt_hf` | 越接近 1 越好 | 恢复 object 高频比例是否接近 GT；太低偏糊，太高可能 fake sharpness |
| `recon_gain_rsd_to_gt_rsd` | 越接近 1 越好 | 恢复 object 对比度/RSD 是否接近 GT；太高可能 prior 过强 |
"""
    ),
    md(
        r"""
# Part 1：只从 `operator_error_calibrated` 分析

这一部分只回答一个问题：**恢复出来的 Seidel 进入 physical forward operator 后，和 GT operator 有多像？**

这里不看 `relative_wavefront_error`，也不把 object fidelity 纳入排序。
"""
    ),
    code(
        r"""
config_op = (
    df.groupby("param_label")
    .agg(
        mean_operator_error=(OP, "mean"),
        median_operator_error=(OP, "median"),
        max_operator_error=(OP, "max"),
        std_operator_error=(OP, "std"),
        mean_ssim=("ssim_recon_gain_vs_gt", "mean"),
        mean_nrmse=("nrmse_recon_gain_vs_gt", "mean"),
        mean_hf_ratio=("recon_gain_hf_to_gt_hf", "mean"),
        mean_rsd_ratio=("recon_gain_rsd_to_gt_rsd", "mean"),
        n=(OP, "size"),
    )
    .sort_values("mean_operator_error")
)

display(Markdown("### Operator-only 全局 top 参数"))
display(config_op.head(12).style.format({
    "mean_operator_error": "{:.6f}",
    "median_operator_error": "{:.6f}",
    "max_operator_error": "{:.6f}",
    "std_operator_error": "{:.6f}",
    "mean_ssim": "{:.4f}",
    "mean_nrmse": "{:.4f}",
    "mean_hf_ratio": "{:.3f}",
    "mean_rsd_ratio": "{:.3f}",
}))
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

beta_tbl = df.groupby("nerf_beta")[OP].mean().sort_index()
rsd_tbl = df.groupby("rsd_weight")[OP].mean().sort_index()
max_tbl = df.groupby("max_val")[OP].mean().sort_index()

axes[0].bar([str(int(x)) for x in beta_tbl.index], beta_tbl.values, color="#4C78A8")
axes[0].set_title("nerf_beta 趋势")
axes[0].set_xlabel("beta")
axes[0].set_ylabel("mean operator error")

axes[1].bar([rsd_human(x) for x in rsd_tbl.index], rsd_tbl.values, color="#59A14F")
axes[1].set_title("RSD weight 趋势")
axes[1].set_xlabel("rsd_weight")

axes[2].bar([str(int(x)) for x in max_tbl.index], max_tbl.values, color="#F28E2B")
axes[2].set_title("max_val 趋势")
axes[2].set_xlabel("max_val")

for ax in axes:
    ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

display(Markdown("### 参数主效应表"))
display(pd.DataFrame({
    "beta_mean_operator_error": beta_tbl,
}).style.format("{:.6f}"))
display(pd.DataFrame({
    "rsd_mean_operator_error": rsd_tbl,
}).style.format("{:.6f}"))
display(pd.DataFrame({
    "max_val_mean_operator_error": max_tbl,
}).style.format("{:.6f}"))
"""
    ),
    code(
        r"""
pivot_rsd_beta = df.pivot_table(
    index="rsd_weight", columns="nerf_beta", values=OP, aggfunc="mean"
)

fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.imshow(pivot_rsd_beta.values, cmap="viridis_r")
ax.set_xticks(range(len(pivot_rsd_beta.columns)))
ax.set_xticklabels([f"beta={int(x)}" for x in pivot_rsd_beta.columns])
ax.set_yticks(range(len(pivot_rsd_beta.index)))
ax.set_yticklabels([f"rsd={rsd_human(x)}" for x in pivot_rsd_beta.index])
ax.set_title("RSD × beta 的 operator error 热力图")
for i in range(pivot_rsd_beta.shape[0]):
    for j in range(pivot_rsd_beta.shape[1]):
        ax.text(j, i, f"{pivot_rsd_beta.values[i, j]:.4f}", ha="center", va="center", color="white")
fig.colorbar(im, ax=ax, label="mean operator error")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
profile_best = (
    df.groupby(["profile_id", "param_label"])
    .agg(
        mean_operator_error=(OP, "mean"),
        max_operator_error=(OP, "max"),
        mean_ssim=("ssim_recon_gain_vs_gt", "mean"),
        n=(OP, "size"),
    )
    .reset_index()
    .sort_values(["profile_id", "mean_operator_error"])
    .groupby("profile_id")
    .head(5)
)

image_difficulty = (
    df.groupby("image")
    .agg(
        mean_operator_error=(OP, "mean"),
        median_operator_error=(OP, "median"),
        min_operator_error=(OP, "min"),
        max_operator_error=(OP, "max"),
        mean_ssim=("ssim_recon_gain_vs_gt", "mean"),
        n=(OP, "size"),
    )
    .sort_values("mean_operator_error")
)

display(Markdown("### 每个 Seidel profile 的 operator-only top 参数"))
display(profile_best.style.format({
    "mean_operator_error": "{:.6f}",
    "max_operator_error": "{:.6f}",
    "mean_ssim": "{:.4f}",
}))

display(Markdown("### 四张图从 operator 角度的难度"))
display(image_difficulty.style.format({
    "mean_operator_error": "{:.6f}",
    "median_operator_error": "{:.6f}",
    "min_operator_error": "{:.6f}",
    "max_operator_error": "{:.6f}",
    "mean_ssim": "{:.4f}",
}))
"""
    ),
    code(
        r"""
def crop_case_quad(summary_path):
    # Return four cropped panels: measurement, recon, forward, heldout_gt.
    # Existing summary_comparison.png has five panels:
    # GT / Measurement / joint recon / Pred meas / Gain abs err.
    # We reuse it and reorder to match the requested four-panel layout.
    img = Image.open(summary_path).convert("RGB")
    w, h = img.size
    panel_w = w // 5
    # Crop the image area under the old titles. These constants fit the saved
    # matplotlib summary layout and degrade gracefully if dimensions differ.
    y0 = int(h * 0.075)
    y1 = int(h * 0.92)
    margin_x = max(6, int(panel_w * 0.04))
    source_indices = [1, 2, 3, 0]  # Measurement, joint recon, Pred meas, GT
    crops = []
    for idx in source_indices:
        x0 = idx * panel_w + margin_x
        x1 = (idx + 1) * panel_w - margin_x
        crops.append(img.crop((x0, y0, x1, y1)))
    return crops

def show_case_quad(row, prefix="", compact=False):
    p = Path(row["summary_png"])
    if not p.exists():
        display(Markdown(f"Missing summary image: `{p}`"))
        return

    panels = crop_case_quad(p)
    titles = ["measurement", "recon", "forward", "heldout_gt"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.1))
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, cmap="gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    title = (
        f"{prefix}{row['image']} | {row['profile_id']} | "
        f"max={int(row['max_val'])}, rsd={rsd_human(row['rsd_weight'])}, beta={int(row['nerf_beta'])} | "
        f"op={row[OP]:.5f}, SSIM={row['ssim_recon_gain_vs_gt']:.4f}, "
        f"NRMSE={row['nrmse_recon_gain_vs_gt']:.4f}, "
        f"HF={row['recon_gain_hf_to_gt_hf']:.3f}, RSD={row['recon_gain_rsd_to_gt_rsd']:.3f}"
    )
    fig.suptitle(title, fontsize=12, y=1.04)
    plt.tight_layout()
    plt.show()

N_SHOW_OPERATOR = 8
operator_cases = df.sort_values(OP).head(N_SHOW_OPERATOR)

display(Markdown(f"### Operator-only top {N_SHOW_OPERATOR} single cases 四连图"))
display(operator_cases[[
    "image", "profile_id", "param_label", OP, "best_physical_transform",
    "ssim_recon_gain_vs_gt", "nrmse_recon_gain_vs_gt",
    "recon_gain_hf_to_gt_hf", "recon_gain_rsd_to_gt_rsd",
]].style.format({
    OP: "{:.6f}",
    "ssim_recon_gain_vs_gt": "{:.4f}",
    "nrmse_recon_gain_vs_gt": "{:.4f}",
    "recon_gain_hf_to_gt_hf": "{:.3f}",
    "recon_gain_rsd_to_gt_rsd": "{:.3f}",
}))

for _, row in operator_cases.iterrows():
    show_case_quad(row, prefix="Operator-only top case | ")
"""
    ),
    md(
        r"""
## Part 1 小结

只看 `operator_error_calibrated` 时，最强信号是：

- `nerf_beta=5` 明显最好。
- `max_val=10/20/40` 差异很小，不是决定因素。
- `rsd_weight` 最优区域偏向中高值；在 `beta=5` 下，`rsd=1e-3` 是全局最稳点。
- 不同 Seidel profile 对 RSD 强度有差异：`cocoa_signed__rms0p10` 更喜欢强一点的 `rsd=2e-3`，而两个 `0.06 RMS` profile 更偏 `5e-4`。
"""
    ),
    md(
        r"""
# Part 2：同时考虑 operator 和 object 恢复

这一部分把排序改成折中目标：

```text
operator_error_calibrated 越低越好
SSIM 越高越好
NRMSE 越低越好
HF ratio 越接近 1 越好
RSD ratio 越接近 1 越好
```

这里的想法是避免两种偏科：

- 只让 operator 像，但 object 明显变糊或变形。
- 只让 object 看起来 sharp，但可能靠过强 RSD 产生 fake sharpness。
"""
    ),
    code(
        r"""
config_bal = config_op.copy()
config_bal["hf_abs_error"] = (config_bal["mean_hf_ratio"] - 1).abs()
config_bal["rsd_abs_error"] = (config_bal["mean_rsd_ratio"] - 1).abs()

rank_specs = {
    "mean_operator_error": True,
    "mean_ssim": False,
    "mean_nrmse": True,
    "hf_abs_error": True,
    "rsd_abs_error": True,
}
for col, ascending in rank_specs.items():
    config_bal[col + "_rank"] = config_bal[col].rank(ascending=ascending, method="min")

config_bal["balanced_rank_score"] = (
    0.40 * config_bal["mean_operator_error_rank"]
    + 0.20 * config_bal["mean_ssim_rank"]
    + 0.20 * config_bal["mean_nrmse_rank"]
    + 0.10 * config_bal["hf_abs_error_rank"]
    + 0.10 * config_bal["rsd_abs_error_rank"]
)

balanced_top = config_bal.sort_values("balanced_rank_score")

display(Markdown("### Operator + object 综合 top 参数"))
display(balanced_top.head(12)[[
    "mean_operator_error", "mean_ssim", "mean_nrmse",
    "mean_hf_ratio", "hf_abs_error", "mean_rsd_ratio", "rsd_abs_error",
    "balanced_rank_score",
]].style.format({
    "mean_operator_error": "{:.6f}",
    "mean_ssim": "{:.4f}",
    "mean_nrmse": "{:.4f}",
    "mean_hf_ratio": "{:.3f}",
    "hf_abs_error": "{:.3f}",
    "mean_rsd_ratio": "{:.3f}",
    "rsd_abs_error": "{:.3f}",
    "balanced_rank_score": "{:.2f}",
}))
"""
    ),
    code(
        r"""
plot_df = balanced_top.head(10).copy()
labels = [x.replace("softplus__", "").replace("__", "\n") for x in plot_df.index]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

axes[0].barh(labels[::-1], plot_df["mean_operator_error"].values[::-1], color="#4C78A8")
axes[0].set_title("operator error")
axes[0].set_xlabel("lower is better")

axes[1].barh(labels[::-1], plot_df["mean_ssim"].values[::-1], color="#59A14F")
axes[1].set_title("object SSIM")
axes[1].set_xlabel("higher is better")

axes[2].barh(labels[::-1], plot_df["mean_hf_ratio"].values[::-1], color="#E15759")
axes[2].axvline(1.0, color="black", lw=1, alpha=0.6)
axes[2].set_title("HF ratio: recon / GT")
axes[2].set_xlabel("closer to 1 is better")

for ax in axes:
    ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
case_df = df.copy()

# Per-case balanced score, using ranks across all 432 cases.
case_df["hf_abs_err"] = (case_df["recon_gain_hf_to_gt_hf"] - 1).abs()
case_df["rsd_abs_err"] = (case_df["recon_gain_rsd_to_gt_rsd"] - 1).abs()

case_df["op_rank"] = case_df[OP].rank(ascending=True, method="min")
case_df["ssim_rank"] = case_df["ssim_recon_gain_vs_gt"].rank(ascending=False, method="min")
case_df["nrmse_rank"] = case_df["nrmse_recon_gain_vs_gt"].rank(ascending=True, method="min")
case_df["hf_rank"] = case_df["hf_abs_err"].rank(ascending=True, method="min")
case_df["rsd_rank"] = case_df["rsd_abs_err"].rank(ascending=True, method="min")

case_df["balanced_case_score"] = (
    0.40 * case_df["op_rank"]
    + 0.20 * case_df["ssim_rank"]
    + 0.20 * case_df["nrmse_rank"]
    + 0.10 * case_df["hf_rank"]
    + 0.10 * case_df["rsd_rank"]
)

N_SHOW_BALANCED = 8
balanced_cases = case_df.sort_values("balanced_case_score").head(N_SHOW_BALANCED)

display(Markdown(f"### Operator + object 综合 top {N_SHOW_BALANCED} single cases 四连图"))
display(balanced_cases[[
    "image", "profile_id", "param_label", OP,
    "ssim_recon_gain_vs_gt", "nrmse_recon_gain_vs_gt",
    "recon_gain_hf_to_gt_hf", "recon_gain_rsd_to_gt_rsd",
    "balanced_case_score",
]].style.format({
    OP: "{:.6f}",
    "ssim_recon_gain_vs_gt": "{:.4f}",
    "nrmse_recon_gain_vs_gt": "{:.4f}",
    "recon_gain_hf_to_gt_hf": "{:.3f}",
    "recon_gain_rsd_to_gt_rsd": "{:.3f}",
    "balanced_case_score": "{:.1f}",
}))

for _, row in balanced_cases.iterrows():
    show_case_quad(row, prefix="Balanced top case | ")
"""
    ),
    code(
        r"""
display(Markdown("### 三个推荐区域的直观对比"))

recommended = [
    "softplus__max20__rsd1e-3__beta5",
    "softplus__max40__rsd5e-4__beta5",
    "softplus__max10__rsd2e-3__beta2",
]

recommend_table = config_bal.loc[recommended, [
    "mean_operator_error", "mean_ssim", "mean_nrmse",
    "mean_hf_ratio", "mean_rsd_ratio", "balanced_rank_score",
]].copy()

recommend_table.index = [
    "首选：operator + object 平衡",
    "保守：object fidelity 更稳",
    "更锐：object HF 更高，但更 aggressive",
]

display(recommend_table.style.format({
    "mean_operator_error": "{:.6f}",
    "mean_ssim": "{:.4f}",
    "mean_nrmse": "{:.4f}",
    "mean_hf_ratio": "{:.3f}",
    "mean_rsd_ratio": "{:.3f}",
    "balanced_rank_score": "{:.2f}",
}))
"""
    ),
    md(
        r"""
## Part 2 小结

综合 operator 和 object 后，推荐排序是：

1. **首选**：`softplus__max20__rsd1e-3__beta5`
   - operator error 最低；
   - object SSIM/NRMSE 没有明显牺牲；
   - HF/RSD 比 `5e-4` 更接近 GT，但没有 `2e-3` 那么 aggressive。

2. **保守备选**：`softplus__max40__rsd5e-4__beta5`
   - object fidelity 更稳，SSIM/NRMSE 略好；
   - 但 HF ratio 偏低，object 会相对更糊一点。

3. **sharpness 对照**：`softplus__max10__rsd2e-3__beta2`
   - HF ratio 更接近 GT，视觉上可能更锐；
   - 但它不是 operator-only 最优，并且更容易进入“prior 过强”的区域，所以建议作为对照，而不是默认。

一句话总结：

```text
如果目标是 physical operator 等价 + object 可用性，当前最合理默认是
softplus, max_val=20, rsd_weight=1e-3, nerf_beta=5, tv_weight=0。
```
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
