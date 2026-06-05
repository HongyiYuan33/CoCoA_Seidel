from __future__ import annotations

import nbformat as nbf
from pathlib import Path


HERE = Path(__file__).resolve().parent
NOTEBOOK_PATH = HERE / "seidel_sweep_report_cn.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
    md(
        """
# 中文 Seidel Sweep Report

这份 notebook 汇总 `seidel_recovery_sweep_20260525_full` 的三阶段结果，重点用表格、统计图和视觉化图回答一个问题：

> 在目前的 **CoCoA-like object + joint Seidel** 设置下，什么样的 Seidel wavefront 最容易被恢复？

核心提醒：这里的“最容易恢复”是 **相对当前 sweep 内其他候选更好**，不等于 Seidel 已经恢复得非常准。最后的结论会同时看 wavefront relative error、object SSIM 和 seed 稳定性。
"""
    ),
    md(
        """
## 0. 实验设计速览

这次 sweep 分成三层：

| 阶段 | 设置 | 目的 |
|---|---:|---|
| Stage 1 | `size=128`, `pretrain=200`, `joint=500`, `seed=0` | 快筛 4 张图 × 7 个方向 × 6 个 RMS，共 168 个 case |
| Stage 2 | `size=256`, `pretrain=400`, `joint=1000`, `seed=0` | 对 Stage 1 选出的候选做 full setting 确认 |
| Stage 3 | `size=256`, `pretrain=400`, `joint=1000`, `seed=1,2` | 对 Stage 2 top 3 做随机初始化稳定性验证 |

四张图：`Test_figure_1`、`Iksung_beads`、`dendrites`、`dendrites_dense`。主指标是 `relative_wavefront_error = RMS(recovered - GT) / RMS(GT)`，越低越好。
"""
    ),
    code(
        r"""
from pathlib import Path
import ast
import json
import math
import textwrap

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from IPython.display import HTML, Image, Markdown, display

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 160,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.sans-serif": [
        "PingFang SC", "Heiti TC", "STHeiti", "Arial Unicode MS",
        "Noto Sans CJK SC", "SimHei", "DejaVu Sans"
    ],
    "axes.unicode_minus": False,
})

RUN_NAME = "seidel_recovery_sweep_20260525_full"
REL_OUTPUT = Path("outputs") / "cocoa_like_2d_mechanism" / RUN_NAME
CODEX_ROOT = Path("/Users/hongyimac/Desktop/Neural_RIng_AO_Codex")
DESKTOP_EXPERIMENT_ROOT = Path("/Users/hongyimac/Desktop/CoCoA_like_2D_Seidel_Experiment")

def find_project_root():
    candidates = []
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    candidates.extend([CODEX_ROOT, DESKTOP_EXPERIMENT_ROOT])
    for base in candidates:
        if (base / REL_OUTPUT / "stage1_metrics.csv").is_file():
            return base
    raise FileNotFoundError(f"找不到 sweep 输出目录：*/{REL_OUTPUT}")

PROJECT_ROOT = find_project_root()
SWEEP_ROOT = PROJECT_ROOT / REL_OUTPUT
print(f"项目根目录: {PROJECT_ROOT}")
print(f"Sweep 输出: {SWEEP_ROOT}")
"""
    ),
    code(
        r"""
def read_metrics_csv(name):
    path = SWEEP_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

stage1 = read_metrics_csv("stage1_metrics.csv")
stage2 = read_metrics_csv("stage2_metrics.csv")
stage3_raw = read_metrics_csv("stage3_metrics_raw.csv")
stage3_stability = read_metrics_csv("stage3_seed_stability.csv")

expected_counts = {
    "Stage 1 快筛": (stage1, 168),
    "Stage 2 full": (stage2, 36),
    "Stage 3 raw": (stage3_raw, 24),
    "Stage 3 seed stability": (stage3_stability, 15),
}
for label, (df, expected) in expected_counts.items():
    assert len(df) == expected, f"{label} 行数异常：{len(df)} != {expected}"

assert "ALL" in set(stage3_stability["image"]), "Stage 3 稳定性表缺少 ALL 聚合行"

count_table = pd.DataFrame([
    {"数据表": label, "实际行数": len(df), "期望行数": expected, "状态": "OK"}
    for label, (df, expected) in expected_counts.items()
])
display(count_table)
"""
    ),
    md(
        """
## 1. 指标解释表

下面这张表是读后面所有结果的钥匙。主结论只按 `relative_wavefront_error` 排名；object SSIM 用来判断恢复出来的 object 是否还能用。
"""
    ),
    code(
        r"""
metric_table = pd.DataFrame([
    {
        "指标": "relative_wavefront_error",
        "中文解释": "恢复 wavefront 与 GT wavefront 的 RMS 差，除以 GT wavefront RMS",
        "方向": "越低越好",
        "用途": "主排名指标；最接近物理 Seidel 恢复准确度",
    },
    {
        "指标": "wavefront_error_rms",
        "中文解释": "恢复 wavefront 与 GT wavefront 的绝对 RMS 差",
        "方向": "越低越好",
        "用途": "绝对误差，单位是 waves",
    },
    {
        "指标": "seidel_l2_relative",
        "中文解释": "6D Seidel coefficient 的相对 L2 误差",
        "方向": "越低越好",
        "用途": "辅助看 coefficient 是否接近；不如 wavefront RMS 物理直观",
    },
    {
        "指标": "ssim_recon_gain_vs_gt",
        "中文解释": "gain-aligned object reconstruction 与 sharp GT 的 SSIM",
        "方向": "越高越好",
        "用途": "判断 object 是否可用",
    },
    {
        "指标": "nrmse_recon_gain_vs_gt",
        "中文解释": "gain-aligned object reconstruction 与 sharp GT 的 NRMSE",
        "方向": "越低越好",
        "用途": "辅助判断 object 误差",
    },
    {
        "指标": "measurement_hf_drop",
        "中文解释": "measurement 高频比例相对 GT sharp object 的下降",
        "方向": "越高表示测量越被 blur 低通",
        "用途": "理解这个 case 对 object/Seidel 的难度",
    },
])
display(metric_table)
"""
    ),
    md(
        r"""
## 2. 实验空间：7 个方向 × 6 个 RMS

所有方向都先按项目里的 Seidel polynomial 计算 field-weighted wavefront RMS，再缩放到目标 RMS。`0.04 waves` 是小像差对照，不作为主要结论优先级。

在这份报告里，WF 会同时用 **field-weighted RMS** 和 **具体 6D Seidel coefficient** 表示。项目里的 Seidel wavefront 写成：

\[
W(x,y,H) =
W_{040}\rho^4
+ W_{131}H\rho^2x
+ W_{222}H^2x^2
+ W_{220}H^2\rho^2
+ W_{311}H^3x
+ W_d\rho^2
\]

其中 \(\rho^2=x^2+y^2\)，\(H\) 是 field radius。RMS 计算时会在 pupil 内去 piston，然后按 field area weighting 做平均。
"""
    ),
    code(
        r"""
coeff_names = ["W040", "W131", "W222", "W220", "W311", "Wd"]
direction_order = [
    "pos_balanced", "signed_balanced", "cocoa_signed", "coma_dominant",
    "astig_field", "spherical_defocus", "distortion_mixed",
]
rms_order = sorted(stage1["target_wavefront_rms"].unique())

design_table = (
    stage1[["candidate_id", "direction", "target_wavefront_rms", "actual_wavefront_rms"]]
    .drop_duplicates()
    .sort_values(["direction", "target_wavefront_rms"])
)
design_table["direction"] = pd.Categorical(design_table["direction"], direction_order, ordered=True)
design_table = design_table.sort_values(["direction", "target_wavefront_rms"])
display(design_table.rename(columns={
    "candidate_id": "候选 ID",
    "direction": "Seidel 方向",
    "target_wavefront_rms": "目标 RMS (waves)",
    "actual_wavefront_rms": "实际 RMS (waves)",
}).head(50))
"""
    ),
    code(
        r"""
config = json.loads((SWEEP_ROOT / "sweep_config.json").read_text())
seidel_table = pd.DataFrame([
    {
        "candidate_id": c["candidate_id"],
        "direction": c["direction"],
        "target_wavefront_rms": c["target_rms"],
        "actual_wavefront_rms": c["actual_rms"],
        **{name: value for name, value in zip(coeff_names, c["seidel"])},
    }
    for c in config["candidates"]
])
seidel_table["direction"] = pd.Categorical(seidel_table["direction"], direction_order, ordered=True)
seidel_table = seidel_table.sort_values(["direction", "target_wavefront_rms"]).reset_index(drop=True)

display(Markdown("### 所有 candidate 的具体 Seidel WF 表达"))
display(
    seidel_table.rename(columns={
        "candidate_id": "候选 ID",
        "direction": "Seidel 方向",
        "target_wavefront_rms": "目标 RMS (waves)",
        "actual_wavefront_rms": "实际 RMS (waves)",
    }).style.format({
        "目标 RMS (waves)": "{:.3f}",
        "实际 RMS (waves)": "{:.3f}",
        **{name: "{:.6f}" for name in coeff_names},
    }).background_gradient(subset=coeff_names, cmap="coolwarm", axis=None)
)
"""
    ),
    md(
        """
## 3. Stage 1 快筛：全局趋势表

Stage 1 的作用是快速看趋势。这里每个 candidate 都在四张图上跑一次，然后按四张图聚合。
"""
    ),
    code(
        r"""
def aggregate_candidate(df):
    grouped = (
        df.groupby(["candidate_id", "direction", "target_wavefront_rms"], as_index=False)
        .agg(
            平均相对WF误差=("relative_wavefront_error", "mean"),
            中位相对WF误差=("relative_wavefront_error", "median"),
            最大相对WF误差=("relative_wavefront_error", "max"),
            相对WF误差标准差=("relative_wavefront_error", "std"),
            平均SSIM=("ssim_recon_gain_vs_gt", "mean"),
            平均NRMSE=("nrmse_recon_gain_vs_gt", "mean"),
            平均测量HF下降=("measurement_hf_drop", "mean"),
            图像数=("image", "nunique"),
        )
        .sort_values(["平均相对WF误差", "最大相对WF误差"])
        .reset_index(drop=True)
    )
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))
    return grouped

stage1_summary = aggregate_candidate(stage1)
stage2_summary = aggregate_candidate(stage2)

stage1_show = stage1_summary.head(20).copy()
display(
    stage1_show.style
    .format({
        "target_wavefront_rms": "{:.3f}",
        "平均相对WF误差": "{:.3f}",
        "中位相对WF误差": "{:.3f}",
        "最大相对WF误差": "{:.3f}",
        "相对WF误差标准差": "{:.3f}",
        "平均SSIM": "{:.3f}",
        "平均NRMSE": "{:.3f}",
        "平均测量HF下降": "{:.3f}",
    })
    .background_gradient(subset=["平均相对WF误差", "最大相对WF误差"], cmap="YlGn_r")
    .background_gradient(subset=["平均SSIM"], cmap="YlGn")
)
"""
    ),
    code(
        r"""
fig, ax = plt.subplots(figsize=(11, 11))
candidate_order = (
    stage1[["candidate_id", "direction", "target_wavefront_rms"]]
    .drop_duplicates()
    .assign(direction=lambda d: pd.Categorical(d["direction"], direction_order, ordered=True))
    .sort_values(["direction", "target_wavefront_rms", "candidate_id"])
)
pivot = stage1.pivot_table(
    index="candidate_id",
    columns="image",
    values="relative_wavefront_error",
    aggfunc="mean",
).reindex(candidate_order["candidate_id"])
im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r", vmin=0, vmax=np.nanpercentile(pivot.values, 95))
ax.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns, rotation=25, ha="right")
labels = [
    f"{row.direction}\\n{row.target_wavefront_rms:.2f}"
    for row in candidate_order.itertuples(index=False)
]
ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=7)
ax.set_title("Stage 1：四张图上的 relative wavefront error 热力图（越深越好）")
fig.colorbar(im, ax=ax, label="relative wavefront error")
fig.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

data_by_rms = [
    stage1.loc[np.isclose(stage1["target_wavefront_rms"], rms), "relative_wavefront_error"].values
    for rms in rms_order
]
axes[0].boxplot(data_by_rms, labels=[f"{r:.2f}" for r in rms_order], showmeans=True)
axes[0].set_title("Stage 1：不同 RMS 强度下的恢复误差分布")
axes[0].set_xlabel("目标 wavefront RMS (waves)")
axes[0].set_ylabel("relative wavefront error")

data_by_dir = [
    stage1.loc[stage1["direction"] == d, "relative_wavefront_error"].values
    for d in direction_order
]
axes[1].violinplot(data_by_dir, showmeans=True, showmedians=True)
axes[1].set_xticks(np.arange(1, len(direction_order) + 1), labels=direction_order, rotation=35, ha="right")
axes[1].set_title("Stage 1：不同 Seidel 方向的恢复误差分布")
axes[1].set_ylabel("relative wavefront error")

fig.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 4. Stage 2 full setting：正式候选确认

Stage 2 只跑 Stage 1 选出的 9 个候选，但使用 `size=256` 和更长训练。这个阶段比 Stage 1 更适合排名。
"""
    ),
    code(
        r"""
assert stage2_summary.iloc[0]["candidate_id"] == "cocoa_signed__rms0p06", (
    "Stage 2 第一名不是预期的 cocoa_signed__rms0p06，请检查输入数据是否换过"
)

stage2_show = stage2_summary.merge(seidel_table[["candidate_id", *coeff_names]], on="candidate_id", how="left")
display(
    stage2_show.style
    .format({
        "target_wavefront_rms": "{:.3f}",
        "平均相对WF误差": "{:.3f}",
        "中位相对WF误差": "{:.3f}",
        "最大相对WF误差": "{:.3f}",
        "相对WF误差标准差": "{:.3f}",
        "平均SSIM": "{:.3f}",
        "平均NRMSE": "{:.3f}",
        "平均测量HF下降": "{:.3f}",
        **{name: "{:.6f}" for name in coeff_names},
    })
    .background_gradient(subset=["平均相对WF误差", "最大相对WF误差"], cmap="YlGn_r")
    .background_gradient(subset=["平均SSIM"], cmap="YlGn")
    .background_gradient(subset=coeff_names, cmap="coolwarm", axis=None)
)
"""
    ),
    code(
        r"""
stage1_rank = stage1_summary[["candidate_id", "rank"]].rename(columns={"rank": "Stage 1 rank"})
stage2_rank = stage2_summary[["candidate_id", "rank", "direction", "target_wavefront_rms", "平均相对WF误差", "平均SSIM"]].rename(columns={"rank": "Stage 2 rank"})
rank_compare = stage2_rank.merge(stage1_rank, on="candidate_id", how="left")
rank_compare = rank_compare.sort_values("Stage 2 rank")

fig, ax = plt.subplots(figsize=(8, 5))
for row in rank_compare.to_dict("records"):
    ax.plot([1, 2], [row["Stage 1 rank"], row["Stage 2 rank"]], marker="o", linewidth=1.8)
    ax.text(2.03, row["Stage 2 rank"], row["candidate_id"], va="center", fontsize=8)
ax.set_xlim(0.8, 3.1)
ax.set_xticks([1, 2], labels=["Stage 1 快筛排名", "Stage 2 full 排名"])
ax.invert_yaxis()
ax.set_ylabel("排名（越靠上越好）")
ax.set_title("Stage 1 到 Stage 2 的排名变化")
fig.tight_layout()
plt.show()

display(rank_compare.rename(columns={
    "candidate_id": "候选 ID",
    "direction": "方向",
    "target_wavefront_rms": "目标 RMS",
    "平均相对WF误差": "Stage 2 平均相对WF误差",
    "平均SSIM": "Stage 2 平均SSIM",
}))
"""
    ),
    code(
        r"""
fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(
    stage2["ssim_recon_gain_vs_gt"],
    stage2["relative_wavefront_error"],
    c=stage2["target_wavefront_rms"],
    s=80,
    cmap="viridis",
    alpha=0.85,
    edgecolor="white",
    linewidth=0.7,
)
for row in stage2.itertuples(index=False):
    if row.candidate_id == "cocoa_signed__rms0p06":
        ax.annotate(row.image, (row.ssim_recon_gain_vs_gt, row.relative_wavefront_error), fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("object SSIM（越高越好）")
ax.set_ylabel("relative wavefront error（越低越好）")
ax.set_title("Stage 2：object 可用性 vs Seidel 恢复准确度")
fig.colorbar(scatter, ax=ax, label="目标 RMS (waves)")
fig.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 5. Stage 3 seed 稳定性：top 3 是否可靠？

Stage 3 只验证 Stage 2 的 top 3：`cocoa_signed__rms0p06`、`cocoa_signed__rms0p1`、`cocoa_signed__rms0p12`。每个候选在四张图上用 seed=1 和 seed=2 再跑。
"""
    ),
    code(
        r"""
stage3_all = (
    stage3_stability[stage3_stability["image"] == "ALL"]
    .sort_values("mean_relative_wavefront_error")
    .reset_index(drop=True)
)
stage3_all_with_seidel = stage3_all.merge(seidel_table[["candidate_id", *coeff_names]], on="candidate_id", how="left")
display(
    stage3_all_with_seidel.rename(columns={
        "candidate_id": "候选 ID",
        "direction": "方向",
        "target_wavefront_rms": "目标 RMS",
        "mean_relative_wavefront_error": "平均相对WF误差",
        "std_relative_wavefront_error": "相对WF误差标准差",
        "min_relative_wavefront_error": "最小相对WF误差",
        "max_relative_wavefront_error": "最大相对WF误差",
        "mean_ssim": "平均SSIM",
        "std_ssim": "SSIM标准差",
        "mean_nrmse": "平均NRMSE",
        "num_runs": "运行数",
        "seeds": "seeds",
    })[
        ["候选 ID", "方向", "目标 RMS", *coeff_names, "平均相对WF误差", "相对WF误差标准差", "最小相对WF误差", "最大相对WF误差", "平均SSIM", "SSIM标准差", "平均NRMSE", "运行数", "seeds"]
    ].style.format({
        "目标 RMS": "{:.3f}",
        **{name: "{:.6f}" for name in coeff_names},
        "平均相对WF误差": "{:.3f}",
        "相对WF误差标准差": "{:.3f}",
        "最小相对WF误差": "{:.3f}",
        "最大相对WF误差": "{:.3f}",
        "平均SSIM": "{:.3f}",
        "SSIM标准差": "{:.3f}",
        "平均NRMSE": "{:.3f}",
    }).background_gradient(subset=["平均相对WF误差", "最大相对WF误差"], cmap="YlGn_r")
    .background_gradient(subset=coeff_names, cmap="coolwarm", axis=None)
)
"""
    ),
    code(
        r"""
fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
x = np.arange(len(stage3_all))
bars = ax1.bar(
    x,
    stage3_all["mean_relative_wavefront_error"],
    yerr=stage3_all["std_relative_wavefront_error"],
    capsize=5,
    color=["#4c78a8", "#72b7b2", "#f58518"],
    alpha=0.9,
)
ax1.set_xticks(x, labels=stage3_all["candidate_id"], rotation=18, ha="right")
ax1.set_ylabel("mean relative wavefront error")
ax1.set_title("Stage 3：seed 稳定性（误差均值 ± 标准差）")
ax1.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.55, label="误差 = GT RMS")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(x, stage3_all["mean_ssim"], color="#54a24b", marker="o", linewidth=2, label="平均 object SSIM")
ax2.set_ylabel("mean object SSIM")
ax2.set_ylim(0.82, 1.0)
ax2.legend(loc="upper right")

for rect, val in zip(bars, stage3_all["mean_relative_wavefront_error"]):
    ax1.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
per_image_stage3 = (
    stage3_stability[stage3_stability["image"] != "ALL"]
    .sort_values(["candidate_id", "image"])
    .rename(columns={
        "candidate_id": "候选 ID",
        "image": "图像",
        "target_wavefront_rms": "目标 RMS",
        "mean_relative_wavefront_error": "平均相对WF误差",
        "std_relative_wavefront_error": "相对WF误差标准差",
        "mean_ssim": "平均SSIM",
        "std_ssim": "SSIM标准差",
        "mean_nrmse": "平均NRMSE",
    })
)
display(
    per_image_stage3[["候选 ID", "图像", "目标 RMS", "平均相对WF误差", "相对WF误差标准差", "平均SSIM", "SSIM标准差", "平均NRMSE"]]
    .style.format({
        "目标 RMS": "{:.3f}",
        "平均相对WF误差": "{:.3f}",
        "相对WF误差标准差": "{:.3f}",
        "平均SSIM": "{:.3f}",
        "SSIM标准差": "{:.3f}",
        "平均NRMSE": "{:.3f}",
    })
    .background_gradient(subset=["平均相对WF误差"], cmap="YlGn_r")
    .background_gradient(subset=["平均SSIM"], cmap="YlGn")
)
"""
    ),
    md(
        """
## 6. Top candidate 的 Seidel coefficient 细节

下面只看 Stage 2 第一名 `cocoa_signed__rms0p06`。这张表不是最终排名依据，但能帮助理解：有些 coefficient L2 看起来不小，物理 wavefront RMS 排名仍可能更好，因为不同 coefficient 对 pupil/field 的影响权重不同。
"""
    ),
    code(
        r"""
def parse_vec(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    return np.asarray(ast.literal_eval(str(value)), dtype=float)

best_id = stage2_summary.iloc[0]["candidate_id"]
best_rows = stage2[stage2["candidate_id"] == best_id].copy().sort_values("image")
coeff_names = ["W040", "W131", "W222", "W220", "W311", "Wd"]

detail_rows = []
for row in best_rows.itertuples(index=False):
    gt = parse_vec(row.seidel_gt)
    rec = parse_vec(row.seidel_final)
    detail = {
        "图像": row.image,
        "relative WF error": row.relative_wavefront_error,
        "object SSIM": row.ssim_recon_gain_vs_gt,
        "object NRMSE": row.nrmse_recon_gain_vs_gt,
        "coefficient relative L2": row.seidel_l2_relative,
    }
    for name, g, r in zip(coeff_names, gt, rec):
        detail[f"{name} GT"] = g
        detail[f"{name} recovered"] = r
        detail[f"{name} diff"] = r - g
    detail_rows.append(detail)

best_coeff_table = pd.DataFrame(detail_rows)
display(
    best_coeff_table.style.format({col: "{:.4f}" for col in best_coeff_table.columns if col != "图像"})
    .background_gradient(subset=["relative WF error", "coefficient relative L2"], cmap="YlGn_r")
    .background_gradient(subset=["object SSIM"], cmap="YlGn")
)
"""
    ),
    md(
        """
## 7. 视觉化图：原始 overview / heatmap / scatter

这些图是 sweep driver 直接生成的 artifact。它们适合放进汇报里做快速视觉总结。
"""
    ),
    code(
        r"""
for filename, title in [
    ("overview_top_candidates.png", "Top candidates overview"),
    ("heatmap_relative_wavefront_error.png", "Stage 1 relative wavefront error heatmap"),
    ("scatter_recoverability_vs_rms.png", "Recoverability vs RMS scatter"),
]:
    path = SWEEP_ROOT / filename
    display(Markdown(f"### {title}\\n`{path}`"))
    display(Image(filename=str(path)))
"""
    ),
    md(
        """
## 8. Top 3 candidate × 四张图的 comparison 缩略图

下面的图来自 Stage 2 full setting，每个 cell 是对应 case 的 `summary_comparison.png`。横向比较 candidate，纵向比较图像。
"""
    ),
    code(
        r"""
top3 = stage3_all["candidate_id"].tolist()
images = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]

fig, axes = plt.subplots(len(images), len(top3), figsize=(5.2 * len(top3), 3.4 * len(images)))
for i, image_name in enumerate(images):
    for j, candidate_id in enumerate(top3):
        ax = axes[i, j]
        img_path = SWEEP_ROOT / "stage2" / f"{image_name}__{candidate_id}" / "summary_comparison.png"
        if img_path.is_file():
            ax.imshow(mpimg.imread(img_path))
        ax.axis("off")
        if i == 0:
            ax.set_title(candidate_id, fontsize=10)
        if j == 0:
            ax.text(-0.02, 0.5, image_name, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=10)
fig.suptitle("Stage 2 full setting：Top 3 candidates 的 summary comparison", y=0.995)
fig.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 9. 自动生成的中文结论

这一节的数字全部从 CSV 计算得到，不手写固定结果。这样之后如果替换 sweep 数据，结论会跟着更新。
"""
    ),
    code(
        r"""
best_stage2 = stage2_summary.iloc[0]
best_stage3 = stage3_all.iloc[0]
stage3_top = stage3_all.sort_values("mean_relative_wavefront_error")
region_direction = stage3_top["direction"].mode().iloc[0]
region_min = stage3_top["target_wavefront_rms"].min()
region_max = stage3_top["target_wavefront_rms"].max()
best_stage3_seidel = seidel_table.loc[seidel_table["candidate_id"] == best_stage3.candidate_id, coeff_names].iloc[0]
best_stage3_vector = "[" + ", ".join(f"{v:.6f}" for v in best_stage3_seidel.values) + "]"

summary_md = f'''
### 最重要结论

1. **Stage 2 full setting 的最佳 candidate 是 `{best_stage2.candidate_id}`**：
   - 平均 relative wavefront error = **{best_stage2['平均相对WF误差']:.3f}**
   - 平均 object SSIM = **{best_stage2['平均SSIM']:.3f}**

2. **Stage 3 seed 稳定性验证后，最稳单点仍是 `{best_stage3.candidate_id}`**：
   - mean relative wavefront error = **{best_stage3.mean_relative_wavefront_error:.3f}**
   - std relative wavefront error = **{best_stage3.std_relative_wavefront_error:.3f}**
   - mean object SSIM = **{best_stage3.mean_ssim:.3f}**
   - 具体 Seidel vector `[W040, W131, W222, W220, W311, Wd]` = **`{best_stage3_vector}`**

3. **最容易恢复的 Seidel wavefront 区域**：
   - 方向集中在 **`{region_direction}`**
   - RMS 大致在 **{region_min:.2f} - {region_max:.2f} waves**
   - 单点上 **{best_stage3.target_wavefront_rms:.2f} waves RMS** 最稳。

4. **风险判断**：
   - 最佳 Stage 3 mean relative wavefront error 仍然约为 **{best_stage3.mean_relative_wavefront_error:.3f}**。
   - 这意味着 recovered wavefront 与 GT 的误差 RMS 已经接近 GT wavefront RMS，本问题仍有明显 identifiability 难度。
   - object SSIM 明显更好，说明目前模型更容易恢复“看起来可用的 object”，但 Seidel 参数本身还没有稳定准确。
'''
display(Markdown(summary_md))
"""
    ),
    code(
        r"""
recommendation_md = f'''
## 10. 汇报建议

如果这份结果用于 mentor / 组会，可以用下面三句话组织汇报：

1. 我们把 Seidel wavefront 的方向和 RMS 系统 sweep 了一遍，并用 field-weighted wavefront relative error 作为主指标。
2. 在当前 CoCoA-like object + joint Seidel 下，`{region_direction}`、尤其 `{best_stage3.target_wavefront_rms:.2f} waves RMS` 是最容易恢复的区域；对应 Seidel vector 是 `{best_stage3_vector}`，同时 object SSIM 很高（Stage 3 mean SSIM = {best_stage3.mean_ssim:.3f}）。
3. 但最好的 seed stability mean relative error 仍是 {best_stage3.mean_relative_wavefront_error:.3f}，所以这不是“Seidel 已解决”，而是说明当前机制只在某个较温和 signed Seidel 区域相对更稳。
'''
display(Markdown(recommendation_md))
"""
    ),
]


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
