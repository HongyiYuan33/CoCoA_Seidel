from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_NAME = "seidel_recovery_sweep_20260525_full"
REL_OUTPUT = Path("outputs") / "cocoa_like_2d_mechanism" / RUN_NAME
CODEX_ROOT = Path("/Users/hongyimac/Desktop/Neural_RIng_AO_Codex")
DESKTOP_EXPERIMENT_ROOT = Path("/Users/hongyimac/Desktop/CoCoA_like_2D_Seidel_Experiment")

COEFF_NAMES = ["W040", "W131", "W222", "W220", "W311", "Wd"]
IMAGE_ORDER = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]


def find_project_root() -> Path:
    candidates = []
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    candidates.extend([CODEX_ROOT, DESKTOP_EXPERIMENT_ROOT])
    for base in candidates:
        if (base / REL_OUTPUT / "stage1_metrics.csv").is_file():
            return base
    raise FileNotFoundError(f"Could not find sweep output: */{REL_OUTPUT}")


def parse_vector(value) -> list[float]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return [float(v) for v in value]


def vector_text(vec: list[float]) -> str:
    return "[" + ", ".join(f"{v:+.3f}" for v in vec) + "]"


def compact_coeff_text(prefix: str, vec: list[float]) -> str:
    return f"{prefix} {vector_text(vec)}"


def detail_coeff_text(prefix: str, vec: list[float]) -> str:
    left = " ".join(f"{name}={value:+.3f}" for name, value in zip(COEFF_NAMES[:3], vec[:3]))
    right = " ".join(f"{name}={value:+.3f}" for name, value in zip(COEFF_NAMES[3:], vec[3:]))
    return f"{prefix} {left}\n{right}"


def load_data(sweep_root: Path) -> tuple[pd.DataFrame, list[str]]:
    stage2 = pd.read_csv(sweep_root / "stage2_metrics.csv")
    stage3_stability = pd.read_csv(sweep_root / "stage3_seed_stability.csv")

    top3 = (
        stage3_stability[stage3_stability["image"] == "ALL"]
        .sort_values(["mean_relative_wavefront_error", "max_relative_wavefront_error"])
        ["candidate_id"]
        .tolist()
    )
    if len(top3) != 3:
        raise ValueError(f"Expected exactly 3 top candidates, got {len(top3)}: {top3}")

    config = json.loads((sweep_root / "sweep_config.json").read_text())
    gt_by_candidate = {
        item["candidate_id"]: [float(v) for v in item["seidel"]]
        for item in config["candidates"]
    }

    stage2 = stage2.copy()
    stage2["seidel_gt_vec"] = stage2["candidate_id"].map(gt_by_candidate)
    stage2["seidel_final_vec"] = stage2["seidel_final"].map(parse_vector)

    missing_gt = stage2[stage2["seidel_gt_vec"].isna()]["candidate_id"].unique()
    if len(missing_gt):
        raise ValueError(f"Missing GT Seidel vectors for: {missing_gt}")

    return stage2, top3


def row_for(stage2: pd.DataFrame, image_name: str, candidate_id: str) -> pd.Series:
    matched = stage2[(stage2["image"] == image_name) & (stage2["candidate_id"] == candidate_id)]
    if len(matched) != 1:
        raise ValueError(f"Expected one row for {image_name} / {candidate_id}, got {len(matched)}")
    return matched.iloc[0]


def image_path(sweep_root: Path, image_name: str, candidate_id: str) -> Path:
    path = sweep_root / "stage2" / f"{image_name}__{candidate_id}" / "summary_comparison.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def add_panel(
    ax,
    sweep_root: Path,
    row: pd.Series,
    *,
    title_prefix: str,
    detailed: bool,
    fontsize: float,
) -> None:
    ax.imshow(mpimg.imread(image_path(sweep_root, row.image, row.candidate_id)))
    ax.axis("off")

    gt = row.seidel_gt_vec
    recovered = row.seidel_final_vec
    if detailed:
        gt_text = detail_coeff_text("GT:", gt)
        recovered_text = detail_coeff_text("Recovered:", recovered)
    else:
        gt_text = compact_coeff_text("GT:", gt)
        recovered_text = compact_coeff_text("Recovered:", recovered)

    title = (
        f"{title_prefix}\n"
        f"WF rel={row.relative_wavefront_error:.3f}, coeff L2 rel={row.seidel_l2_relative:.3f}\n"
        f"{gt_text}\n"
        f"{recovered_text}"
    )
    ax.set_title(title, fontsize=fontsize, pad=8)


def save_full_overview(sweep_root: Path, stage2: pd.DataFrame, top3: list[str], out_dir: Path) -> Path:
    fig, axes = plt.subplots(
        len(IMAGE_ORDER),
        len(top3),
        figsize=(7.6 * len(top3), 3.0 * len(IMAGE_ORDER)),
    )
    for i, image_name in enumerate(IMAGE_ORDER):
        for j, candidate_id in enumerate(top3):
            row = row_for(stage2, image_name, candidate_id)
            add_panel(
                axes[i, j],
                sweep_root,
                row,
                title_prefix=f"{image_name} | {candidate_id}",
                detailed=False,
                fontsize=6.8,
            )

    fig.suptitle(
        "Stage 2 full setting - Top 3 candidates with Seidel GT and recovered coefficients",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.982))
    out_path = out_dir / "01_full_top3_summary_with_seidel.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_image_strips(sweep_root: Path, stage2: pd.DataFrame, top3: list[str], out_dir: Path) -> list[Path]:
    saved = []
    for index, image_name in enumerate(IMAGE_ORDER, start=2):
        fig, axes = plt.subplots(1, len(top3), figsize=(7.6 * len(top3), 3.05))
        for ax, candidate_id in zip(np.ravel(axes), top3):
            row = row_for(stage2, image_name, candidate_id)
            add_panel(
                ax,
                sweep_root,
                row,
                title_prefix=candidate_id,
                detailed=True,
                fontsize=8.2,
            )
        fig.suptitle(f"Stage 2 full setting - {image_name}", fontsize=14, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = out_dir / f"{index:02d}_by_image_{image_name}_with_seidel.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
    return saved


def save_candidate_strips(sweep_root: Path, stage2: pd.DataFrame, top3: list[str], out_dir: Path) -> list[Path]:
    saved = []
    first_index = 2 + len(IMAGE_ORDER)
    for offset, candidate_id in enumerate(top3):
        fig, axes = plt.subplots(len(IMAGE_ORDER), 1, figsize=(8.4, 3.05 * len(IMAGE_ORDER)))
        for ax, image_name in zip(np.ravel(axes), IMAGE_ORDER):
            row = row_for(stage2, image_name, candidate_id)
            add_panel(
                ax,
                sweep_root,
                row,
                title_prefix=image_name,
                detailed=True,
                fontsize=8.2,
            )
        fig.suptitle(f"Stage 2 full setting - {candidate_id}", fontsize=14, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        safe_candidate = candidate_id.replace("/", "_")
        out_path = out_dir / f"{first_index + offset:02d}_by_candidate_{safe_candidate}_with_seidel.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
    return saved


def main() -> None:
    project_root = find_project_root()
    sweep_root = project_root / REL_OUTPUT
    out_dir = sweep_root / "stage2_top3_seidel_exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    stage2, top3 = load_data(sweep_root)
    saved = [save_full_overview(sweep_root, stage2, top3, out_dir)]
    saved.extend(save_image_strips(sweep_root, stage2, top3, out_dir))
    saved.extend(save_candidate_strips(sweep_root, stage2, top3, out_dir))

    print(f"Wrote {len(saved)} images to {out_dir}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
