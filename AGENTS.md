# Workflow Notes for Codex

## Shared GPU Server Policy

- Default GPU server: `waller-caml`, SSH target `hongyi_waller@waller-caml.eecs.berkeley.edu`.
- Treat `waller-caml` as the default server for this workspace unless the user explicitly chooses another server.
- Do not use `waller-fourier` or the `waller` SSH alias unless the user explicitly asks to use that server. The `waller` alias points to `hongyi@waller-fourier.eecs.berkeley.edu`.
- `waller-fourier` can still be used, but it should be yielded to graduate students when needed. Long-running work on `waller-fourier` must have checkpoint/resume protection before launch.
- Before launching any long GPU job on any shared server, check current GPU and process usage with `nvidia-smi` and an appropriate `ps`/job-status command.
- Prefer running long experiments in a resilient session such as `tmux`, and make sure outputs/checkpoints are written to persistent data storage rather than a nearly-full root partition.
- If a job disappears and the logs do not show normal completion, a Python traceback, CUDA/OOM failure, or server reboot, treat it as a likely manual kill by another user who needs the server.
- When a likely manual kill is detected, stop the current work, report the last completed checkpoint/case and expected lost progress to the user, and do not restart or resume the job until the user explicitly says to continue.

## GPU Index and Host Safety

- When the user says `GPU0`, `GPU1`, or similar without naming a server, interpret that as the physical `nvidia-smi` GPU index on `waller-caml`, not on `waller-fourier`.
- Default remote commands must use `ssh waller-caml` or `ssh hongyi_waller@waller-caml.eecs.berkeley.edu`, never `ssh waller`, unless the user explicitly asks for `waller-fourier`.
- Before launching a long remote GPU job, confirm the target identity with `hostname` and `whoami`. Expected default target is hostname `CIVO-ML-SERVER` and user `hongyi_waller`. If the command lands on `waller-fourier` or user `hongyi`, stop and report instead of running.
- To bind a physical GPU by `nvidia-smi` index, set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<physical_index>`.
- After setting `CUDA_VISIBLE_DEVICES` to one physical GPU, PyTorch should normally use `cuda:0` inside that process. Do not use unmasked `torch.cuda` indices as physical GPU identities.
- For parallel jobs on physical GPU0 and GPU1, start separate sessions/processes with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` and `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`; each process should use PyTorch `cuda:0` internally.
- If GPU mapping is uncertain, verify with `nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv` plus a small masked PyTorch device-name check before launching the long job.

## Project Execution Rule

- Do not run this project's `python`, `pytest`, notebooks, or scripts locally unless the command is a pure syntax-only check such as `python -m py_compile path/to/file.py`.
- Treat README commands that show local `python ...` execution as historical reproduction examples, not as Codex execution instructions.
- For tests, smoke runs, golden regressions, and experiment scripts, run on `waller-caml` by default.
- Use the existing remote Python environment first:
  `/hdd10tb/hongyi_waller/miniconda3/envs/hybrid_ring/bin/python3.10`.
- Do not use remote system Python for this project when PyTorch or project imports are involved.
- Do not install or reinstall PyTorch/dependencies unless the user explicitly asks. If the `hybrid_ring` environment is missing or broken, report that and stop before changing the environment.
- Keep remote project files on the data disk, not the root/home partition. Preferred remote workspace:
  `/hdd10tb/hongyi_waller/projects/CoCoA_like_2D_Seidel_Experiment`.
- Do not place conda envs, package caches, cloned projects, test outputs, or golden artifacts under `/` or a nearly-full home/root filesystem.
- Before syncing, check whether the preferred remote workspace already exists. Sync only the needed source/test/script files and avoid overwriting remote outputs unless the user explicitly asks.
- For source sync, exclude heavy/generated directories such as `outputs/`, `.pytest_cache/`, `__pycache__/`, and notebook execution artifacts unless they are required for the requested task.
- Before remote execution, check `nvidia-smi` and choose a free GPU. Prefer GPU 0 or 1 when free, but always rely on the live `nvidia-smi` result rather than a remembered state.
- Run remote commands with explicit `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=<physical_gpu>`, and `PYTHONPATH=.` from the remote project root.
- For long golden/regression runs, write JSON/CSV summaries to the remote data disk and pull back only the requested summaries/artifacts, not large tensors unless needed.

## Checkpoint Expectations

- For long sweeps, require at least case-level resume: completed cases must leave durable completion markers such as `metrics.json`, `summary.json`, or equivalent files that the script skips on rerun.
- For expensive single-case training, prefer iteration-level checkpoints that include model/tensor state, optimizer state, current phase/iteration, and RNG state.
- Do not treat a partial checkpoint as a completion marker. A completed-case marker should be separate from an in-progress checkpoint.
- Checkpoint writes should be atomic when possible: write to a temporary file first, then rename/replace the latest checkpoint.
- When restarting, avoid `--force` or equivalent overwrite flags unless the user explicitly asks to discard previous progress.

## Visualization Shorthands

- `Ranked CoeffSim Panel` (aliases: `RCP`, `CoeffSim Panel`, `four_panel_plus_coeff_similarity`) is a standard visual for this project. It means the combined PNG layout with an annotated ranked reconstruction panel on the left and a coefficient physical-similarity card on the right.
- The left side shows the ranked image comparison/header metrics, typically GT, measurement, joint reconstruction, predicted measurement, and gain absolute error. The right side shows the size512 physical-similarity card with operator metrics, field-weighted Seidel wavefront RMS in waves, and a GT/raw/aligned recovered Seidel coefficient comparison.
- New RCPs should report `GT RMS`, `rec_aligned RMS`, and preferably `rec_raw RMS`, computed with `field_weighted_wavefront_rms` from `seidel_gt`, `aligned_seidel_physical`, and raw `seidel_final`. Use the aligned recovered RMS as the primary recovery Seidel RMS because it matches the orange coefficient bars on the card.
- Canonical local directory:
  `outputs/cocoa_like_2d_mechanism/tuned_prior_ranked_by_image_dim_rms_calibrated_with_coeff_similarity`.
- Canonical example:
  `outputs/cocoa_like_2d_mechanism/tuned_prior_ranked_by_image_dim_rms_calibrated_with_coeff_similarity/Iksung_beads/5D/rms0p20/rank001__opcal0p1760__opphys0p1760__5D__pos_balanced__four_panel_plus_coeff_similarity.png`.
- Do not confuse this with `seidel_physical_similarity_cards` alone (right-side card only) or `tuned_prior_ranked_by_image_dim_rms_calibrated` (ranked image panel without the coefficient-similarity card).
