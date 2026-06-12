# 项目审计报告 — CoCoA_like_2D_Seidel_Experiment

**日期**:2026-06-09
**审计对象**:git HEAD `06e5b04`(Add aligned coefficient abs ranking to pretrain contrast stats),工作区干净
**方法**:纯静态审计(Read / grep / `python -m py_compile` / git 只读命令),遵守 AGENTS.md「不在本地运行项目 python/pytest/notebook」的规定。未执行任何训练、测试或数值计算;无法本地验证的数值结论标注「待远端验证」并附命令。
**范围**:`hybrid_ring_cocoa/`(约 7,200 行)、`scripts/`(25 个脚本约 15,400 行)、`tests/`(7 个文件)、`notebooks/` 生成器、git 仓库卫生、文档一致性。重点深查:Seidel 系数顺序跨文件一致性、本次提交的 pretrain-contrast 流水线(5 个脚本)、AGENTS.md 自定运行策略的合规性。范围排除见附录 C。
**证据规则**:每条发现的 `file:line` 均在本次审计中实际读取/grep 核实;高严重度发现由主审计方独立复读代码二次确认。

> 备注:审计启动时这 5 个流水线脚本尚未提交,审计期间被提交为 `06e5b04`(+919/−4)。本报告按提交后的状态撰写。

---

## 1. 总体评价与记分卡

**一句话总评**:这是一个工程纪律明显高于平均水平的研究代码库——核心物理约定经四层交叉验证零分歧、种子全记录、三大 sweep 合规实现了断点续跑;本次发现的唯一高危问题不在核心算法,而在新增评测流水线的 `--resume` 行号拼接逻辑上,**应在下一次对 full150 manifest 跑 report 之前修复**。

| 维度 | 评级 | 高 | 中 | 低 | 建议 | 摘要 |
|---|---|---|---|---|---|---|
| A 正确性风险 | **需改进** | 1 | 0 | 1 | 0 | 核心算法零问题;高危在 evaluator resume 拼接 |
| B 可复现性 | 合格 | 0 | 1 | 2 | 2 | requirements 缺包;种子覆盖与参数落盘本身做得很好 |
| C 健壮性与策略合规 | 合格 | 0 | 1 | 2 | 0 | 三大 sweep 合规;evaluator CSV 写入非原子 |
| D 代码质量与可维护性 | 合格 | 0 | 0 | 4 | 0 | helper 大面积复制,均为漂移风险而非现行错误 |
| E 仓库卫生 | **良好** | 0 | 0 | 0 | 0 | 无发现 |
| F 文档一致性 | 合格 | 0 | 1 | 3 | 0 | 两处 docstring/注释把人引向错误方向 |
| **合计** | | **1** | **3** | **12** | **2** | |

语法门:`hybrid_ring_cocoa/ + scripts/ + tests/ + notebooks/` 全部 `.py` 通过 `py_compile`,无语法错误。

---

## 2. 发现详情

### A. 正确性风险

#### A-1【高】evaluator `--resume` 按行号拼接,分批补 case 后重跑 report 会静默丢行 + 重复行

- **位置**:`scripts/evaluate_seidel_physical_operator_sweep.py:276-289`(resume 用 `_source_row_index` 行号集合判断"已评估")、`:317`(写入行号)、`:331`(新旧行合并重写);触发方 `scripts/report_pretrain_contrast_case_manifest.py:122`(每次按 `(image, candidate_id)` 重排序重写 `stage1_metrics.csv`)、`:154`(固定传 `--resume`)。
- **问题**:resume 的"已完成"判定是输入 CSV 的**位置行号**,而 report 每次运行都会把 `stage1_metrics.csv` 按内容重新排序重写。本流水线的设计用途恰是同一 run prefix 分两批跑:top10plusbase 先跑 36 case(top10 按 image 独立选取,`make_pretrain_contrast_top10plusbase_manifest.py:113-114`,所以一个 method 往往只覆盖部分 image),rms040_remaining 再补 117 case(共用同一 prefix,`make_pretrain_contrast_rms040_remaining_manifest.py:196-210`;`run_pretrain_contrast_case_manifest.py:16` 与 `make_pretrain_contrast_top10plusbase_manifest.py:20` 的 `DEFAULT_PREFIX` 为同一字符串)。第二批补完后对 full150 重跑 report 时:某 method 新增的 `Iksung_beads` 行按字典序(`I` < `d`)排到最前,行号整体后移——旧输出里的 `completed={0}` 会**跳过从未评估过的新行 0**(Iksung_beads)、**重评已评估的行 1**(dendrites)。
- **影响**:最终 `seidel_physical_operator_metrics.csv` 缺失 Iksung_beads 行、dendrites 行重复出现两次(两行 `_source_row_index` 互相矛盾)。下游 `build_pretrain_contrast_rcp_stats.py` 的 `load_rows` 不会报错——缺行静默消失,重复行被 `grouped_summary` 双倍计入均值,`comparison_by_case`、winner counts、README 结论全部被污染,**全程零报错**。
- **最小修复**:resume 判重改用内容键 `(image, direction, candidate_id)` 而非 `_source_row_index`;或 report 在调 evaluator 前删除已存在的 `stage1_operator_eval_dim256/` 输出目录(放弃 resume,全量重评,150 case 的 post-hoc 评测成本可接受)。
- **待远端验证(是否已实际发生)**:本地 outputs/ 中无该流水线产物(产物在远端)。在远端项目根目录运行:

  ```bash
  for f in outputs/cocoa_like_2d_mechanism/pretrain_contrast_top10plusbase4d_*__*/stage1_operator_eval_dim256/seidel_physical_operator_metrics.csv; do
    python3 - "$f" <<'EOF'
  import csv, sys, collections
  rows = list(csv.DictReader(open(sys.argv[1])))
  keys = [(r.get("image"), r.get("candidate_id")) for r in rows]
  dup = [k for k, c in collections.Counter(keys).items() if c > 1]
  idx = [r.get("_source_row_index") for r in rows]
  dup_idx = [k for k, c in collections.Counter(idx).items() if c > 1]
  if dup or dup_idx:
      print("CORRUPTED:", sys.argv[1], "dup_case=", dup, "dup_idx=", dup_idx)
  EOF
  done
  ```

  同时核对每个 method 的输出行数 == 其 `stage1_metrics.csv` 行数。若已有重复/缺失,受影响 method 的 eval 目录删除重评即可(stage1 训练结果本身不受影响)。

#### A-2【低】`coerce_seidel_vector` 会把 GT 的 fixed 位也静默清零,潜在抹掉 misspecification

- **位置**:`hybrid_ring_cocoa/evaluation/seidel_operator_evaluator.py:184-185`(6 维输入也强制 `out[idx] = 0.0`)、`:979`(GT 同样经过该清零);消费方 `scripts/evaluate_seidel_physical_operator_sweep.py:292-301`。
- **问题**:`fixed_indices` 的语义是"固定为 0"。若未来把「GT 含非零 W311/Wd、模型 fixed=[4,5]」的 misspecified 数据喂给 post-hoc evaluator,GT 的 W311 会被静默清零,`operator_error_*` 比较的是清零后的 GT,**系统性低估 misspecified 模型的误差**。
- **当前无害(已逐一验证)**:现有全部数据流的 GT fixed 位在源头就是 0——direction 候选生成清零(`run_cocoa_like_seidel_accuracy_sweep.py:200-202`)、gt_locked_front4 强制 classical5d/6d(`:232-233`)、trace misspecification 走 trace evaluator、symmetry ablation 的 classical 分支显式传 `fixed_indices=[]`。
- **最小修复(防御)**:6 维输入的 fixed 位 |值| > atol 时 warn 或 raise——`run_cocoa_like_2d_mechanism.py:133-137` 的 `coerce_classical_backend` 已经是这种严格语义,两处行为应统一。
- **待远端验证**:构造 `theta_gt=[0.2,0.1,0.05,0.05,0.08,0]`、`fixed_indices=[4,5]` 调 `evaluate_seidel_recovery`,确认其与 `fixed_indices=[]` 的 operator error 不同且无任何告警。

### B. 可复现性

#### B-1【中】requirements.txt 缺 4 个实际使用的依赖

- **位置**:`requirements.txt`(仅 matplotlib/numpy/scikit-image/torch);缺失项的实际 import:
  - `scipy` — `hybrid_ring_cocoa/preprocessing.py:22-24`、`hybrid_ring_cocoa/_rdm/_src/util.py:12-13`(**核心包必需**)
  - `tqdm` — `hybrid_ring_cocoa/_rdm/blur.py:8`、`_rdm/_src/psf_model.py:13`(**核心包必需**,PSF 生成路径)
  - `Pillow` — `scripts/build_*.py` 共 8 处(如 `build_pretrain_contrast_rcp_stats.py:22`)
  - `pytest` — `tests/` 5 个文件
- **影响**:新环境 `pip install -r requirements.txt` 后,任何涉及 preprocessing 或 PSF 生成的代码在 import 即失败。日常在远端固定 conda env(`hybrid_ring`)运行缓解了实际影响(AGENTS.md「Use the existing remote Python environment first」),故定级中而非高。
- **最小修复**:requirements.txt 追加 `scipy`、`tqdm`、`Pillow`、`pytest` 四行(pytest 也可拆到 requirements-dev.txt)。pandas 仅 notebooks 报告生成器使用,可选声明。

#### B-2【低】stats builder 的 `read_csv` 对缺失文件静默返回空列表

- **位置**:`scripts/build_pretrain_contrast_rcp_stats.py:206-210`(`if not path.is_file(): return []`),消费方 `load_rows`(:328-344)。
- **问题**:13 份 `read_csv` 副本中唯独此版本缺文件不报错。某 method 的 eval CSV 缺失时该 method 全部行静默消失,只有**全部**为空才触发 `main` 的 SystemExit(:1098-1099)。在 report 流水线内安全(evaluator subprocess `check=True` 先行保证),但手动/部分运行 builder 时结论会无声缺 method。
- **最小修复**:缺文件时打印 warning(列出缺失路径),或维持 FileNotFoundError 与其他副本一致。

#### B-3【低】manifest 溯源列依赖"rms020 stats 已用新版 builder 重跑过",否则静默置空

- **位置**:`scripts/make_pretrain_contrast_rms040_remaining_manifest.py:87-90`(`row.get(..., "")`);该列 `aligned_coeff_absolute_error_physical` 是 `06e5b04` 才加入 `comparison_by_case.csv` 的(`build_pretrain_contrast_rcp_stats.py:458`)。
- **影响**:若 rms020 的 stats 未用新代码重新生成,manifest 的 `source_*` 溯源列全为空且无告警。不影响 run/report 执行(下游不消费),属数据血缘退化。
- **最小修复**:validate 中断言该列至少一行非空,或缺失时打 warning。

#### B-4【建议】accuracy sweep 的 stage1/stage2 种子硬编码为 0,CLI 不可配

- **位置**:`scripts/run_cocoa_like_seidel_accuracy_sweep.py:1269`、`:1292`(`seed=0`);stage3 有 `--stage3-seeds`(默认 [1,2])。
- **说明**:固定种子是确定性的且被完整记录(sweep_config.json + 各 case metrics.json 的 config 字段),可复现性本身无损;限制只在"不改代码无法换种子重筛"。建议补 `--stage1-seed/--stage2-seed` 参数或在注释中说明设计意图。

#### B-5【建议】symmetry ablation 脚本无 `torch.manual_seed`

- **位置**:`scripts/run_seidel_symmetry_ablation_sweep.py:493`(仅 `np.random.default_rng(7127 + case.seed)`)。
- **说明**:该脚本当前不含 torch 随机操作,技术上不需要;但与其余入口的「torch+numpy 双 seed」模式不一致,未来引入 torch 随机路径时容易被遗漏。建议补一行保持一致。

### C. 健壮性与运行策略合规(对照 AGENTS.md「Checkpoint Expectations」)

#### C-1【中】evaluator 输出 CSV 非原子写,且 resume 模式逐行全量重写

- **位置**:`scripts/evaluate_seidel_physical_operator_sweep.py:102`(`path.open("w")` 直写,无 temp+rename)、`:329`(resume 模式每评完一行就全量重写一次)。
- **问题**:这是被 report 流水线 subprocess 高频调用、写入最频繁的关键 CSV,却是非原子直写——违反 AGENTS.md「Checkpoint writes should be atomic」。AGENTS.md 自己就记录了共享服务器上进程被其他用户手动 kill 是常态;写中途被杀 → CSV 截断;下次 resume 读到残缺尾行(若恰好丢了 `_source_row_index` 字段)会既保留残行又重评该 case,产生重复行,并放大 A-1 的错位。
- **最小修复**:改 temp+`os.replace`(仓库内已有现成实现可抄:`run_seidel_blind_recovery_sweep.py:351` 的 `write_csv_atomic`);resume 加载时丢弃字段不完整的尾行。

#### C-2【低】7 个脚本的 argparse 默认值为 cwd 相对路径,未重锚定到 PROJECT_ROOT

- **位置**(均已逐个确认 main() 内无 `PROJECT_ROOT /`、`is_absolute()`、`resolve()` 重锚定):
  `build_capacity4d_rcp_stats.py:849,854`、`build_seidel_optimizer_rcp_comparison.py:340`、`build_seidel5d6d_gtlocked_rcp_stats.py:451`、`build_rms_floor_operator_eval_input.py:110`(该文件连 PROJECT_ROOT 都未定义)、`build_single_coeff_rcp_stats.py:622`、`build_size512_rcp_four_images.py:390,395`、`make_pretrain_contrast_sweep_manifest.py:14`。
- **影响**:从项目根目录以外运行(如 `cd scripts && python ...`)会把输出写错位置或找不到输入。项目惯例从根目录跑,故为低;但缺乏防御。
- **最小修复**:main() 开头统一加 `if not args.output_root.is_absolute(): args.output_root = PROJECT_ROOT / args.output_root`。

#### C-3【低】`run_cocoa_like_2d_mechanism.py` 独立运行时无断点续跑,关键 JSON 非原子写

- **位置**:`main()`(:1473-1556)无完成标记检查;`:1325`(metrics.json)与 `:1554`(summary.json)均 `write_text` 直写。
- **说明**:该脚本设计为单次运行(per run_name);作为 sweep 子单元被调用时,sweep 层(accuracy sweep)有完备的 case 级 resume 与原子写。独立长跑被中断需从头再来。按 AGENTS.md「expensive single-case training prefer iteration-level checkpoints」属软性不合规。
- **最小修复**:文档声明"独立运行不支持 resume";若有独立长跑需求再加 iteration checkpoint。

**合规矩阵(正面结论为主)**:

| 脚本 | 跳过已完成 case | 完成标记 | 原子写 | --force 默认关 |
|---|---|---|---|---|
| run_cocoa_like_seidel_accuracy_sweep.py | ✓ :1118-1125 | ✓ `sweep_case_complete`(:581),与进行中状态分离 | ✓ :348-359 | ✓ :1388 |
| run_seidel_blind_recovery_sweep.py | ✓ :823-829 | ✓ metrics.json | ✓ :351 | ✓ :896 |
| run_seidel_symmetry_ablation_sweep.py | ✓ :840-846 | ✓ metrics.json | ✓ :663 | (无 force,可安全重跑) |
| run_pretrain_contrast_case_manifest.py | 委托子脚本 ✓ | 委托 | 委托 | 无 |
| run_cocoa_like_2d_mechanism.py(独立) | ✗(见 C-3) | ✗ | ✗ | 无 |

11 个 `scripts/*.sh` 启动器均无 `--force`,合规。

### D. 代码质量与可维护性

#### D-1【低】CSV/settings helper 大面积复制,且 `load_settings` 已同名不同义

- **位置**:`def write_csv` 16 份 + `write_csv_atomic` 2 份(列序三种:首现序 / sorted / preferred;原子性两种);`def read_csv` 13 份(行为分叉见 B-2);`def method_id` 4 份(`make_pretrain_contrast_rms040_remaining_manifest.py:48`、`make_pretrain_contrast_top10plusbase_manifest.py:53`、`report_pretrain_contrast_case_manifest.py:55`、`run_pretrain_contrast_case_manifest.py:29`——**经逐字节比对当前完全一致**,这是流水线 join key,暂无错位风险);`load_settings` 3 份**同名不同义**(`run_..._case_manifest.py:33` 返回 dict / `report_...py:59` 返回 list 并注入 `method` 键 / `make_..._top10plusbase...py:74` 返回 dict 并**额外注入 scalar5 合成 setting**),另有同语义的 `read_settings`(`make_..._rms040_...py:52`)。
- **说明**:`extrasaction="ignore"` 的有无(6 份带、其余不带)经核实**不构成现行行为分歧**——所有不带 ignore 的版本 fieldnames 都取全行键并集,多余键结构上不可能出现。真实分叉在:列序、原子性、缺文件行为、`load_settings` 的注入逻辑。
- **影响**:当前无错;但任何一处"顺手改进"或跨脚本拷贝都可能静默改变兄弟脚本依赖的行为。
- **最小修复**:抽一个 `scripts/_contrast_pipeline_lib.py`(或 `scripts/_lib.py`)收编 `read_csv / write_csv_atomic / method_id / load_settings`,新脚本一律 import。

#### D-2【低】`candidate_id` token 存在两套生成逻辑,是潜在的 join key 分叉点

- **位置**:manifest 端 `str(target_rms).replace(".", "p")`(make_* 两脚本,如 `make_..._rms040_...py:73`)vs runner 端 `tag_float` 即 `f"{v:.3f}".rstrip(...)`(`run_cocoa_like_seidel_accuracy_sweep.py:77,209`)。
- **问题**:`candidate_id` 是 manifest → runner 输出目录 → report 完成标记轮询的跨进程 join key。默认 0.40/0.20 下两套逻辑同为 "0p4"/"0p2";但 rms 小数超过 3 位(如 0.0625 → manifest "0p0625" vs runner "0p062")即分叉,report 将**永远等不到完成标记**(死循环轮询,fail-silent)。
- **最小修复**:manifest 端改用与 runner 相同的 `tag_float`(并入 D-1 的共享模块)。

#### D-3【低】top10plusbase 的"baseline 已进 top10"分支与计数断言自相矛盾

- **位置**:`scripts/make_pretrain_contrast_top10plusbase_manifest.py:116-117`(`if extra not in existing` 明确处理 baseline 已在 top10 的情形)vs `:131-132`(紧接着断言 `len(chosen) == top_k + 2`)。
- **影响**:一旦 baseline_scalar1 真的进入某 image 的 top10(rms020 的 comparison_by_case.csv 对 baseline 同样输出行,完全可能),chosen 只有 11 个 → RuntimeError,脚本对合法输入拒绝运行。fail-loud,不产生错数据。
- **最小修复**:断言放宽为 `top_k <= len(chosen) <= top_k + 2`(validate 同步),或选 top10 时先排除 baseline/scalar5。

#### D-4【低】rms040 manifest 的 validate 硬编码期望值,CLI 参数化形同虚设

- **位置**:`scripts/make_pretrain_contrast_rms040_remaining_manifest.py:109`(settings==50)、`:114`(per-image 50/39)、`:124`(`candidate_id != "signed_balanced__rms0p4"` 写死)。
- **影响**:脚本提供 `--target-rms/--expected-full/--expected-remaining`,但 validate 只对默认参数成立;改参数即使数据正确也必然抛错。fail-loud。
- **最小修复**:per-image 期望改为 `expected_full // len(IMAGES)`,candidate_id 比对复用 :73 的 token 生成式。

### E. 仓库卫生

**无发现。** 135 个 tracked 文件;最大 tracked 二进制为 4 张参考 PNG 共约 5MB(合理的引用数据);`outputs/`(9.9GB)、`*.pt`、`__pycache__` 等均被 .gitignore 正确排除;`experiments/` 39 个 symlink 无一悬空;工作区干净。

### F. 文档一致性

#### F-1【中】`load_baboon_gt` 默认路径指向不存在的文件,docstring 声称"随包发布",golden 的 natural_image probe 因此永久静默跳过

- **位置**:`hybrid_ring_cocoa/training/data.py:29`(默认 `hybrid_ring_cocoa/data/baboon.png`,**该文件不在仓库中**)、`:68-69`(FileNotFoundError)、`:3-4` 与 `:50-51`(docstring 称 shipped as package data);`hybrid_ring_cocoa/__init__.py:31,54` 顶层导出。
- **影响面(已 grep 全仓库)**:6 处调用都显式传了 `path=`,不受影响;唯一用默认参数的是 `scripts/run_frozen_rdm_forward_golden.py:193`,被 try/except 包裹——后果是 golden 一致性检查的 `natural_image` probe 组**在任何机器上都永久静默缺席**(skip 原因只进输出 JSON,不打告警),golden 覆盖面无声缩水,且公开 API 的默认调用 100% 抛异常、docstring 失实。
- **最小修复**:默认路径改为仓库内实际存在的 `hybrid_ring_cocoa/data/sharpe_simulation_figure_package/Test_figure_1.png`,或将 `path` 设为必填并修正 docstring;golden 脚本 skip 时向 stdout 打一行警告。

#### F-2【低】`single_mode_control` docstring 把 defocus 写成 `num = 0`,实际是 index 5

- **位置**:`hybrid_ring_cocoa/training/losses.py:106-109`:"anchors that coefficient at zero — our use case for the defocus term (`num = 0` in the Seidel convention, see `hybrid_ring_cocoa/optics/seidel_psf.py`)"。本仓库约定 defocus 在 index **5**;num=0 是 W040 球差。它引用的 seidel_psf.py 正是声明 defocus-last 的文件。
- **影响**:全部现有调用正确(`train.py:103,246` 默认 `defocus_index=5`;mechanism 脚本 :843、:1443 同),纯文档错误——但这是新人最可能照抄的 docstring;照抄会把 anchor 打到球差上,训练不报错、只是悄悄变差,极难排查。
- **最小修复**:`num = 0` → `num = 5`。

#### F-3【低】vendored `psf_model.py:21` 的顺序注释与其自身代码相反

- **位置**:`hybrid_ring_cocoa/_rdm/_src/psf_model.py:21`:`# Note coeffs expected in following order: Wd, W040, W131, W222, W220, W311`(defocus-first)——与同文件 `:61-68` 实际代码(defocus-last,见 §3 映射表)相反,也与同包 `_rdm/calibrate.py:42` docstring(正确)矛盾。
- **影响**:任何照注释扩展代码的人会把 6 个系数全部错位一格。`optics/seidel_psf.py:5-14` 已显式记录并警告此事(好),但雷本身还在。
- **最小修复**:改正该行注释,标注 `# [patched: order verified against code]`。

#### F-4【低】README 的 classical4d/5d「模型」仅存在于脚本层,库级 `train()` 无对应参数

- **位置**:README.md:45-49 vs `hybrid_ring_cocoa/training/train.py:90-285`(无 `fixed_seidel_indices` 参数,仅 :246 的 defocus 软 anchor);硬冻结实现只在 `scripts/run_cocoa_like_2d_mechanism.py:707,720-729`。
- **影响**:想通过库 API 复现 README 的 classical4d 的用户会发现无入口,需走脚本。属 API/文档差距,非数值错误。
- **最小修复**:README 注明 classical 预设由 `scripts/run_cocoa_like_2d_mechanism.py` 的 `--seidel-convention` 提供。

---

## 3. 核心算法交叉验证结果(本次审计最重要的「无发现」)

Seidel 系数顺序约定 `[W040, W131, W222, W220, W311, Wd]`(defocus-last)经 **5 处独立实现**逐项交叉核对,**零分歧**:

| index | psf_model.py `compute_pupil_phase` :61-68 | seidel_psf.py 批量路径 :546-553 | evaluator `_seidel_wavefront` :277-284 | mechanism 脚本 :557-585 | accuracy sweep :131-138 | 物理项 |
|---|---|---|---|---|---|---|
| 0 | ρ⁴ | ρ⁴ | ρ⁴ | ρ⁴ | ρ⁴ | W040 球差 |
| 1 | h·ρ³cosθ | 同 | 同 | 同 | 同 | W131 彗差 |
| 2 | h²ρ²cos²θ | 同 | 同 | 同 | 同 | W222 像散 |
| 3 | h²ρ² | 同 | 同 | 同 | 同 | W220 场曲 |
| 4 | h³ρcosθ | 同 | 同 | 同 | 同 | W311 畸变 |
| 5 | ρ² | 同 | 同 | 同 | 同 | Wd 离焦 |

配套核验同样通过:mirror/twin 符号表(evaluator:41-46)与各项 cosθ 奇偶性逐项自洽,且与 :373-381 的直接波前定义闭环;trace 参数化(`W222=2A`、`W220=F−A`)与符号表可交换;`fixed_seidel_indices` 语义(classical4d→[4,5]=W311+Wd)与 README:47-48 逐字一致,冻结实现三重保险(初始化清零 + 前向 mask 切梯度 + step 后硬清零,mechanism:726-729,:822,:876-878);6 个 GT preset 与 sweep 方向向量的命名↔索引对应全部正确;device 流转零漏传(`_rdm` 的 CPU 默认被封装层完全屏蔽,所有出口锚定输入张量的 device)。

---

## 4. 做得好的地方

1. **约定漂移有运行时防线**:`validate_hardcoded_transform_wavefronts`(evaluator:350-388)用直接波前定义验证全部硬编码符号表,且 `evaluate_seidel_physical_operator_sweep.py:223` 每次运行都执行。这类「约定静默漂移」是这类项目最危险的故障模式,这里有 guard。
2. **`optics/seidel_psf.py:5-14` 的顺序文档堪称教科书**:主动记录 vendored 注释的错误并给出三处权威代码依据,本次审计逐一复核全部属实。
3. **种子纪律**:全部训练入口逐 case `torch.manual_seed + np.random.seed`,且种子随完整 config 落盘(`vars(args)` → sweep_config.json / 各 case metrics.json),任何历史结果都可追溯参数。本次提交的 manifest runner 的 30+ 个硬编码超参也全部经 CLI 传给子进程并被记录——固定但可追溯,设计正确。
4. **三大 sweep 的断点续跑合规**:case 级完成标记(`sweep_case_complete`)与进行中状态分离、temp+rename 原子写、`--force` 默认关(见 §2-C 矩阵);report 的轮询对半写 JSON 有双重容错。
5. **流水线契约总体扎实**:producer→consumer 9 段列契约逐段核对全部通过;30+ 个 subprocess CLI 参数与三个被调脚本的 argparse 全部对得上;`method_id` 4 份副本逐字节一致;150/117/36/33 计数体系跨脚本闭环自洽;manifest 必需键用 `row[...]`(KeyError fail-loud)而非 `.get` 静默空值。
6. **`06e5b04` 的新指标实现质量好**:`aligned_coeff_absolute_error_physical` 来源核实存在(evaluator:1100/:1316),解析、聚合、best-row 方向(lower-is-better)与既有 aligned_* 指标完全同构,下游全链消费无死代码;baseline 分母改动修正了旧版的错误假设。
7. **仓库卫生与知识管理**:.gitignore 完整、9.9GB outputs 全部隔离、experiments/ symlink 目录化索引、`.learnings/` 记录基础设施教训形成闭环;notebooks 走 build_*.py 生成器模式;`RECENT_SEIDEL_EVALUATOR_ARTIFACTS.md` 引用的产物路径逐一存在。

---

## 5. 修复优先级清单

### 立即(在对 full150 manifest 跑 report 之前)

| # | 修复 | 对应发现 |
|---|---|---|
| 1 | evaluator resume 改内容键 join(或 report 调 evaluator 前删旧 eval 目录全量重评);先用 A-1 的远端命令核查既有输出是否已损 | A-1 |
| 2 | evaluator `write_csv` 改 temp+rename(抄 `run_seidel_blind_recovery_sweep.py:351`) | C-1 |

### 快速修复(每项 < 15 分钟)

| # | 修复 | 对应发现 |
|---|---|---|
| 3 | requirements.txt 追加 scipy / tqdm / Pillow / pytest | B-1 |
| 4 | losses.py docstring `num = 0` → `num = 5` | F-2 |
| 5 | psf_model.py:21 注释顺序改正 | F-3 |
| 6 | baboon 默认路径改 `Test_figure_1.png` 或必填化 + 修 docstring;golden skip 打警告 | F-1 |
| 7 | top10plusbase 断言放宽 | D-3 |
| 8 | builder read_csv 缺文件打 warning;manifest 溯源列空值打 warning | B-2, B-3 |

### 较大改造(规划后做)

| # | 修复 | 对应发现 |
|---|---|---|
| 9 | 抽 `scripts/_lib.py` 收编 read_csv / write_csv_atomic / method_id / load_settings / tag_float,流水线脚本统一 import | D-1, D-2 |
| 10 | 7 处 argparse 默认路径 PROJECT_ROOT 锚定 | C-2 |
| 11 | `coerce_seidel_vector` 对非零 fixed 位 warn/raise,与 `coerce_classical_backend` 语义统一 | A-2 |
| 12 | validate 期望值参数化;`--stage1/2-seed` 暴露;symmetry ablation 补 torch seed | D-4, B-4, B-5 |

---

## 6. 附录

### A. 可复跑的关键审计命令

```bash
# 语法门(唯一允许的本地"执行")
find hybrid_ring_cocoa scripts tests notebooks -name "*.py" -not -path "*__pycache__*" -print0 \
  | xargs -0 python3 -m py_compile

# 依赖缺口
grep -rn "^import scipy\|^from scipy\|^from PIL\|import tqdm" hybrid_ring_cocoa scripts

# helper 重复矩阵
grep -n "def read_csv\|def write_csv\|def method_id\|def load_settings" scripts/*.py

# cwd 相对默认路径
grep -n 'Path("outputs' scripts/*.py

# resume / 原子写 / force
grep -n "sweep_case_complete\|write_json_atomic\|write_csv_atomic\|--force" scripts/run_*.py

# 系数顺序锚点
sed -n '1,30p' hybrid_ring_cocoa/optics/seidel_psf.py
sed -n '15,25p' hybrid_ring_cocoa/_rdm/_src/psf_model.py

# 仓库卫生
git ls-files | wc -l
find experiments -type l ! -exec test -e {} \; -print
```

### B. 严重度定义(研究代码标尺)

- **高**:可能静默污染科学结论或毁掉昂贵算力(系数错位、join 错位、覆盖已完成 case)。
- **中**:必然踩坑或阻断复现(缺依赖、公开 API 必抛异常、关键写入非原子)。
- **低**:维护摩擦/漂移风险,当前未伤害结果。
- **建议**:改进项。明确不计入:无 CI/打包/类型注解、长而有文档的领域函数、vendored 代码风格。

### C. 范围排除及理由

- `_rdm/`(约 3,067 行 vendored rdmpy):只审接口边界(device 默认、系数索引行),不做行级/风格审查——上游快照,分歧归上游。
- 9 个 `build_*_rcp_stats` 变体:只全读本次被修改的 `build_pretrain_contrast_rcp_stats.py`,其余仅横切 grep——同构兄弟,重复性发现已集体覆盖。
- `outputs/` 内容与 `experiments/` symlink 目标内容:只查存在性。
- notebooks 生成器:只验证其引用的路径存在(均存在)。
- 评估器全文精读:按约定的"中等深度抽查"执行,只深读了顺序/符号/device 相关函数。
- 未执行任何 pytest / torch import / pip / 网络操作。

### D. 审计执行说明

四个并行审计通道(复现性与环境 / 策略合规与路径 / 核心算法一致性 / 脚本重复与流水线契约)+ 主审计方独立做语法门、git 卫生、文档一致性与高危发现复读。每条入报发现的 file:line 均经实际读取核实;唯一「高」级发现(A-1)由主审计方独立复读 `evaluate_seidel_physical_operator_sweep.py:276-331` 与 `report_pretrain_contrast_case_manifest.py:110-157` 二次确认成立。
