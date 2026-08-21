# True Ellipse V7 Gate V2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 105 mm 目标达成与当前严格几何前沿上的模型训练授权解耦，在不改写旧 strict 结论的前提下生成 102.5/95.0 mm 动态 holdout 数据与正式五 seed 模型，并用完整下游证据校准 kappa gate。

**Architecture:** 保留 `kappa P95 <= 150` 作为 legacy strict conditioning gate，新增只负责允许完整证据链继续运行的 downstream admission gate。V7 gate-v2 以版本化字段和任务指纹传播当前 strict frontier、105 mm KPI、tube/data/training admission；阈值 sweep 使用独立 runner 和输出目录，对 150/200/250/300/400 五个候选策略分别形成可审计的几何、tube、support、holdout 与模型结果，且不会覆盖冻结的 V7/V7-D1 产物。

**Tech Stack:** Python 3.11、NumPy、Pandas、SciPy、PyArrow、scikit-learn、Matplotlib、pytest。

---

## 固定语义与非目标

- `target_105_achieved` 是独立 KPI，仅表示已通过相应 policy 的连续严格几何/完整证据前沿达到至少 105 mm。
- `formal_radial_gate_pass`、`formal_tube_gate_pass`、`formal_dataset_gate_pass` 表示当前注册前沿上的相应阶段证据完整，不再暗含 `R >= 105 mm`。
- `conditioning_gate_pass` 和 `centerline_gate_pass` 保持 legacy strict 兼容语义：`sigma3 P05 >= 0.0015 m` 且 `kappa P95 <= 150`。
- 新增 `downstream_admission_conditioning_gate_pass` 与 `downstream_admission_gate_pass`。它们要求 residual、smoothness、seam、joint-domain/margin 与基本 `sigma3` 质量，但不自动授予 strict geometry/model claim。
- kappa 候选阈值用于回放/划分候选 policy 的 strict frontier；没有完整八 job、cut invariance、repeatability、tube、support、五 seed model 证据时不得注册为正式阈值。
- 冻结目录 `runs/true_ellipse_standard_domain_v7` 和 `runs/true_ellipse_standard_domain_training_v7_diagnostic_centerline` 不原地覆盖。

### Task 1: 固化基线、策略版本和 gate 分层契约

**Files:**

- Create: `docs/plans/2026-07-17-true-ellipse-v7-gate-v2.md`
- Modify: `tests/test_true_ellipse_atlas_utils.py`
- Modify: `scripts/analysis/true_ellipse_atlas_utils.py`
- Modify: `tests/test_true_ellipse_radial_bundle_v6_runner.py`
- Modify: `scripts/analysis/run_true_ellipse_radial_bundle_v6.py`

**Step 1: 写失败测试**

- 断言 legacy strict gate 在 `kappa=151` 时仍失败。
- 断言相同 residual/smoothness/seam、`sigma3` 合格的报告可通过 downstream admission。
- 断言 joint margin 失败时 V6 job admission fail-closed。
- 断言 gate mode/候选阈值进入 radius task fingerprint，避免复用旧缓存。

**Step 2: 运行 RED**

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/quasi-exp-py311-packages LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 -m pytest -q tests/test_true_ellipse_atlas_utils.py tests/test_true_ellipse_radial_bundle_v6_runner.py
```

Expected: 新字段/参数断言失败。

**Step 3: 最小实现**

- 在 atlas gate 评估中同时返回 strict 与 admission 字段。
- V6 radial job 根据版本化 `radial_job_gate_mode` 选择 strict 或 admission，但始终合取 joint margin。
- 结果中同时记录原始 strict 与实际 admission 判定。
- 将 gate mode、admission sigma3 与候选 kappa policy 写入 task fingerprint。

**Step 4: 运行 GREEN**

运行 Step 2 同一命令，预期全部通过。

### Task 2: 解耦 105 mm KPI 与 V7 当前前沿授权

**Files:**

- Modify: `tests/test_true_ellipse_standard_domain_v7.py`
- Modify: `scripts/analysis/run_true_ellipse_standard_domain_v7.py`

**Step 1: 写失败测试**

- 102.5 mm strict radial frontier 应得到 `target_105_achieved=false`，同时 `downstream_radial_admission_gate_pass=true`。
- tube phase 在上述报告下必须开始物化，不能再返回 `formal_radial_gate_failed_below_105mm`。
- 当前 strict radial frontier 的所有 tube radius 合格时，tube gate 可通过且 KPI 仍为 false。
- summary 的训练授权依赖 audit/radial-admission/tube/dataset 完整链，而不依赖 KPI。

**Step 2: 运行 RED**

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/quasi-exp-py311-packages LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 -m pytest -q tests/test_true_ellipse_standard_domain_v7.py
```

Expected: 旧的 105 mm 短路测试和新授权字段断言失败。

**Step 3: 最小实现**

- 升级 V7 runner/tube/dataset strategy version。
- 增加 `target_105_achieved`、`downstream_radial_admission_gate_pass` 和授权原因。
- tube 尝试上限取当前可用 radial strict frontier；正式 tube/data 前沿取不超过该上限、且其以内所有要求半径形成完整通过前缀的最高注册 checkpoint。外层失败保留诊断但不阻断较小完整前沿的训练。
- cache fail-closed 检查新策略字段、任务指纹和 artifact hash。

**Step 4: 运行 GREEN**

运行 Step 2 同一命令，预期全部通过。

### Task 3: 动态 support-backed holdout 与正式训练协议

**Files:**

- Modify: `tests/test_true_ellipse_standard_domain_v7.py`
- Modify: `tests/test_true_ellipse_standard_domain_training_v7.py`
- Modify: `scripts/analysis/run_true_ellipse_standard_domain_v7.py`
- Modify: `scripts/analysis/run_true_ellipse_standard_domain_training_v7.py`

**Step 1: 写失败测试**

- 候选 test radius 必须从当前 tube strict frontier 向下选择，而不是硬编码 `>=105`。
- 102.5/95.0 两个 radius support 均通过时，必须注册 `Rtest=102.5`、`Rval=95.0`。
- test radius 不得泄漏到 train/nonformal 访问链，validation/test challenge 都必须存在且 hash 当前。
- 正式训练协议接受经上游注册的 102.5/95.0 holdout；`target_105_achieved` 只进入报告，不进入训练协议的 all-gates 合取。

**Step 2: 运行 RED**

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/quasi-exp-py311-packages LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 -m pytest -q tests/test_true_ellipse_standard_domain_v7.py tests/test_true_ellipse_standard_domain_training_v7.py
```

Expected: 105 mm protocol check/候选过滤断言失败。

**Step 3: 最小实现**

- 用最高 support-backed 当前严格半径选择动态 holdout。
- 在 dataset report 中记录候选顺序、选择理由、frontier/KPI 和 support hashes。
- 将训练协议检查改为上游注册 holdout 与当前 frontier 一致、gap=7.5 mm、上游正式数据链通过。
- 保留 test access 只在 formal final evaluation 打开的现有 fail-closed 行为。

**Step 4: 运行 GREEN**

运行 Step 2 同一命令，预期全部通过。

### Task 4: 实现独立 kappa 候选 policy sweep

**Files:**

- Create: `tests/test_true_ellipse_standard_domain_gate_sweep_v7.py`
- Create: `scripts/analysis/run_true_ellipse_standard_domain_gate_sweep_v7.py`
- Modify: `scripts/analysis/run_true_ellipse_standard_domain_v7.py`
- Modify: `scripts/analysis/run_true_ellipse_radial_bundle_v6.py`

**Step 1: 写失败测试**

- CLI 默认且只接受 `150,200,250,300,400`，formal sweep 必须 `stop_after_first_failed_job=false`。
- 每个候选 policy 必须要求 4 cuts × 2 predictors、cut invariance 和 deterministic repeatability。
- 缺失任一 job/tube/support/five-seed 证据时 `policy_registration_allowed=false`。
- sweep 输出目录、policy fingerprint 与 legacy strict 输出隔离。

**Step 2: 运行 RED**

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/quasi-exp-py311-packages LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 -m pytest -q tests/test_true_ellipse_standard_domain_gate_sweep_v7.py
```

Expected: 新 runner 尚不存在而失败。

**Step 3: 最小实现**

- 先以 downstream admission 完整计算困难 radius 的八 job/cut/repeatability 证据。
- 对每个 kappa 候选阈值从已审计 job 重新计算连续 policy frontier。
- 为每个候选建立隔离的 tube/dataset/training 子目录；允许 artifact hash 相同的只读复用，但每个 policy 报告单独指纹。
- 汇总所有阶段 gate、模型五 seed 通过数和注册建议；不自动改正式阈值。

**Step 4: 运行 GREEN**

运行 Step 2 同一命令，预期全部通过。

### Task 5: 运行 gate-v2 数据链、正式模型与 sweep

**Files:**

- Generate: `runs/true_ellipse_standard_domain_v7_gate_v2/**`
- Generate: `runs/true_ellipse_standard_domain_training_v7_gate_v2/**`
- Generate: `runs/true_ellipse_standard_domain_gate_sweep_v7/**`
- Modify: `docs/TrueEllipseStandardDomainV7实验记录.md`
- Create: `docs/checkpoints/2026-07-17-true-ellipse-standard-domain-v7-gate-v2.md`

**Step 1: 运行 gate-v2 几何/tube/dataset**

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/quasi-exp-py311-packages LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 scripts/analysis/run_true_ellipse_standard_domain_v7.py --out-dir /mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_v7_gate_v2 --skip-existing
```

Expected: 若当前证据保持一致，注册 102.5/95.0 动态 holdout；任一质量/support/challenge gate 失败则 fail-closed 并记录准确原因。

**Step 2: 运行正式 48 配置筛选与五 seed 模型**

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/quasi-exp-py311-packages LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 scripts/analysis/run_true_ellipse_standard_domain_training_v7.py --upstream-dir /mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_v7_gate_v2 --out-dir /mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_training_v7_gate_v2 --skip-existing
```

Expected: 只在 upstream 全链通过时运行；五 seed 结果和真实 model gate 如实登记。

**Step 3: 生成各模型对应 3D 轨迹**

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/quasi-exp-py311-packages LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 scripts/analysis/plot_true_ellipse_standard_domain_model_trajectories_v7.py --run-dir /mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_training_v7_gate_v2
```

Expected: 每个模型/seed 的 validation/test 3D 轨迹采用现有绘图方式和可辨识的斜视角，不出现退化成直线的视角。

**Step 4: 运行非正式阈值 sweep**

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/quasi-exp-py311-packages LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 scripts/analysis/run_true_ellipse_standard_domain_gate_sweep_v7.py --out-dir /mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_gate_sweep_v7 --skip-existing
```

Expected: 五个阈值均生成完整证据矩阵；只给出 evidence-backed 推荐，不修改注册 policy。

### Task 6: 全量验证、artifact 审计和文档冻结

**Files:**

- Modify: `docs/TrueEllipseStandardDomainV7实验记录.md`
- Create: `docs/checkpoints/2026-07-17-true-ellipse-standard-domain-v7-gate-v2.md`

**Step 1: 运行定向与全量测试**

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/quasi-exp-py311-packages LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 -m pytest -q
```

Expected: 全部通过。

**Step 2: 审计所有注册 artifact**

- 逐一复算 dataset、challenge、model、prediction、图像 manifest 的 SHA-256。
- 检查 radius/trajectory/sample leakage、test access、五 seed 完整性、8-job 完整性。
- 检查正式 claim 与 KPI/diagnostic 字段没有语义串线。

**Step 3: 冻结结论**

- 文档记录 102.5/95.0 模型结果、105 KPI 状态、每个 kappa 候选的全链结果与是否可注册。
- 如果没有候选满足全部证据，明确保持 150；如果有，也只提出下一版注册建议，不在本次 sweep 中追溯改写 V7 strict 结论。

**Step 4: 交付前审查**

- 使用 `verification-before-completion` 做证据核对。
- 使用 `code-review`/`requesting-code-review` 检查 fixed point 之后的完整 diff。
- 用户未授权前不提交、不 push；完成后按 `finishing-a-development-branch` 提供合并/PR/保留/丢弃四个选项。
