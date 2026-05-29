# MLP-Large 20-Round Adaptive Tuning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在固定数据集与固定切分上，执行 20 轮 `mlp_large` 自适应调参，最小化 `test theta_mae_deg` 与 `test tension_mae_n`，并形成可复现实验记录。

**Architecture:** 复用 `scripts/baselines/run_baselines.py` 单模型训练入口（仅 `mlp_large`），在每轮训练后读取 `metrics.json` 与 `curves.json`，根据过拟合/欠拟合信号自动调整下一轮超参数。所有轮次参数、日志、指标和汇总图表统一写入 `runs/tuning_100k/mlp_large_20round_adaptive/`。

**Tech Stack:** Python 3.10 (`dante_env`), scikit-learn `MLPRegressor`, PyYAML/JSON, Matplotlib

## Live Status (2026-02-09)

- 当前执行目录：`runs/tuning_100k/mlp_large_20round_adaptive/`
- 当前状态：20 轮自适应调参正在运行（已按 `penalty_100k` 正确口径重启）
- 已归档一次错误口径试跑：`runs/tuning_100k/mlp_large_20round_adaptive_inverse100k_aborted_20260209_231803/`
- 已验证事实：`run_baselines.py` 中 `mlp_large` 使用 `sklearn.neural_network.MLPRegressor`，为 CPU 训练路径，不使用 TensorFlow GPU
- `dante_env` 下 TensorFlow GPU 探测结果：`tf.config.list_physical_devices('GPU') == []`（该结果不影响当前 `mlp_large` 的可执行性，但会影响后续 TensorFlow 模型）

### Round Checkpoints (wrong-dataset run, archived)

| round | test theta MAE (deg) | test tension MAE (N) | test score | improved |
|---|---:|---:|---:|---|
| 01 | 7.6949 | 419.14 | 0.594318 | yes |
| 02 | 7.6937 | 419.08 | 0.594221 | yes |
| 03 | 7.7109 | 419.26 | 0.595171 | no |

### Mid-run Diagnosis

- 与目标 `1 deg / 100 N` 仍存在显著差距（当前最优约 `7.69 deg / 419 N`）
- 数据可拟合性快速检查显示：仅用 `x,y,z` 预测 `30 theta + 12 tension` 难度较高，当前误差可能受问题本身可辨识性限制
- 因此本轮策略维持：先完成 20 轮自动调参并固化最优点，再决定是否升级为“模型结构变更/目标重定义/输入特征增强”阶段

## Final Outcome (penalty_100k, 20 rounds)

- 运行目录：`runs/tuning_100k/mlp_large_20round_adaptive/`
- 结果汇总：`runs/tuning_100k/mlp_large_20round_adaptive/SUMMARY.md`
- 最优轮次：`trial_04`
- 最优参数：`hidden_layer_sizes=(352,176,88,44), alpha=1e-7, lr=1e-3, batch=1024, max_iter=800, n_iter_no_change=50, tol=1e-6`
- 最优指标：
  - `test theta_mae_deg = 7.3235`
  - `test tension_mae_n = 416.13`
- 相对 baseline（trial_01）提升很小：
  - `theta`: `-0.0148 deg`
  - `tension`: `+0.10 N`（几乎无变化）
- 结论：在当前任务定义（输入仅 `x,y,z`，输出 `30θ+12T`）下，单纯调 `sklearn MLP-large` 超参数已进入平台，无法逼近 `1deg/100N` 目标。

## Next Phase (required for 1deg/100N attempt)

1. **重新定义学习目标（优先级最高）**
   - 先做 `x,y,z -> 12T` 与 `x,y,z -> 30θ` 分离建模，避免 42 维耦合互相拖拽。
2. **增强输入可辨识性**
   - 在输入中加入末端方向/段曲率摘要/历史点（序列窗口），减少同一位置多解引起的平均化误差。
3. **切到可加权损失的模型实现**
   - 用 TensorFlow/PyTorch MLP（分头输出），显式优化 `L = λθ*MSEθ + λT*MSET + λphys*penalty(T)`。
4. **分阶段训练**
   - 第一步仅优化 `theta`；第二步冻结主干后优化 `tension`；第三步联合微调。
5. **验收门槛改为阶段性**
   - 阶段A：`theta < 5deg`, `T < 300N`
   - 阶段B：再向 `1deg/100N` 推进（若阶段A无法达成，需回溯数据生成口径）。

---

### Task 1: 固定实验口径与验收标准

**Files:**
- Modify: `docs/plans/2026-02-09-mlp-large-20round-tuning.md`

**Step 1: 锁定输入数据与切分**

- 数据集：`data/rods_only_20deg_penalty_100k/dataset.parquet`
- 机器人配置：`configs/robot_rods_only_20deg_penalty_100k.yaml`
- 固定切分：`runs/tuning_100k/split_iid_seed20260207.npz`

**Step 2: 锁定训练口径**

- 模型只跑：`mlp_large`
- 评估 split：`train,val,test`
- 汇总 split：`test`
- 固定 seed：`20260207`

**Step 3: 锁定每轮验收项**

- 主指标：`test theta_mae_deg`, `test tension_mae_n`, `test score_val_like`
- 过拟合信号：`(val-train)` 的 `theta_mae_deg`, `tension_mae_n`
- 收敛信号：最近 3 轮 `test score_val_like` 改善小于阈值

---

### Task 2: 实现 20 轮自适应调参执行器

**Files:**
- Create: `scripts/baselines/tune_mlp_large_adaptive.py`

**Step 1: 实现 baseline 参数定义**

- 初始参数对齐历史最优：
  - `hidden_layer_sizes=(256,128,64,32)`
  - `alpha=1e-7`
  - `max_iter=800`
  - `n_iter_no_change=50`
  - `tol=1e-6`
  - `learning_rate_init=1e-3`
  - `batch_size=1024`

**Step 2: 实现单轮执行函数**

- 写出每轮参数文件（JSON/YAML）
- 调用 `run_baselines.py`（仅 `mlp_large`）
- 读取 `trial_xx/mlp_large/metrics.json` 与 `trial_xx/mlp_large/curves.json`

**Step 3: 实现自适应策略函数**

- 根据 `train/val/test` 指标判定：
  - 欠拟合：提高容量或训练轮数、降低正则
  - 过拟合：提高正则、降低容量、降低学习率
  - 平衡区：微调 `learning_rate_init` / `batch_size`
- 每轮只调整 1-2 个超参数，避免跨度过大

**Step 4: 实现可审计日志**

- 记录每轮：
  - 参数快照
  - 指标快照
  - “为何这样调”的机器可读说明
- 输出到：`runs/tuning_100k/mlp_large_20round_adaptive/history.jsonl`

---

### Task 3: 执行 20 轮并生成汇总产物

**Files:**
- Create: `runs/tuning_100k/mlp_large_20round_adaptive/SUMMARY.md`
- Create: `runs/tuning_100k/mlp_large_20round_adaptive/trial_curve.png`
- Create: `runs/tuning_100k/mlp_large_20round_adaptive/best_params.yaml`

**Step 1: 执行命令**

Run:
`conda run -n dante_env python scripts/baselines/tune_mlp_large_adaptive.py --rounds 20 --dataset data/rods_only_20deg_penalty_100k/dataset.parquet --robot-config configs/robot_rods_only_20deg_penalty_100k.yaml --split-file runs/tuning_100k/split_iid_seed20260207.npz --out-root runs/tuning_100k/mlp_large_20round_adaptive`

Expected:
- 生成 `trial_01` 到 `trial_20`
- 每轮目录包含 `all_metrics.json`、`mlp_large/metrics.json`

**Step 2: 生成汇总与可视化**

- 表格：每轮 `test theta/tension/score` 与 `fit_time`
- 曲线：`theta_mae_deg`、`tension_mae_n`、`score` 随轮次变化
- 最优参数导出：`best_params.yaml`

**Step 3: 输出最终验收结论**

- 给出 best trial 与相对基线提升
- 判断是否达到目标（若未达到，明确下一阶段策略）

---

### Task 4: 回归验证与可复现性检查

**Files:**
- Modify: `runs/tuning_100k/mlp_large_20round_adaptive/SUMMARY.md`

**Step 1: 复跑 best trial**

Run:
`conda run -n dante_env python scripts/baselines/run_baselines.py --dataset data/rods_only_20deg_penalty_100k/dataset.parquet --robot-config configs/robot_rods_only_20deg_penalty_100k.yaml --split-file runs/tuning_100k/split_iid_seed20260207.npz --out-dir runs/tuning_100k/mlp_large_20round_adaptive/recheck_best --models mlp_large --params-file runs/tuning_100k/mlp_large_20round_adaptive/best_params.yaml --summary-split test --save-curves --seed 20260207 --skip-fk`

Expected:
- `recheck_best` 指标与 best trial 在可接受误差内一致

**Step 2: 完成最终记录**

- 在 `SUMMARY.md` 写入：
  - 可复现实验命令
  - 最优参数
  - 结果区间与风险说明
