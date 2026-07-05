# Fixed-Layer 100k 数据集生成与模型测试执行计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将已通过 20k 验证的 `s1=0.125,s2=0.25` fixed-layer 单支路数据路线扩展到 100k，并训练现有 MLP/TF MLP 模型评估拟合性能。

**Architecture:** 使用现有 fixed-layer scale pipeline 生成 `100k raw`，再用 graph canonical relabel 生成 `100k relabel`。训练前先做完整质量验收，避免在张力标签质量不达标时浪费 GPU 训练时间。

**Tech Stack:** Python, pytest, parquet datasets, tmux longrun helper, existing dante conda environment, existing baseline training scripts.

---

## Task 1: Pipeline accept-infeasible relabel 开关

**Files:**
- Modify: `scripts/pipelines/run_fixed_layer_scale_experiment.py`
- Modify: `tests/test_fixed_layer_scale_pipeline.py`

**Steps:**
1. 增加失败测试，确认 `build_relabel_command(..., accept_infeasible=True)` 会包含 `--accept-infeasible`。
2. 增加失败测试，确认 parser 接受 `--accept-infeasible-relabel`。
3. 修改 pipeline，给 `build_relabel_command` 增加 `accept_infeasible` 参数。
4. 当 CLI 指定 `--accept-infeasible-relabel` 时，将 `--accept-infeasible` 透传给 `scripts/analysis/relabel_tension_graph_canonical.py`。
5. 在 summary 中记录 `relabel.accept_infeasible`。
6. 运行：
   ```bash
   /mnt/ML_projects/conda_envs/dante_env/bin/python -m pytest tests/test_fixed_layer_scale_pipeline.py -q
   ```

## Task 2: 100k 生成与质量检查

**Files:**
- Output: `data/priority_grid_fixed_layer_s1_0125_s2_0250_100k`
- Output: `data/priority_grid_fixed_layer_s1_0125_s2_0250_100k_relabel_t1_k32_w40_huber_mean_iter1`
- Output: `runs/diagnostics/fixed_layer_scale_s1_0125_s2_0250_v1/summary.json`
- Output: `runs/diagnostics/fixed_layer_scale_s1_0125_s2_0250_v1/summary.md`

**Steps:**
1. 检查当前 longrun 状态：
   ```bash
   scripts/pipelines/longrun_tmux.sh status
   ```
2. 如果无任务运行，启动生成与质量阶段：
   ```bash
   scripts/pipelines/longrun_tmux.sh start fixed_layer_100k_generate_quality runs/logs/fixed_layer_100k_generate_quality.log -- \
     /mnt/ML_projects/conda_envs/dante_env/bin/python scripts/pipelines/run_fixed_layer_scale_experiment.py \
       --config configs/robot_rods_only_priority_grid_third_joint_first_v1.yaml \
       --s1 0.125 \
       --s2 0.25 \
       --seed 20260614 \
       --train-seed 20260207 \
       --workers 8 \
       --anchor-k 32 \
       --w-anchor 40 \
       --anchor-stat huber_mean \
       --stages quality20k,maybe100k,relabel100k,quality100k \
       --accept-infeasible-relabel
   ```
3. 监控：
   ```bash
   scripts/pipelines/longrun_tmux.sh status
   scripts/pipelines/longrun_tmux.sh logs 80
   ```

## Task 3: 100k 质量验收

**Pass criteria:**
- `100k_raw` 和 `100k_relabel` 行数期望为 `100000`，低于 `99900` 失败。
- `hard_gate_passed=True`。
- `rms_rnorm_q95 <= 0.06`。
- `max_tension_n <= 2000`。
- `multi_branch_ball_ratio <= 0.02`。
- `all10_theta_p95_deg <= 0.5`。
- relabel 张力连续性：
  - `all10_tension_p95_n <= 50`
  - `beta_close_tension_p95_n <= 50`
  - `same_beta_tension_p95_n <= 90`
  - `xyz_nn_tension_mae_n <= 20`
  - `beta_nn_tension_mae_n <= 20`
- `accepted_infeasible / rows <= 0.1%` 可接受，`>0.5%` 阻止训练。

## Task 4: 100k 模型训练

**Only run if Task 3 passes.**

**Command:**
```bash
scripts/pipelines/longrun_tmux.sh start fixed_layer_100k_train runs/logs/fixed_layer_100k_train.log -- \
  /mnt/ML_projects/conda_envs/dante_env/bin/python scripts/pipelines/run_fixed_layer_scale_experiment.py \
    --config configs/robot_rods_only_priority_grid_third_joint_first_v1.yaml \
    --s1 0.125 \
    --s2 0.25 \
    --seed 20260614 \
    --train-seed 20260207 \
    --workers 8 \
    --anchor-k 32 \
    --w-anchor 40 \
    --anchor-stat huber_mean \
    --stages train100k \
    --accept-infeasible-relabel
```

**Models and splits:**
- datasets: `100k_raw`, `100k_relabel`
- splits: `iid`, `radius`, `beta_block`, `angular_sector`
- models: `mlp`, `mlp_large`, `tf_mlp`, `tf_mlp_large`

## Task 5: 结果记录

**Files:**
- Modify: `docs/FixedLayer20k100k扩展实验记录.md`

**Report:**
- 100k raw 生成耗时。
- 100k relabel 耗时。
- infeasible 样本数量和比例。
- raw vs relabel 质量指标对比。
- raw vs relabel 模型指标对比。
- 判断模型瓶颈更偏向模型容量、OOD 泛化，还是张力标签生成。
