# 方案A执行记录（inverse 数据生成）

## 目标
- 将原先 `beta -> theta -> FK(xyz) + PSO(T)` 的正向采样流程，扩展为“给定 `xyz` 目标，反解 `beta` 与 `T`”的数据生成流程。
- 保留旧流程可用（`dataset.mode=forward`），新增反解流程（`dataset.mode=inverse_joint`）。

## 实施步骤与验收

### Step 1. 新增 inverse 求解器与双模式生成
- 代码改动：
  - `src/quasi_exp/opt/pso_inverse.py`
  - `scripts/generate_dataset.py`
  - `scripts/worker_generate_sample.py`
  - `src/quasi_exp/opt/__init__.py`
- 核心实现：
  - 反解流程采用两阶段：
    1) `beta`-PSO：最小化 `xyz` 误差 + `beta` L2 正则；
    2) 固定 `beta` 后调用现有张力 PSO 解 `T`，并用准静态残差筛选。
  - 输出结构保持与旧数据一致：`x,y,z + theta_1..30 + tension_1..12`，并在 `dataset_meta` 额外记录：
    - `target_x_m,target_y_m,target_z_m,xyz_err_m`
- 验收方式：
  - `python3 -m compileall src/quasi_exp/opt/pso_inverse.py scripts/generate_dataset.py scripts/worker_generate_sample.py`
  - 结果：通过。

### Step 2. smoke 验收（流程通路）
- 配置：`configs/robot_rods_only_20deg_inverse_smoke.yaml`
- 命令：
  - `python3 scripts/generate_dataset.py --config configs/robot_rods_only_20deg_inverse_smoke.yaml --num-samples 20 --workers 4 --max-tried 400`
- 验收门槛：
  - 能稳定生成 `dataset.parquet` 与 `dataset_meta.parquet`
  - `meta` 包含 `xyz_err_m`
  - 接受率明显大于 0
- 结果（已通过）：
  - 接受率：`20/38 = 52.63%`
  - `xyz_err_m`: mean `0.0394m`, max `0.0561m`（smoke 阈值较宽）

### Step 3. 2k 正式验收（严格阈值）
- 配置：`configs/robot_rods_only_20deg_inverse_2k.yaml`
- 命令：
  - `python3 scripts/generate_dataset.py --config configs/robot_rods_only_20deg_inverse_2k.yaml --num-samples 2000 --workers 16 --max-tried 20000`
- 验收门槛：
  - `accepted=2000`
  - `xyz_err_m < 0.012`（配置阈值）
  - `rms_rnorm < 0.06`（配置阈值）
- 结果（已通过）：
  - 输出：`data/rods_only_20deg_inverse_2k/dataset.parquet`
  - 耗时：`631.35s`，吞吐 `0.3157s/accepted`
  - 接受率：`2000/3061 = 65.34%`
  - `xyz_err_m`: mean `0.00373m`, p95 `0.00817m`, max `0.01196m`
  - `rms_rnorm`: mean `0.03784`, p95 `0.05541`, max `0.05995`
  - 张力打满样本比：`10.7%`

### Step 4. 新数据可学习性 smoke（2k）
- 命令：
  - `python3 scripts/baselines/run_baselines.py --dataset data/rods_only_20deg_inverse_2k/dataset.parquet --out-dir runs/baselines_inverse_2k_smoke --models mlp,rf,lgbm,knn --robot-config configs/robot_rods_only_20deg_inverse_2k.yaml --split iid --seed 20260208 --save-curves --save-preds 1000 --summary-split test --eval-splits train,val,test --n-jobs -1 --wrapper-n-jobs 1`
  - `python3 scripts/baselines/plot_baselines.py --out-dir runs/baselines_inverse_2k_smoke`
- 结果：
  - 指标输出：`runs/baselines_inverse_2k_smoke/all_metrics.json`
  - 可视化输出：`runs/baselines_inverse_2k_smoke/plots`

## 全量任务状态（100k）
- 已启动后台任务：
  - tmux 会话：`quasi_inverse_100k`
  - 日志：`runs/diagnostics/inverse_100k_gen.log`
  - 输出目录：`data/rods_only_20deg_inverse_100k`
- 启动命令：
  - `python3 scripts/generate_dataset.py --config configs/robot_rods_only_20deg_inverse_100k.yaml --num-samples 100000 --workers 16 --max-tried 180000`

## 监控命令
- 查看 tmux：
  - `tmux ls | rg quasi_inverse_100k`
- 查看进度日志：
  - `tail -n 20 runs/diagnostics/inverse_100k_gen.log`
- 查看进程占用：
  - `ps -eo pid,ppid,%cpu,%mem,etime,cmd | rg "inverse_100k|worker_generate_sample.py"`
