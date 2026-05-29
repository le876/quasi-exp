# Baselines（独立脚本，兼容 Python 3.9+）

该目录下脚本用于运行 `机器学习方法补充实验方案.md` 中的 baseline 对比实验。

设计目标：
- **不依赖** `src/quasi_exp`（避免环境差异导致 import 失败）
- 直接读取 `dataset.parquet`（需要 `pyarrow`）
- 支持输出端点回代误差（内置 DH FK，复用本仓库 DH 规则）

## 入口
- `run_baselines.py`：训练/评估多个模型并输出 `metrics.json`
- `summarize_results.py`：把 `all_metrics.json` 打印成 Markdown 表格（便于贴论文）
- `plot_baselines.py`：根据 `all_metrics.json` 与 `preds_test.parquet` 输出图表（loss/pearson/残差等）

## 推荐环境
- `dante_env`（默认）：已用于当前实验主流程，支持 TensorFlow-GPU；补齐 `pyarrow/lightgbm` 后可跑全套 baseline
- `quant_env`（兼容）：可作为回退环境

## 当前阶段（GPU-only TensorFlow）
当前实验阶段默认采用 `TensorFlow + GPU`，建议显式加 `--backend tf --require-gpu --tf-device gpu`。

```bash
conda run -n dante_env python scripts/baselines/run_baselines.py \
  --dataset data/rods_only_20deg_penalty_100k/dataset.parquet \
  --robot-config configs/robot_rods_only_20deg_penalty_100k.yaml \
  --out-dir runs/tf_only_rods_only_20deg_penalty_100k \
  --backend tf --require-gpu --tf-device gpu \
  --models tf_mlp,tf_mlp_large \
  --feature-set poly_heavy \
  --save-preds 2000 --save-curves \
  --split iid
```

20轮自适应调参（TF MLP-Large）：
```bash
conda run -n dante_env python scripts/baselines/tune_tf_mlp_large_adaptive.py \
  --dataset data/rods_only_20deg_penalty_100k/dataset.parquet \
  --robot-config configs/robot_rods_only_20deg_penalty_100k.yaml \
  --split-file runs/tuning_100k/split_iid_seed20260207.npz \
  --out-root runs/tuning_100k/tf_mlp_large_20round_adaptive
```

## 经典基线（后续对比阶段）
在后续基线对比阶段，可复用同一特征口径（`poly_heavy`）改回 classic/both。

## 示例
```bash
conda run -n dante_env python scripts/baselines/run_baselines.py \
  --dataset data/rods_only_20deg_penalty_100k/dataset.parquet \
  --robot-config configs/robot_rods_only_20deg_penalty_100k.yaml \
  --out-dir runs/baselines_rods_only_20deg_penalty_100k \
  --models mlp,mlp_large,rf,lgbm,knn \
  --save-preds 2000 --save-curves \
  --n-jobs -1 \
  --split iid
```

工作空间外推评估（按 `r=||p||` 把“最远端点”留作 val/test）：
```bash
conda run -n dante_env python scripts/baselines/run_baselines.py \
  --dataset data/rods_only_20deg_penalty_100k/dataset.parquet \
  --robot-config configs/robot_rods_only_20deg_penalty_100k.yaml \
  --out-dir runs/baselines_rods_only_20deg_penalty_100k_radius_split \
  --models mlp,mlp_large,lgbm,knn \
  --save-preds 2000 --save-curves \
  --n-jobs -1 \
  --split radius
```

生成图表：
```bash
conda run -n dante_env python scripts/baselines/plot_baselines.py \
  --out-dir runs/baselines_rods_only_20deg_penalty_100k
```

生成对比表：
```bash
python scripts/baselines/summarize_results.py --all-metrics runs/baselines_rods_only_20deg_penalty_100k/all_metrics.json
```

等待数据集生成完成后自动跑 baseline（适合挂后台）：
```bash
tmux new -d -s baselines20 \
  'cd /mnt/ML_projects/quasi_exp && bash scripts/wait_and_run_baselines.sh dante_env data/rods_only_20deg_penalty_100k/dataset.parquet configs/robot_rods_only_20deg_penalty_100k.yaml runs/baselines_rods_only_20deg_penalty_100k'
```

统一的长任务管理（推荐，避免会话中断后任务丢失）：
```bash
bash scripts/pipelines/longrun_tmux.sh start baselines20 runs/maintenance/baselines20.log -- \
  bash scripts/wait_and_run_baselines.sh dante_env data/rods_only_20deg_penalty_100k/dataset.parquet configs/robot_rods_only_20deg_penalty_100k.yaml runs/baselines_rods_only_20deg_penalty_100k

bash scripts/pipelines/longrun_tmux.sh status
bash scripts/pipelines/longrun_tmux.sh logs 120
```
