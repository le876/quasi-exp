# 长任务防中断运行 SOP（tmux 单会话）

## 目标
- 让数据集生成/调参在 Codex 会话中断后继续运行。
- 全项目只维护一个固定 `tmux` 会话，避免反复创建多个会话导致混乱。
- Codex 与人工都可以用同一组命令监控状态。

## 固定约定
- 会话名：`quasi_exp_longrun`
- 窗口名：`worker`
- 状态目录：`runs/maintenance/longrun`
- 管理脚本：`scripts/pipelines/longrun_tmux.sh`

## 常用命令
1. 启动任务（建议）
```bash
bash scripts/pipelines/start_inverse_5deg_autogate_longrun.sh
```

2. 查看状态
```bash
bash scripts/pipelines/longrun_tmux.sh status
```

3. 查看日志（最近 120 行）
```bash
bash scripts/pipelines/longrun_tmux.sh logs 120
```

4. 进入 tmux 观察实时输出
```bash
bash scripts/pipelines/longrun_tmux.sh attach
```

5. 停止当前任务
```bash
bash scripts/pipelines/longrun_tmux.sh stop
```

6. 启动“自动汇报” watcher（默认 300 秒，按进度自适应）
```bash
nohup bash scripts/pipelines/watch_inverse_tune_progress.sh 300 runs/maintenance/inverse_tune_watch.log \
  > runs/maintenance/inverse_tune_watch.nohup.log 2>&1 &
```

7. 查看 watcher 状态与输出
```bash
cat runs/maintenance/inverse_tune_watch.pid
tail -n 50 runs/maintenance/inverse_tune_watch.log
```

8. watcher 自适应规则（默认）
- 进度停滞或很慢：缩短汇报间隔（更快告警）
- 进度稳定推进：按基础间隔汇报
- 接近完成（>=90%）：加密到更短间隔
- 任务结束：自动写结论报告 `runs/maintenance/inverse_tune_final_report.md`

## 防重复启动机制
- `start` 会检查 `runs/maintenance/longrun/current.pid`。
- 如果已有活跃 PID，脚本直接拒绝再次启动，避免并发污染同一输出目录。
- 任务结束后自动写入：
  - `current.status`（`success`/`failed`）
  - `current.rc`
  - `current.started_at` / `current.finished_at`
  - `current.log`（日志路径）

## 自定义任务示例
```bash
bash scripts/pipelines/longrun_tmux.sh start tune_inverse_2k runs/maintenance/tune_inverse_2k.log -- \
  python3 scripts/pipelines/tune_inverse_5deg_2k_gate.py \
    --python-bin /mnt/ML_projects/conda_envs/dante_env/bin/python \
    --max-attempts 8 --num-samples 2000
```

## Codex 协作建议
- 长任务运行期间，Codex 只做 `status/logs` 轮询，不在同一前台命令里跑长任务。
- 如果出现 `turn_aborted`，优先检查：
  1) `bash scripts/pipelines/longrun_tmux.sh status`
  2) `bash scripts/pipelines/longrun_tmux.sh logs 120`
- 若任务确已终止，再决定是否重启，不直接覆盖旧日志。
- 对逆解调参任务，建议同时启用 `watch_inverse_tune_progress.sh`，让进度/精度快照按固定周期写日志，便于定时汇报。
