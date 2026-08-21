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

3. 阻塞等待当前任务结束，并返回任务记录的退出码
```bash
bash scripts/pipelines/longrun_tmux.sh wait
```

`wait` 只读取 PID、状态和退出码，不会启动、重启、停止或修改任务。worker
存活时每 300 秒检查一次；如果 PID 已消失但状态不是已知终态，`wait` 会以
非零码失败退出，不会无限等待。

4. 查看日志（最近 120 行）
```bash
bash scripts/pipelines/longrun_tmux.sh logs 120
```

5. 进入 tmux 观察实时输出
```bash
bash scripts/pipelines/longrun_tmux.sh attach
```

6. 停止当前任务
```bash
bash scripts/pipelines/longrun_tmux.sh stop
```

7. 启动“自动汇报” watcher（默认 300 秒，按进度自适应）
```bash
nohup bash scripts/pipelines/watch_inverse_tune_progress.sh 300 runs/maintenance/inverse_tune_watch.log \
  > runs/maintenance/inverse_tune_watch.nohup.log 2>&1 &
```

8. 查看 watcher 状态与输出
```bash
cat runs/maintenance/inverse_tune_watch.pid
tail -n 50 runs/maintenance/inverse_tune_watch.log
```

9. watcher 自适应规则（默认）
- 进度停滞或很慢：缩短汇报间隔（更快告警）
- 进度稳定推进：按基础间隔汇报
- 接近完成（>=90%）：加密到更短间隔
- 任务结束：自动写结论报告 `runs/maintenance/inverse_tune_final_report.md`

## 防重复启动机制
- `start` 会检查 `runs/maintenance/longrun/current.pid`。
- 如果已有活跃 PID，脚本直接拒绝再次启动，避免并发污染同一输出目录。
- 任务结束后自动写入：
  - `current.status`（运行中为 `running`；终态为 `success`/`failed`/`stopped_by_user`）
  - `current.rc`
  - `current.started_at` / `current.finished_at`
  - `current.log`（日志路径）

## 自定义任务示例
```bash
bash scripts/pipelines/longrun_tmux.sh start tune_inverse_2k runs/maintenance/tune_inverse_2k.log -- \
  python3 scripts/pipelines/tune_inverse_5deg_2k_gate.py \
    --python-bin /mnt/ML_projects/conda_envs/dante_env/bin/python \
    --max-attempts 8 --num-samples 2000 \
  && bash scripts/pipelines/longrun_tmux.sh wait
```

Codex 的长时间 monitor 必须持有上面这条 `start ... && wait` 前台命令链，
以项目 launcher 的状态与退出码为唯一事实来源；不要创建
`run_<task>_checkpoint.sh` 一类任务专用轮询脚本。

## Codex 协作建议
- 新的 `long_wait_monitor` 在一个前台命令链中持有 `start ... && wait`；
  Codex 主线程不自行高频轮询。人工排障时仍可使用 `status` 和 `logs`。
- 如果出现 `turn_aborted`，优先检查：
  1) `bash scripts/pipelines/longrun_tmux.sh status`
  2) `bash scripts/pipelines/longrun_tmux.sh logs 120`
- 若任务确已终止，再决定是否重启，不直接覆盖旧日志。
- 对逆解调参任务，建议同时启用 `watch_inverse_tune_progress.sh`，让进度/精度快照按固定周期写日志，便于定时汇报。
