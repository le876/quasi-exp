# Generalized Ellipse Region V11.4 Full-Loop Feasible 实验记录

## 1. 实验目标

V11.4 将 branch 的定义从单个 root 点提升为满足完整闭环约束的轨迹：

1. 映射 root IK fibre；
2. 运行双向 continuation viability；
3. 只保留 residual、joint margin 和 trust 均通过硬约束的 phase 节点；
4. 先在 hard-feasible graph 中选择闭环 cycle，再应用 canonical posture；
5. 只有 720-phase Formal 严格通过后，才授权 tube、区域数据集和 Student。

Formal Gate 沿用预注册阈值，本次并行化不修改 Gate、随机种子、候选生成、数值求解、稳定排序或 shared method ranking。

## 2. 串行基线与中断现象

首次 Formal 使用单进程、单 BLAS 线程运行：

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python \
  scripts/analysis/run_generalized_ellipse_full_loop_v11_4.py \
  --preset formal \
  --stage branch \
  --project-root /mnt/ML_projects/quasi_exp
```

已观测事实：

- 运行时间约 `5h59m23s` 后按用户授权停止，长任务状态为 `stopped_by_user`；
- Python 主线程持续约占一个 CPU 核心，属于计算密集状态；
- Python RSS 约 `846 MB`，系统仍有约 `16 GiB available`；
- 块设备实际读写很低，未发现磁盘吞吐瓶颈；
- 未使用 GPU；
- 输出目录约 `94 MiB`；
- root-fibre 和 viability 阶段完整；
- Pilot 已写出两个 candidate 的 E0、3 个 `hard_graph`、308 个 K/method 证据文件和 106 个 `gate.json`；
- `pilot_matrix.csv`、`FULL_LOOP_EXPERIMENT_COMPLETED.json` 和 `04_formal/gate.json` 均不存在，因此该目录不能作为完整实验结论。

这说明主要瓶颈是 Python 层 Pilot matrix 的串行粗粒度任务，而不是内存、磁盘、GPU 或大型 BLAS kernel。仅把 `OMP_NUM_THREADS` 从 1 改为 16 不会并行 Python 嵌套循环，并可能引入小矩阵线程调度开销。

## 3. 确定性多进程设计

Formal Pilot 的第一层独立任务固定为：

```text
candidate A2_178 / Nroot 256
candidate A2_178 / Nroot 1024
candidate A2_143 / Nroot 256
candidate A2_143 / Nroot 1024
```

执行约束：

- 主进程按 frozen candidate 顺序和 `root_seed_counts` 顺序生成 4 个 task JSON；
- 每个 task 由独立 `subprocess` 执行，不使用 `multiprocessing.Pool`、共享可变缓存或 SemLock；
- 初始并发数为 4；
- 每个 worker 固定
  `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、
  `MKL_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`；
- solver seed 继续使用原有
  `graph_seed + 100000 * candidate_offset + root_seed_count`；
- 每个 worker 只写自己的
  `candidate_id/Nroot_xxxx` 子树；
- E0 由每个 candidate 的第一个 root task 唯一写入，避免并发覆盖；
- worker 必须生成预期行数、全部行级 gate 和带 SHA-256 的 slice gate；
- 主进程等待全部 slice 完成后按原串行嵌套循环顺序稳定归并；
- 任一 worker 非零退出、切片缺失、行数不足、任务身份不匹配或 CSV 哈希不匹配都会 fail closed；
- 已存在但不完整的切片不会被静默覆盖。

当前自然切片只有 4 个，因此即使请求 8 workers，有效 worker 也会被限制为 4。是否继续细分 hard-graph 或 E1–E4 matrix，要根据 4-worker Formal 的负载不均衡和资源数据决定。

## 4. 资源与运行现象证据

每次 Pilot 在以下文件记录可审计的并行证据：

```text
03_pilot_matrix/pilot_parallel_manifest.json
```

字段包括：

- requested/configured/effective worker 数；
- logical CPU 数；
- worker 数受限原因；
- 每个 task 的 PID、wall time、CPU time、CPU utilization、peak RSS；
- worker stdout/stderr 日志路径；
- 并发 worker 峰值 RSS；
- 启动和结束时的系统 memory/swap；
- aggregate 和 mean effective worker utilization；
- 完整 task 状态与 failure tail。

正式运行结束后，本节补充 4-worker 实测数据、与串行基线的阶段耗时对比，以及是否需要扩大并行度的判断。

## 5. 并行语义验证

实现完成后的验证：

- V11.4 定向测试：两个 Python 环境均 `23 passed`；
- V11/V11.3/V11.4 相关回归：`64 passed`；
- 真实 2-worker smoke：通过；
- 同一 smoke 分别使用 1 worker 和 2 workers：
  `pilot_matrix.csv` 逐单元一致，
  `selected_formal_method.json` 一致；
- 全量测试未发现新的 V11.4 回归；仍为 37 个既有 V7 失败，其中多数来自标准环境 Python 3.10 缺少 `enum.StrEnum`，其余为既有 registered gate 预期。

## 6. 结果状态

截至本记录当前版本：

- 串行残留：已整体归档为
  `runs/generalized_ellipse_region_v11_full_loop_feasible_branch_serial_interrupted_20260723`；
- 4-worker Formal：等待在干净输出目录启动；
- Formal Gate：尚无结论；
- downstream tube/dataset/student：未授权。
