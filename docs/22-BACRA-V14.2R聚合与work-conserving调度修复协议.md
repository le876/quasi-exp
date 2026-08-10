# BACRA V14.2R 聚合与 work-conserving 调度修复协议

## 1. retry2 的终止口径

`bacra_v14_2r_stitched_atlas_retry2_4cad4ac` 于 2026-08-10 11:00:03 CST
经 canonical launcher 记录为 `stopped_by_user`，退出码为 130。该运行保留了前
16 个完整 search-stability jobs 和 `patch_09/task_graph_refined/chart_initial`；
当时两个 `fragment_reaudit` shards 均未写出完成报告。

该运行记为 `operationally_aborted_for_performance_and_aggregation_repair`。它没有生成
`06_search_stability/gate.json`，不得解释为 mechanism、representation 或无状态
`xyz-only` inverse 的科学失败。

## 2. 必修正确性问题

retry2 固定点将 tuple 型 `BETA_COLUMNS` 直接传给 pandas `.loc`。在标准解释器和
真实 retry2 primary-atlas artifacts 上，`_frame_stability()` 会确定性触发
`AssertionError`。retry3 必须使用显式 column list，并由 runner 层回归测试覆盖真实
DataFrame 索引语义。

## 3. work-conserving 执行 contract

每个 patch/audit phase 仍先封存完整 registry，但统一注册 12 个稳定 hash shards。
所有并行 patch coordinators 共享一个 stage-local、容量为 12 的 advisory-lock token
pool。只有取得 token 的 coordinator 才能启动一个单线程 numerical shard；shard
结束或进程退出时 token 自动释放，其他 ready shard 可立即回填。

该调度只改变执行顺序和粒度，不删除或抽样以下对象：

- node-induced retained edges；
- root paths；
- fundamental cycle basis；
- multipath tree/chord；
- forward/reverse；
- 每方向三次注册扰动 repeat；
- R0、R1、R2；
- fragment re-audit；
- primary certificate；
- MILP strong reference。

每个 shard 继续封存 source/config/input hash、完整 execution keys、Parquet hash 和
completion report。聚合继续要求 shard exact set、schedule exact set、无重复 key、
完整 source/config/input closure，并以 fail-closed 方式处理 worker failure。

## 4. 可观测性

每个 shard 每完成 256 个 schedules 原子更新一次 `progress.json`，只记录
`completed/total/status`。该文件不参与 continuation 结果、Gate 或 input closure，
不得用于删减工作量。最终 `executions.parquet` 和 shard `report.json` 仍只在完整执行
后写出。

## 5. 性能准入与 retry3

完整 retry3 前必须在新的 clean SHA 和新的 output root 运行 12-shard patch-07
benchmark。准入要求：

1. required audit phases 与 exact-set Gates 完整；
2. 每个 phase 注册 12 shards；
3. source/config hash 与 benchmark inventory 一致；
4. worktree clean；
5. `runtime_s / 64 <= 120 s/parent`；
6. regression tests 证明 token 释放后可回填，且全局容量不超过 12。

benchmark 通过后才允许启动
`scripts/pipelines/run_bacra_v14_2r_stitched_atlas_retry3.sh`。retry2 artifacts 只作为
只读诊断和历史 2-shard 性能基线，不导入 retry3 output root，也不伪装为同一 fixed
point。

跨 variant execution cache 不属于 retry3。只有完整数值输入、repeat 独立性、Gate、
compared structures 和 artifact closure 的独立等价实验通过后，才可另行考虑复用。
