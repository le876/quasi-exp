# BACRA V14.2R 性能修复与 retry2 协议

## 1. retry1 的终止口径

`bacra_v14_2r_science_retry1_f5b8977c` 于 2026-08-06 18:42:51 CST 从
`f5b8977c61692b243d5c253850e61c718c6b49d0` 启动，并于 23:17:45 CST
经 canonical launcher 记录为 `stopped_by_user`，退出码为 130。

该运行完成并保留 inventory、replacement `patch_12`、entity diagnostics 和
Reach Round 7；四个 64-cell rooted-baseline worker 在约 4.5 小时后仍未生成
patch report。该运行必须记为 `operationally_aborted_for_performance_redesign`，
不得记为 mechanism、representation 或 stateless inverse 的科学失败。

## 2. 根因

原执行器以 patch 为并行单位。四个 diagnostic patches 因而只能使用四个 CPU
worker。每个 patch 内的 edge、root path、fundamental cycle、multipath 和 repeat
schedules 又在一个进程中顺序执行，并反复求解重叠路径；fragment re-audit 和
primary certificate 会再次执行完整 schedules。所有中间结果仅在 patch 结束时落盘。

该问题属于执行粒度、重复数值工作和 checkpoint 缺失，不构成对几何 Gate、
node-induced certificate 或无状态 `xyz-only` 表示的反证。

## 3. retry2 不改变的科学语义

retry2 不删除、抽样或放宽任何以下对象：

- node-induced retained edges；
- root paths；
- fundamental cycle basis；
- multipath tree/chord；
- forward/reverse；
- 每方向三次注册扰动 repeat；
- R0、R1、R2；
- fragment re-audit；
- primary certificate；
- MILP strong reference；
- 原 geometry、solver、certificate 和 representation Gate。

## 4. 新执行 contract

每个 audit phase 先封存完整 schedule registry，再按
`sha256(schedule_id) mod shard_count` 分片。每个 shard 使用独立 subprocess，数值线程
固定为 1。全机 audit worker 总预算为 12；四个 patch 同时运行时每 patch 分配三个
shards，一个 patch benchmark 则分配十二个 shards。

每个 shard 必须封存 source/config/input hash、expected schedules、完整
`(schedule_id, direction, repeat_index)` execution keys、Parquet hash 和报告。聚合要求：

1. shard exact set；
2. schedule exact set；
3. 每 schedule 恰好 `2 * repeats_per_direction` 条 execution；
4. 不允许重复 key；
5. source/config/input/artifact hash 全部闭合。

growth 和每个 audit phase 均可恢复。候选 residual、margin、posture cost、condition、
quality 和 bounds 状态作为附加 schema 完整封存，防止 resume 改变 primary unary cost。

## 5. 性能准入

完整 retry2 前，必须在新 clean SHA 上运行 `patch_07` baseline benchmark：

```bash
scripts/pipelines/run_bacra_v14_2r_patch07_shard_benchmark.sh
```

准入要求：

- required audit phases 存在；
- 每个 phase exact-set Gate 通过；
- 每个 phase 注册 12 shards；
- source/config hash 与 benchmark inventory 一致；
- worktree clean；
- `runtime_s / 64 <= 120 s/parent`。

patch 的科学 Gate 不作为性能准入的前提；它仍作为真实科学结果报告。只有性能 Gate
通过后，`run_bacra_v14_2r_stitched_atlas_retry2.sh` 才允许启动。

## 6. GPT-5 Pro 转向条件

若 exact schedule semantics 下的 12-shard benchmark 仍不能满足 120 s/parent，停止
retry2，不直接减少审计量。届时将 schedule/entity/runtime/retry-tier artifacts 交给
GPT-5 Pro，集中判断“全部 edges + fundamental cycle basis + 分层 long-path/
multipath/repeat”能否成为部署证书的充分审计集合。
