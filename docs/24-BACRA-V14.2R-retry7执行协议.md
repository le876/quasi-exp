# BACRA V14.2R retry7 执行协议

## local-abstention retry1 修订

首次 retry7 已封存并证明 K/R 对照稳定，但没有已注册 kernel 单独通过
critical-cycle Gate。该结果不授权放宽 `geometry max <= 1 deg`。后续采用本
协议原有的最小局部 abstention 后备路线：

1. repeat 只比较到达同一闭环物理端点且整条 trace 成功的执行；失败或截断
   trace 单独计入 `incomplete_trace_count`，并继续阻断 solver Gate；
2. 只有 solver 全完成、repeat/FK 通过、`>1 deg` 几何超标集中在至多一条物理
   转移边、且最大 gap 不超过 `1.10 deg` 的最便宜 kernel，才可作为
   `localized_abstention` 候选；这不等价于 critical-cycle pass；
3. 候选仍须 fresh guard-set 通过，随后在 patch_07 上显式移除失败端点，重建
   node-induced graph、完整 cycle basis 和 fresh primary certificate；
4. retained coverage 必须 `>=0.90`、largest coherent region 必须 `>=0.60`，
   且 geometry/solver/repeat/certificate/edge/cycle Gate 全部通过；否则科学停止；
5. 已封存 retry7 的 lineage、K/R、holonomy 和 raw critical traces仅作为带 SHA
   的只读证据引用，不复制成 fresh execution；guard、patch repair 及所有下游
   certificate 必须重新执行。

修订输出使用独立目录
`runs/bacra_v14_2r_stitched_atlas_retry7_abstention_retry1`，不得覆盖首次
retry7 artifacts。

本协议冻结 GPT-5 Pro 对 retry6 科学 Gate 失败的审计结论，并只对后续结果生效。retry6 artifacts 保持只读，不重写其结论。

## 科学假设

retry7 分别检验：root budget 是否导致 canonical component switch；固定 canonical anchor 后 K/R 是否稳定；唯一 refined cycle 的失败属于 static section 还是 free transport holonomy；显式 null-space gauge 是否可在不破坏 guard set 的情况下修复该反例。

## 固定 Gate

- FK residual max：3 mm；
- geometry P95/max：0.5°/1°；
- repeat P95：0.2°；
- retained unresolved entity：0；
- patch coverage/coherent measure：0.90/0.60；
- K/R stability：coverage Jaccard 0.95，beta P95/max 1°/2°；
- 四 diagnostic patches：4/4；
- development/confirmation：6/8 与 3/4。

## 漏斗顺序

1. inventory 与 retry6 lineage 的无 solver 审计；
2. patch_07 的 K1/K4 × R5/R8 严格嵌套对照；
3. 唯一 failed cycle 的 prefix、canonical-reset、all-start 和 refinement-edge ablation；
4. C0→C4 的定向 gauge kernel 微实验；
5. 重新生成 patch_07 atlas；
6. 四 patch 4/4 Gate；
7. Reach Round 8；
8. fresh 12-patch confirmation；
9. 一个 512-cell meso bridge；
10. 只有上述 Gate 通过才授权 repaired 5k。

任何扩大阶段都不得在其前置 Gate 失败后启动。scientific failure 必须完整产出 summary，但不等同 operational failure。

## Canonical identity

R8 root registry 构造一次，R5 是其冻结前缀。anchor root 始终位于第一优先级；ordering/dropout 只能改变 non-anchor roots。只要第一 anchor component 满足 coverage、coherent measure 和 retained certificate，就不得被覆盖更高但不兼容的 component 替代。

## Gauge 与审计

gauge 只可使用 source beta、predictor、frozen anchor posture 和已接受父节点，不得使用 target canonical beta。static canonical-reset edge Gate 与 free-transport holonomy Gate 分开报告。物理实体和 repeat perturbation 由规范化 physical path 决定，不依赖 chart/schedule ID。

## 计算与复现

全机最多 12 个单线程 numerical workers。critical diagnostics 少于 9,500 execution rows。K/R 对照先做 growth/lightweight audit，物理 field 与完整 certificate closure 相同时才允许内容寻址复用。所有扩大实验从单一 clean scientific SHA 运行，并绑定 source/config/runtime/upstream artifact closure。
