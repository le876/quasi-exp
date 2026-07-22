# Generalized Ellipse Region V11.3：Canonical Branch-Identity Repair 实验记录

日期：2026-07-22

协议：`generalized-ellipse-region-v11.3-branch-identity`

来源协议：`generalized-ellipse-region-v11.2-relaxed2x`

正式产物：`runs/generalized_ellipse_region_v11_branch_identity/`

最终状态：**实验执行完成，Formal Gate FAIL，Phase 1 重启未获授权，Tube/Student 未启动。**

## 1. 实验问题与结论边界

V11.2 在放宽数值 Gate 后暴露出新的主瓶颈：同一条目标 XYZ 闭环，仅改变 traversal 方向或 cyclic cut，慢速 teacher 就可能输出相差 9--14 deg 的不同 beta 逆解分支。V11.3 不再继续放宽 Gate，而是按下一步计划修复 teacher 的 branch identity，使 traversal/cut 只承担审计功能，不再独立定义标签。

本轮实现并比较五种策略：

| ID | 策略 | 作用 |
|---|---|---|
| BI-0 | 冻结的 V11.2 独立 traversal/cut | 基线与负对照 |
| BI-1 | 共享 canonical root | 判断起点分支是否足够 |
| BI-2 | shared root + reference continuation | 用冻结 reference 锚定局部分支 |
| BI-3 | cyclic candidate graph | 用全环候选图选择连续分支 |
| BI-4 | graph + consensus whole-curve optimization | 对共享 consensus 做整轨迹优化 |

Pilot 使用 A2_178、A2_143 的 180-phase 轨迹；A2_134 只作为冻结的负对照。按冻结排名规则，Pilot 前二为 BI-3、BI-2，二者随后在 A2_178、A2_143 上运行 720-phase Formal。Formal 目录严格只包含这两个预先选定的方法，没有将 BI-4 或其他事后方案混入排名。

所有正式判断恢复严格阈值：residual P95/max 为 1/3 mm，joint margin 为 1.5 deg，repeat P95 为 0.2 deg，traversal/cut P95/max 为 1/2 deg，`gap > 1 deg` 比例不超过 1%，且每个 phase 必须只有一个 branch cluster。禁止删 phase、禁止平均不同分支、禁止在 Gate 失败后进入 Tube。

## 2. 实现范围

新增协议配置 `configs/generalized_ellipse_region_v11_branch_identity.yaml`，新增 branch-identity 模块 `src/quasi_exp/teacher/branch_identity.py`，并新增独立执行器 `scripts/analysis/run_generalized_ellipse_branch_identity_v11_3.py`。最终实现还包含四项会影响结论可信度的约束：

- Formal 的 BI-3 每层实际保留 8--16 个不同候选节点，并使用带加速度代价的二阶 cyclic DP；
- `primary` 与 `repeat` 使用完全相同的 seed、参数和策略独立重跑，repeat 指标直接比较二者；
- audit solver 失败或无法留在 trust radius 内会 fail closed，不能用越界结果伪装成分支一致；
- Formal 只执行 Pilot 冻结选出的 top-2，并从实际轨迹文件读取 720-phase inventory。

执行器将产物拆成以下阶段：

| 阶段 | 主要产物 | 状态 |
|---|---|---|
| `00_protocol` | 解析协议、运行环境、冻结输入哈希 | PASS |
| `00_branch_forensics` | per-phase/per-joint gap、nullspace 分解、cluster、transition 图 | PASS |
| `01_canonical_root` | 64-seed root 候选、0.5 deg 聚类、唯一 canonical root | PASS（阶段完成；选中簇均为 singleton） |
| `02_pilot` | BI-0--BI-4，180 phase，方法排名 | PASS（执行完整；没有方法通过严格策略 Gate） |
| `03_formal` | BI-3/BI-2，2 candidates × 720 phase，最终 Gate | **FAIL** |

`BRANCH_IDENTITY_EXPERIMENT_COMPLETED.json` 只表示本轮 branch-identity 实验完整执行结束，不代表下游协议通过；其中明确记录 `formal_stage_gate_pass=false`、`phase1_restart_authorized=false` 和 `tube_authorized=false`。

## 3. BI-0 冻结数据取证

### 3.1 分支差规模

| 候选 | repeat P95 | 最坏 traversal/cut P95 | 对应 V11.2 最坏值 | 主要关节（P95） |
|---|---:|---:|---:|---|
| A2_134（负对照） | 0.242327 deg | 12.557874 deg | `forward_cut0135` | beta5，26.421249 deg |
| A2_143 | 0.291985 deg | 14.069262 deg | `reverse_cut0000` | beta6，26.889266 deg |
| A2_178 | 1.108191 deg | 9.232195 deg | `forward_cut0045` | beta5，20.840401 deg |

分支转换并非普遍的小抖动。按 1 deg 阈值合并连续区间后，A2_134、A2_143、A2_178 分别存在 11、6、11 个转换区间；其中多个区间持续数百个 phase。

### 3.2 Nullspace 分解

对每个 phase 在 primary beta 处计算 3×6 Jacobian，并把 variant-primary 差分投影为 task-space 与 nullspace 分量。nullspace 能量占比的中位数为：

| 候选 | nullspace 能量占比中位数 | P05 |
|---|---:|---:|
| A2_134 | 99.9212% | 71.8366% |
| A2_143 | 99.9814% | 50.0360% |
| A2_178 | 99.9290% | 83.9846% |

这与 V11.2 的现象一致：不同 beta 分支大多仍能准确命中相同 XYZ，差异主要沿任务冗余的 nullspace 展开。因此问题不是简单的 target/phase 对齐错误，也不能靠继续放宽 FK residual Gate 解决。

## 4. Canonical root 搜索

每个 Pilot 候选使用 64 个确定性 seed，在选定 root phase 求解后按 0.5 deg RMS 聚类，并综合 residual、joint margin、conditioning 和冻结 phase 的 branch separation 选择 canonical root。这里必须区分两个概念：`root_cluster_count` 是 64-seed 解的簇数；`cluster_count` 是冻结 variants 在 root phase 的 branch cluster 数。

| 候选 | root phase | seed 数 | 64-seed 解簇数 | 选中簇支持数 | root-phase variant 簇数 | root residual | root margin | 最小簇间距 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A2_143 | 21 | 64 | 62 | **1** | 3 | 0.000079 mm | 3.363024 deg | 5.433182 deg |
| A2_178 | 62 | 64 | 58 | **1** | 4 | 0.000323 mm | 4.021413 deg | 2.073393 deg |

两个 root 点本身都具有极低 residual 和高于 1.5 deg 的 margin，但不是由 58/62 个 seed 共同支持；58/62 是解簇总数，两个被选中簇都只有一个候选。64-seed 解空间高度碎片化，因此当前 canonical root 是单点评分选出的 singleton，不能视为稳定的多数簇。这个更正直接削弱了“root 已稳健确定”的前提，也解释了后续整环 continuation 的脆弱性。

## 5. 180-phase Pilot

### 5.1 方法排名

| 方法 | 通过候选数 | 两候选累计通过检查数 | 最坏 variant P95 | 最低 margin |
|---|---:|---:|---:|---:|
| BI-3 | 0/2 | **18** | 0.104432 deg | 0 deg |
| BI-2 | 0/2 | **16** | 0.104432 deg | 0 deg |
| BI-4 | 0/2 | 15 | 0.059234 deg | 0 deg |
| BI-1 | 0/2 | 10 | 12.944281 deg | 0 deg |

BI-3 和 BI-2 按冻结的排名规则进入 Formal。这里的排名是检查项累计分数，不是 Gate 通过：四种方法均为 0/2。BI-1 表明只共享 root 不足以阻止随后 continuation 重新选支；BI-3 的 8--11 节点候选层和二阶图选择已经能稳定 traversal/cut identity，但仍出现 solver/trust、residual、margin 与 smoothness 失败。

### 5.2 Pilot 的关键轨迹

| 候选/方法 | 通过检查 | residual P95 / max | repeat P95 | variant P95 / max | `>1 deg` 比例 | cluster 数 |
|---|---:|---:|---:|---:|---:|---:|
| A2_178 / BI-3 | 8/15 | 1.452456 / 5.267335 mm | **0** | 0.104432 / 0.242689 deg | 0% | 1 |
| A2_178 / BI-2 | 8/15 | 8.281688 / 17.860498 mm | **0** | 0.104432 / 0.306458 deg | 0% | 1 |
| A2_178 / BI-4 | 8/15 | 8.281665 / 12.057780 mm | **0** | 0.033550 / 0.224743 deg | 0% | 1 |
| A2_143 / BI-3 | 10/15 | 0.270238 / 1.663067 mm | **0** | 0.103531 / 0.293727 deg | 0% | 1 |
| A2_143 / BI-2 | 8/15 | 5.296902 / 12.964893 mm | **0** | 0.039032 / 0.158736 deg | 0% | 1 |
| A2_143 / BI-4 | 7/15 | 3.360120 / 8.723153 mm | **0** | 0.059234 / 0.353316 deg | 0% | 1 |

`repeat P95=0` 是修正后的真实同 seed/同参数独立重跑结果，而不是把某个 traversal variant 误当 repeat。Pilot 的 branch identity 已明显收敛，但所有候选/方法的最低 joint margin 都为 0 deg，所以没有方法通过策略 Gate。

## 6. 720-phase Formal

### 6.1 预先选定的 top-2 结果

| 候选/方法 | residual P95 / max | margin | repeat P95 | variant P95 / max | `>1 deg` 比例 | cluster 数 | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| A2_178 / BI-2 | 15.683221 / 18.887969 mm | 0 deg | **0** | 0.004823 / 0.378088 deg | 0% | 1 | FAIL（9/15） |
| A2_178 / BI-3 | 2.169650 / 8.988982 mm | 0 deg | **0** | 0.035090 / 0.476667 deg | 0% | 1 | FAIL（9/15） |
| A2_143 / BI-2 | 4.849352 / 8.237505 mm | 0 deg | **0** | 0.004182 / 0.602651 deg | 0% | 2 | FAIL（8/15） |
| A2_143 / BI-3 | 1.492651 / 4.823111 mm | 0 deg | **0** | 0.075522 / 0.664155 deg | 0% | 2 | FAIL（7/15） |

四个正式组合共同失败于 `audit_variant_solver_success`、`audit_variant_trust_preserved`、primary residual P95/max、strict joint margin 与 trajectory velocity max。A2_143 的两种方法还因存在 2 个 branch cluster 而失败；A2_143/BI-3 的 acceleration P95 为 0.770897 deg，也超过 0.5 deg。A2_178 两种方法均达到单簇，A2_143 则尚未达到。

BI-3 图求解本身成功，Formal 候选层为 8--9 个不同节点并启用了二阶 acceleration DP；A2_143、A2_178 的图总代价分别为 838.439 和 6346.609。失败发生在图所选择的 chart 不能同时维持严格 numerical/trust 条件，而不是候选层不足或图算法没有运行。

### 6.2 相对 BI-0 的改善

Branch repair 确实产生了强烈、可重复的改善：

- A2_178 的最坏 traversal/cut P95 从 9.232195 deg 降到 BI-2 的 0.004823 deg（约 1914 倍）或 BI-3 的 0.035090 deg（约 263 倍）；
- A2_143 的最坏 traversal/cut P95 从 14.069262 deg 降到 BI-2 的 0.004182 deg（约 3364 倍）或 BI-3 的 0.075522 deg（约 186 倍）；
- 四个 Formal 组合的 repeat P95 都为 0，variant max 都低于 0.67 deg，`gap > 1 deg` 比例全部为 0。

因此原来的 9--14 deg traversal/cut 分支漂移已经被实质消除，得到了用户希望看到的“方法继续走下去”的稳定跟踪轨迹。但这仍不是 Formal PASS：稳定地留在同一参考 branch，并不意味着该 branch 在每个 phase 都存在 trust 内、低 residual、留足 joint margin 的数值解。Gate 正确地阻止了“身份一致但几何跟踪不可用”的轨迹进入数据生成。

## 7. 失败机制解释

以下判断由本轮证据直接支持：

1. **共享 root 是必要但不充分条件。** BI-1 的 Pilot P95 仍在约 13 deg，说明后续 continuation 可以离开共享起点所定义的 branch。
2. **reference continuation 已解决 branch identity，但没有解决可行性。** BI-2 的 Formal variant P95 低于 0.005 deg、max 低于 0.61 deg，却仍有 4.85--15.68 mm residual P95，且 solver/trust 检查失败。当前 fail-closed 行为冻结 reference，得到的是一致但不准确的轨迹，而不是越界“修好”的轨迹。
3. **候选图明显改善 tracking residual，但仍达不到严格阈值。** BI-3 相比 BI-2 将 A2_143 residual P95 从 4.849 降到 1.493 mm、A2_178 从 15.683 降到 2.170 mm；它保留了 branch identity，却依然无法满足 1/3 mm residual、1.5 deg margin 和 trust-preservation。
4. **当前 canonical root 目标缺少整环可行性与统计支撑。** 被选中的 root 簇均为 singleton，root 点 margin 虽为 3.36/4.02 deg，正式轨迹最低 margin 却都降到 0 deg。单点评分最优不能保证所属 branch chart 覆盖完整目标闭环。

基于这些结果，最合理的下一层假设是：root cluster 需要通过“短 continuation 或整环可行性”准入，而不是只按单点 canonical cost 排名；同时图边/整轨迹优化必须把严格 tracking 和 margin 作为可行约束，而不是在 reference、smoothness 与 tracking 之间做可越界的软权衡。这是对现有证据的推断，尚未通过下一轮实验验证。

本轮也尚不足以断言静态 `xyz -> beta` 从根本上不成立。相反，BI-2/BI-3 已使 traversal/cut gap 全部低于 1 deg；当前证据更接近“选中的 canonical chart 不能在严格数值条件下覆盖整个闭环”，而不是“所有低 residual 连续解都不可避免地长期分裂成多个 5--15 deg 分支”。

## 8. Gate 与产物完整性审计

最终封存序列于 14:10:36 启动，15:09:25 完成 Pilot；包含最终代码 provenance 的协议于 15:13:06 重新封存，17:36:57 写入 Formal Gate 与 completion。全序列总耗时约 3 小时 26 分钟，其中 Formal 约 2 小时 24 分钟。runtime git SHA 为 `fb9f5f8708b575022639ab42293bd6203de5b343`。正式目录含 314 个文件、约 48 MiB。

独立审计结果：

- 递归检查 24 个 `gate.json`；
- 重算并匹配 308 个 Gate 声明的产物 SHA-256；
- 重算并匹配 31 个冻结输入 SHA-256；
- 检查 195 个 Pilot/Formal centerline 或 consensus 文件，Pilot 全部为 180 phase，Formal 全部为 720 phase；
- 两个候选的 Formal 方法目录都严格等于 `BI-2`、`BI-3`，不存在 BI-4 Formal 目录；
- V11.2 Phase 1 源 Gate 仍为 `false`，其 SHA-256 与 V11.3 manifest 记录一致；
- 不存在 Tube、Student 或下游 `COMPLETED` 产物。

因此本轮是“实验完整执行、科学 Gate 失败”，不是运行中断、phase 缺失或产物损坏。

## 9. 下一步边界

不得启动 Tube/Student，也不应继续放宽 numerical Gate。下一轮若继续，应建立独立的 V11.4 协议，并至少完成：

1. 对 64-seed root cluster 增加短弧/整环 continuation feasibility admission；
2. 把 residual 与 1.5 deg margin 变为 candidate graph 的硬可行边界，先淘汰不可持续 cluster；
3. 对发生 max-gap 的孤立 phase 输出前后候选节点、边代价、active constraint 和 cluster 转移证据；
4. 若当前 center/plane 上不存在覆盖全环的严格可行 chart，再按原计划小范围 re-search center/plane/root；
5. 只有某个 candidate 同时通过 numerical、margin、repeat、all cuts/directions、smoothness、closure 和 single-cluster Gate，才允许重新运行正式 Phase 1。

## 10. 复现命令

```bash
PYTHONPATH=src \
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python3.11 \
scripts/analysis/run_generalized_ellipse_branch_identity_v11_3.py \
  --config configs/generalized_ellipse_region_v11_branch_identity.yaml \
  --preset formal \
  --stage all
```

正式汇总入口：

- `runs/generalized_ellipse_region_v11_branch_identity/03_formal/formal_ranking.csv`
- `runs/generalized_ellipse_region_v11_branch_identity/03_formal/gate.json`
- `runs/generalized_ellipse_region_v11_branch_identity/BRANCH_IDENTITY_EXPERIMENT_COMPLETED.json`
