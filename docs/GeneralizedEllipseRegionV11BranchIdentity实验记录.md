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

Pilot 使用 A2_178、A2_143 的 180-phase 轨迹；A2_134 只作为冻结的负对照。Pilot 排名前二的 BI-4、BI-2 再在 A2_178、A2_143 上运行 720-phase Formal。Formal 阶段额外生成 BI-3 图分支，作为 BI-4 的显式依赖证据，但不把它混入预先选定的 Formal top-2 排名。

所有正式判断恢复严格阈值：residual P95/max 为 1/3 mm，joint margin 为 1.5 deg，repeat P95 为 0.2 deg，traversal/cut P95/max 为 1/2 deg，`gap > 1 deg` 比例不超过 1%，且每个 phase 必须只有一个 branch cluster。禁止删 phase、禁止平均不同分支、禁止在 Gate 失败后进入 Tube。

## 2. 实现范围

新增协议配置 `configs/generalized_ellipse_region_v11_branch_identity.yaml`，新增 branch-identity 模块 `src/quasi_exp/teacher/branch_identity.py`，并新增独立执行器 `scripts/analysis/run_generalized_ellipse_branch_identity_v11_3.py`。执行器将产物拆成以下阶段：

| 阶段 | 主要产物 | 状态 |
|---|---|---|
| `00_protocol` | 解析协议、运行环境、冻结输入哈希 | PASS |
| `00_branch_forensics` | per-phase/per-joint gap、nullspace 分解、cluster、transition 图 | PASS |
| `01_canonical_root` | 64-seed root 候选、0.5 deg 聚类、唯一 canonical root | PASS |
| `02_pilot` | BI-0--BI-4，180 phase，方法排名 | PASS（执行完整；没有方法通过严格策略 Gate） |
| `03_formal` | Pilot top-2，2 candidates × 720 phase，最终 Gate | **FAIL** |

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

每个 Pilot 候选使用 64 个确定性 seed，在选定 root phase 求解后按 0.5 deg RMS 聚类，并综合 residual、joint margin、conditioning 和冻结 phase 的 branch separation 选择 canonical root。

| 候选 | root phase | seed 数 | cluster 数 | 选中 cluster 支持数 | root residual | root margin | 最小 cluster separation |
|---|---:|---:|---:|---:|---:|---:|---:|
| A2_143 | 21 | 64 | 3 | 62 | 0.000079 mm | 3.363024 deg | 5.433182 deg |
| A2_178 | 62 | 64 | 4 | 58 | 0.000323 mm | 4.021413 deg | 2.073393 deg |

两个 root 点本身都具有极低 residual 和高于 1.5 deg 的 margin，并由绝大多数 seed 支持；因此 root 搜索不是随机挑选孤立小簇。但后续结果说明，“单点 canonical cost 最优”不等价于“该 cluster 可以在严格 residual 与 margin 下沿完整闭环持续”。

## 5. 180-phase Pilot

### 5.1 方法排名

| 方法 | 通过候选数 | 两候选累计通过检查数 | 最坏 variant P95 | 最低 margin |
|---|---:|---:|---:|---:|
| BI-4 | 0/2 | 10 | 0.073303 deg | 0 deg |
| BI-2 | 0/2 | 8 | 0.141562 deg | 0 deg |
| BI-3 | 0/2 | 7 | 2.918653 deg | 0 deg |
| BI-1 | 0/2 | 4 | 12.944281 deg | 0 deg |

BI-4 和 BI-2 按冻结的排名规则进入 Formal。BI-1 表明只共享 root 不足以阻止随后 continuation 重新选支；BI-3 的图选择改善了一部分连续性，但独立图审计仍不能稳定维持 repeat/cut identity。BI-4、BI-2 已把 P95 分支差压到 0.15 deg 以下，但同时出现 residual 变差和 joint margin 触界。

### 5.2 Pilot 的关键轨迹

| 候选/方法 | residual P95 / max | repeat P95 | variant P95 / max | `>1 deg` 比例 | cluster 数 |
|---|---:|---:|---:|---:|---:|
| A2_178 / BI-0 | 0.0073 / 0.2381 mm | 1.079478 deg | 9.238307 / 11.144036 deg | 91.67% | 6 |
| A2_178 / BI-2 | 8.2817 / 17.8605 mm | 0.141562 deg | 0.141562 / 2.696831 deg | 1.11% | 2 |
| A2_178 / BI-4 | 8.2114 / 10.2658 mm | 0.013030 deg | 0.042371 / 0.837315 deg | 0% | 2 |
| A2_143 / BI-0 | 0.0065 / 0.0264 mm | 0.287263 deg | 14.051575 / 14.277953 deg | 94.44% | 5 |
| A2_143 / BI-2 | 5.2969 / 12.9649 mm | 0.110333 deg | 0.110333 / 2.638179 deg | 1.67% | 2 |
| A2_143 / BI-4 | 3.4859 / 6.2870 mm | 0.036072 deg | 0.073303 / 2.752877 deg | 0.56% | 2 |

## 6. 720-phase Formal

### 6.1 预先选定的 top-2 结果

| 候选/方法 | residual P95 / max | margin | repeat P95 | variant P95 / max | `>1 deg` 比例 | cluster 数 | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| A2_178 / BI-4 | 6.246026 / 11.869901 mm | 0 deg | 0.002042 deg | 0.002510 / 1.649237 deg | 0.1389% | 2 | FAIL |
| A2_143 / BI-2 | 4.849352 / 8.237505 mm | 0 deg | 0.006868 deg | 0.006868 / 2.459792 deg | 0.5556% | 2 | FAIL |
| A2_178 / BI-2 | 15.683221 / 18.887969 mm | 0 deg | 0.000003 deg | 0.008528 / 5.410706 deg | 0.5556% | 2 | FAIL |
| A2_143 / BI-4 | 3.441887 / 10.121850 mm | 0 deg | 0.052361 deg | 0.052361 / 4.094633 deg | 0.2778% | 3 | FAIL |

四个正式组合均只通过 12 项严格检查中的 6 项。共同失败项为：

- primary residual P95 与 max；
- strict joint margin；
- trajectory velocity max；
- 单一 branch cluster。

A2_143/BI-2、A2_178/BI-2、A2_143/BI-4 还因孤立 phase 的 traversal/cut max 超过 2 deg 而失败。A2_178/BI-4 的 max 已降至 1.649237 deg，但其 acceleration P95 为 0.632628 deg，超过 0.5 deg；它仍有 2 个 cluster，且 residual/margin 不合格。

### 6.2 BI-3 依赖证据

BI-4 在 Formal 中需要生成同一 720-phase candidate graph，因此 BI-3 作为依赖证据写入各候选目录，但不改变 Pilot 选出的 Formal 方法集合。独立 BI-3 在 A2_143、A2_178 上分别只通过 3/12 项检查，variant P95 分别为 1.797667 和 4.669990 deg，不构成隐藏的通过方案。

### 6.3 相对 BI-0 的改善

Branch repair 确实产生了强烈、可重复的改善：

- A2_178 的最坏 traversal/cut P95 从 9.232195 deg 降到 BI-4 的 0.002510 deg，约缩小 3678 倍；
- A2_143 的最坏 traversal/cut P95 从 14.069262 deg 降到 BI-4 的 0.052361 deg，约缩小 269 倍；
- 四个 Formal top-2 组合的 `gap > 1 deg` 比例都已低于 1%。

但这不是 Formal PASS：P95 的大幅改善掩盖不了少量 branch switch、多个 cluster、margin 触界以及 3.44--15.68 mm 的 primary residual P95。严格 Gate 正确地阻止了“平均表现很好、局部仍跳支”的轨迹进入数据生成。

## 7. 失败机制解释

以下判断由本轮证据直接支持：

1. **共享 root 是必要但不充分条件。** BI-1 的 P95 仍在约 13 deg，说明后续 continuation 可以离开共享起点所定义的 branch。
2. **reference continuation 能稳定大多数 phase。** BI-2 的 Formal P95 均低于 0.01 deg，但 max 仍为 2.46--5.41 deg，说明少数 phase 会跨 cluster。
3. **whole-curve consensus 能进一步压低总体 branch gap。** BI-4 在两候选上都把 P95 控制在 0.053 deg 内，但未同时保持 tracking 与严格 margin。
4. **当前 canonical root 目标缺少整环可行性。** root 点的 margin 为 3.36/4.02 deg，而正式轨迹最低 margin 都降到 0 deg。单点可行且高质量，并不能保证所属 branch chart 覆盖完整目标闭环。

基于这些结果，最合理的下一层假设是：root cluster 需要通过“短 continuation 或整环可行性”准入，而不是只按单点 canonical cost 排名；同时图边/整轨迹优化必须把严格 tracking 和 margin 作为可行约束，而不是在 reference、smoothness 与 tracking 之间做可越界的软权衡。这是对现有证据的推断，尚未通过下一轮实验验证。

本轮也尚不足以断言静态 `xyz -> beta` 从根本上不成立。相反，BI-2/BI-4 已使绝大多数 phase cut-independent；当前证据更接近“选中的 canonical chart 不能在严格数值条件下覆盖整个闭环，并在少量 phase 发生换支”，而不是“所有低 residual 连续解都不可避免地长期分裂成多个 5--15 deg 分支”。

## 8. Gate 与产物完整性审计

正式进程退出码为 0，09:58:15 写入协议 Gate，10:47:55 完成 Pilot，13:21:33 写入 Formal Gate，总耗时约 3 小时 23 分钟。正式目录含 324 个文件、约 53 MiB。

独立审计结果：

- 递归检查 26 个 `gate.json`；
- 重算并匹配 318 个 Gate 声明的产物 SHA-256；
- 重算并匹配 31 个冻结输入 SHA-256；
- 检查 189 个 Pilot/Formal centerline 或 consensus 文件，Pilot 全部为 180 phase，Formal 全部为 720 phase；
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
