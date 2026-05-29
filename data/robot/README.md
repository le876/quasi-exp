## 需要提供的 CAD 导出参数（CSV）

把 SolidWorks 导出的参数放到 `data/robot/`，并确保单位为 **SI**（m/kg/Pa/rad）。

## SolidWorks 导出建议（对齐论文 DH 口径）

论文的运动学用 DH 参数法，且默认约定包括：
- `{0}`：基坐标系，重力沿 `-y0`
- `{i}`：固定在第 i 个圆盘上，**原点在 Disk_i 与 Disk_{i-1} 的转轴中心**
- 相邻关节盘“转 90° 连接”对应相邻转轴正交（DH 里的 `α` 交替 ±90°）

因此推荐在 SolidWorks 里显式建立这些坐标系后再导出点坐标：
1) 在装配体里建立 `{0}`（Assembly Coordinate System），并确认 `+y0` 竖直向上
2) 在每个关节盘零件里建立 `{i}`（Part Coordinate System），原点放在论文定义的转轴中心，轴向按你的 DH-参数.md 约定
3) 用 `Evaluate -> Measure`（或点坐标导出）时，选择对应坐标系读取 X/Y/Z，并把 mm 转成 m

实现上支持“模板盘”复用：如果 2..30 号盘与 1 号盘完全相同，你只要导出 `disk_idx=1` 的孔位/质心/材料参数，其它盘可以整盘缺失，代码会自动用模板复制。

### 坐标系换算提示（SolidWorks → DH）

如果你用 SolidWorks 的“默认坐标系”导出重心，但它与 DH 坐标系轴向不同，可以先做轴换算再写入 `disks.csv`。

你给出的轴关系示例（S=SolidWorks, D=DH）：
- `z_S = x_D`
- `y_S = -z_D`
- `x_S = -y_D`

则重心坐标分量换算为：
- `x_D = z_S`
- `y_D = -x_S`
- `z_D = -y_S`

单位也要从 mm 转成 m。

### 1) lengths.csv
- 列：`name,value_m`
- 必须包含：`l0,l1,...,l30`

对齐论文定义：
- `l0`：线性模组进给量（若你把 `{0}` 原点放在第 1 个转轴中心，可取 0）
- `l1..l29`：相邻转轴中心距离（你的描述为 41mm，即 0.041m）
- `l30`：最后一个转轴中心到末端点/末端盘中心的“末端长度”（用于末端与弯矩项的长度尺度）

### 2) holes.csv
- 列：`disk_idx,hole_idx,side,x_m,y_m,z_m`
- `disk_idx`: **0..30**（注意：需要包含 `disk_idx=0` 的 `side=dist` 行，用作 `h'_{0,j}`）
- `hole_idx`: 1..12
- `side`: `prox` 表示 `h_{i,j}`，`dist` 表示 `h'_{i,j}`
- 坐标：在各盘局部坐标系 `{i}` 下（`disk_idx=0` 时就是 `{0}`）

实现支持一个便捷模式：如果 `disk_idx>=1` 的某些盘在 `holes.csv` 中**整盘缺失**（prox+dist 都缺），会自动用首个完整盘（通常是 `disk_idx=1`）的孔位作为模板复制过去（与论文“各盘设计参数相同”的描述一致）。如果是“缺一部分孔位”，会直接报错。

### 3) disks.csv
- 列：`disk_idx,mass_kg,com_x_m,com_y_m,com_z_m`
- `disk_idx`: 1..30（盘质量与质心，质心坐标在 `{disk_idx}`）

实现支持一个便捷模式：如果 `disks.csv` 里缺少部分 `disk_idx` 的行（导致这些盘的 `mass_kg=0`），会自动用首个非零盘的质量与质心作为模板复制到缺失盘（与论文“各盘设计参数相同”一致）。如果你需要每盘不同，请完整提供 1..30。

### 4) material.csv
- 列：`disk_idx,E_pa,Iz_m4`
- `disk_idx`: 1..30（每盘/每段参数；若相同也请展开）

同样支持缺失盘自动用模板复制（首个 `E_pa>0` 且 `Iz_m4>0` 的盘）。

注意：
- 这里的 `Iz_m4` 指 **截面二次矩（Area moment of inertia）**，单位是 `m^4`。
- SolidWorks 的 `Mass Properties` 里给的是**质量转动惯量**（`kg·mm^2` / `kg·m^2`），不是 `Iz_m4`。
- 获取 `Iz` 的常用方法：
  - 在零件/装配体上做一个与梁轴向垂直的截面，然后用 `Evaluate -> Section Properties` 读取 `Area Moments of Inertia (Izz)`（单位通常 `mm^4`，写入 CSV 时乘 `1e-12` 变成 `m^4`）
  - 或者直接用圆杆公式：`Iz = π d^4 / 64`

提示：如果你需要把“多个平行杆孔偏置”计入（复合截面 + 平行轴项），常用形式为：
- `Iz = Σ (I_single + A * y_i^2)`（关于 z 轴；其中 `y_i` 为每根杆到参考轴的距离）

但该复合截面口径隐含一个强假设：4 根杆像一个整体梁一样工作，截面保持平面，杆间由刚性连接约束，弯曲时偏心杆产生显著轴向拉压应变，所以平行轴项 `A*y_i^2` 全部生效。当前机器人中 4 根 NiTi 杆之间没有刚性连接约束，弯曲时不会彼此直接产生轴向拉压应变；因此完整复合截面 `Iz` 更适合作为刚性组合梁上界，不应作为当前 rods-only 数据生成口径的默认值。

### 5) end_effector.csv（可选但建议提供）
- 列：`p_end_x_m,p_end_y_m,p_end_z_m`
- 解释：`^{30}p_end` 的 3D 部分（齐次最后一维默认 1）

如果你不提供 `end_effector.csv`，代码会默认 `p_end=[0,0,0]`（即取 `{30}` 原点作为末端盘中心）。

建议验收：
- 导出后先跑：`python3 scripts/validate_robot_inputs.py --config configs/robot.yaml`
- 看到 `PASS` 再开始生成数据集，避免坐标系/单位跑偏

## 可选：用“模板参数”直接生成 CSV（不依赖 SolidWorks 导出）

如果你已经从代码/标定得到孔位与长度模板，可以用：
- `python3 scripts/build_robot_csv_from_templates.py --out-dir data/robot/generated`

它会生成：
- `data/robot/generated/lengths.csv`
- `data/robot/generated/holes.csv`
- `data/robot/generated/end_effector.csv`

并可用以下脚本与 `theta2rope`（C++）做绳长变化交叉验证：
- `python3 scripts/cross_validate_rope_length.py --config <你的config>`
