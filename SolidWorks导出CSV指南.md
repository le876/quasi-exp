# SolidWorks → 本项目 CSV（lengths/holes/disks/material）导出指南（对齐 DH）

本项目使用论文的 DH 参数法来做运动学/绳长/准静态求解，因此 **SolidWorks 中测到的几何量必须落在与 DH `{i}` 一致的坐标系里**，否则会出现“整体符号反、孔位镜像、绳长趋势相反”等问题。

下面给出一套“可人工操作 + 可验收”的导出流程；导出后用本仓库脚本交叉验证，确保没跑偏。

---

## 0. 先统一约定（强烈建议）

### 0.1 给每个圆盘建立坐标系（Coordinate System）
在 SolidWorks 每个圆盘零件（或装配体中该零件）里建立坐标系 `CSYS_DH_i`：
- 原点：论文定义的关节转轴中心（`{i}` 原点）
- `x_i`：沿圆盘“轴向/杆长方向”，指向下一个圆盘（与论文 `l_i` 同方向）
- `z_i`：该关节转轴方向（DH 的旋转轴）
- `y_i`：右手系补齐

> 你的模型存在“相邻圆盘旋转 90° 连接”的结构：这意味着 `CSYS_DH_(i+1)` 相对 `CSYS_DH_i` 会有固定的 90° 旋转（在 `DH-参数.md` 里体现为奇/偶盘差异）。

### 0.2 单位
本项目 CSV 统一使用：
- 长度：米（m）
- 张力：牛（N）
- 弹性模量：帕（Pa）
- 截面二次矩：米四次方（m⁴）

---

## 1. 导出 `lengths.csv`

### 需要什么
`l0..l_kD`（本项目示例为 `l0..l30`），含义是 **相邻 DH 坐标系原点之间在 `x_i` 方向的轴向距离**。

### SolidWorks 如何测
对每个 `i`：
1) 在 `CSYS_DH_i` 与 `CSYS_DH_(i+1)` 的原点处各放一个参考点（或直接用坐标系原点）
2) `Evaluate → Measure` 测两点距离，并确认该距离沿 `x_i` 方向（如果不是纯轴向，说明坐标系或原点没对齐转轴中心）

### 写入 CSV
形如：
```
name,value_m
l0,0.0
l1,0.04
...
```

### 验收
- 直线姿态（所有 `θ=0`）时末端 `x` 应接近 `sum(l_i)`（本仓库有检查项，见 `复现执行计划.md`）。

---

## 2. 导出 `holes.csv`（最关键）

### 需要什么
每个圆盘、每个绳孔，需要 **孔在 DH 圆盘坐标系下的 3D 坐标**：
- `disk_idx`：圆盘编号（本仓库：`0` 为基座盘；`1..kD` 为关节盘）
- `hole_idx`：绳编号（本仓库约定 `1..12`）
- `side`：`prox`（入孔/近端）或 `dist`（出孔/远端）
- `(x_m, y_m, z_m)`：在该圆盘 DH 坐标系 `{disk_idx}` 下

### SolidWorks 如何测（人工）
对某个圆盘 `i`：
1) 激活该圆盘的 `CSYS_DH_i` 为“输出坐标系”（Measure 面板里可选）
2) 对每个孔建立两个参考点：
   - 入孔（prox）：绳进入圆盘的孔口中心点
   - 出孔（dist）：绳离开圆盘的孔口中心点
3) `Evaluate → Measure` 读取参考点在 `CSYS_DH_i` 下的坐标

### 注意：入/出孔的 y,z 修正口径
你已确认的修正规则（用于抑制实测误差）：
- 同一个孔（例如左上孔）的出/入只允许 **x 方向发生位移**；
- 因此 `dist_yz` 强制等于 `prox_yz`（仅 `x` 不同）。

### 验收（强烈建议）
导出完成后执行交叉验证，让几何/孔位与 `theta2rope` 趋势一致：
- `python3 scripts/cross_validate_rope_length.py --config configs/robot_generated.yaml --trials 20 --scale 0.2`
期望：
- `mean(corr) >= 0.95`
- `mean(MAE_mm) <= 2.0`

---

## 3. 导出 `disks.csv`（质量与质心）

### 需要什么
每个圆盘的：
- `mass_kg`
- `com_x_m, com_y_m, com_z_m`（在对应圆盘的 DH 坐标系下）

### SolidWorks 如何测
1) 打开圆盘零件
2) `Evaluate → Mass Properties`
3) 在输出坐标系里选择 `CSYS_DH_i`（或先在该坐标系下读取）
4) 记录质量与质心坐标（mm → m）

### 常见坑：坐标轴不一致
你提供过一组坐标轴换算（SolidWorks 报告坐标系 S 与 DH）：
- `z_S = x_DH`
- `y_S = -z_DH`
- `x_S = -y_DH`

如果你不能在 Mass Properties 里直接选 `CSYS_DH_i`，就必须用上述映射把 `(x_S,y_S,z_S)` 转成 `(x_DH,y_DH,z_DH)` 再写入 CSV。

### 验收
- 所有盘质量应为正且与 SolidWorks 报告一致（允许少量四舍五入误差）
- 同型号普通圆盘的质心应一致（你的交叉信息：除基座/末端外普通盘相同）

---

## 4. 导出 `material.csv`（E 与 Iz）

### 4.1 E（弹性模量）怎么得到
`E` 不是几何量，CAD 不会“算出来”，只能来自材料属性（或厂家数据表）。

SolidWorks 操作：
1) 打开 NiTi 杆零件
2) `Material → Edit Material...`
3) 读取 `Elastic Modulus`（单位通常 `N/mm^2 = MPa`）
4) 写入：`E_pa = E_(N/mm^2) * 1e6`

> 没有 NiTi 材料库时：新建自定义材料并填写 E。

### 4.2 Iz（截面二次矩）怎么得到
本项目 `Iz` 指 **弯曲的截面二次矩（Area moment of inertia，单位 m⁴）**，不是 Mass Properties 里那种质量转动惯量（kg·mm²）。

两条路径：

#### 路径 A：SolidWorks 截面属性（更“几何真实”）
1) 在装配体里建立一个垂直于 `x_DH` 的截面平面
2) `Evaluate → Section Properties`
3) 读取 `Area Moments of Inertia` 的 `Iyy/Izz`（mm⁴）
4) 转单位：`Iz_m4 = Iz_mm4 * 1e-12`

#### 路径 B：解析公式（更可控，推荐用于 rods-only 口径）
单根实心圆杆（直径 `d`）：
- `I_single = π d^4 / 64`

你当前采用的 rods-only 口径：
- 4 根杆的支撑弯矩直接相加：`Iz = 4 * I_single`
- 例如 `d=1.6mm`：`Iz ≈ 1.286796351e-12 m^4`

> 说明：如果你改用“复合截面 + 平行轴项”，就需要选定中性轴并用 `I = Σ (I_single + A*d^2)`；但你已决定使用 rods-only（不平移到中性轴），以贴合当前实验口径。
>
> 关键澄清：复合截面 + 平行轴项隐含“4 根杆像一个整体梁一样工作，截面保持平面，杆间由刚性连接约束，弯曲时偏心杆产生显著轴向拉压应变，所以平行轴项 `A*y^2` 全部生效”。该假设不适用于当前结构：4 根 NiTi 杆之间没有刚性连接约束，弯曲时不会彼此直接产生轴向拉压应变。因此完整复合截面 `Iz` 只能视为刚性组合梁上界，不应作为当前数据生成的默认 `Iz` 口径。

### 验收
用小规模诊断检查“可达性”：
- `python3 scripts/diagnose_solver.py --config <你的config> --samples 50 --scale 1.0`
期望：
- `rms_rnorm.p90 <= 0.06`
- `max_tension.p90 < 2000`（不要长期打满）

---

## 5. 导出后的一键验收（推荐顺序）
1) `python3 scripts/validate_robot_inputs.py --config configs/robot_generated.yaml`
2) `python3 scripts/cross_validate_rope_length.py --config configs/robot_generated.yaml --trials 20 --scale 0.2`
3) `python3 scripts/diagnose_solver.py --config configs/robot_rods_only_20deg_penalty_2k.yaml --samples 50 --scale 1.0`
