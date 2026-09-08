# DEC-002: Density kernel — analytic vs. V26Tabulated1024

- 状态：**证据已收集，结论待主 agent 签署**（不是 PENDING 占位，是"有数据，等拍板"）
- 工单：LCWLD-030
- 日期：2026-08-28

## 背景

v2.6 的 `CustomGBForce` 用 1024 点 `Continuous1DFunction` 表示 density kernel
`K(r) = (1-(r/r_env)^2)^2`（`r<r_env`，否则 0）。原生插件第一版最自然的实现
是直接算解析多项式。本文档汇报两组独立证据。

## 证据 1：10,000 点纯数值扫描（`dec002_kernel_comparison.py`）

不猜测 OpenMM `Continuous1DFunction` 用什么插值算法，而是直接问 OpenMM 本身：
用 `CustomCompoundBondForce` 引用与 v2.6 完全相同的 1024 点表，在 10,000 个
`r∈[0,rc]` 上做静态 `Context.getState()` 查询（无 Integrator step），拿到
OpenMM 实际用的 `K(r)` 和 `dK/dr`（力）。

结果（完整 JSON 见 `docs/reports/dec002_kernel_comparison_result.json`）：

| 量 | 最大绝对误差 | 位置 | 最大相对误差（仅在 `>1e-3` 的信息量区间统计，见下） |
|---|---:|---|---:|
| `K(r)` | `2.196e-06` | `r=0.00048 nm` | `2.196e-06` |
| `dK/dr` | `1.102e-02` | `r≈r_env` 附近（spline ringing） | `1.0`（因为解析值在该点严格为 0，属于计划第 36 节"近零值只看绝对误差"的场景） |

**结论**：`K(r)` 本身在整个 `[0,rc]` 上几乎完美吻合（~1e-6 绝对误差）。
`dK/dr` 在 `r` 非常接近 `r_env` 边界处有一个局部"ringing"：解析核在 `r_env`
处值和一阶导数都精确为 0，但二阶导数从有限值跳到 0（不连续），这是标准
natural cubic spline 在这种"局部支撑"函数边界处的已知行为——不是这次测量
的误差，是 v2.6 生产环境里 tabulated kernel **真实存在**的边界伪影，量级约
`1e-2`（绝对）。

## 证据 2：真实 water+Ca 体系交叉验证（`generate_water_ca_cluster_fixture.py`，
`tests/fixtures/water_ca_cluster.json`）

1 个 Ca2+ + 3 个 TIP3P 水（未平衡的合成构型，只做单点能量/力比较），跑
`localcwld_reference`（解析核）与真实 `test_lips_vs_pmeV2.6.py::
setup_cwld_lips_system()`（1024 点 tabulated 核，隔离到独立 force group，
单点 `Context.getState()`，全程无 `Integrator.step()`）：

| 量 | 绝对误差 | 相对误差 | 计划第 9/25 节 CPU double 门槛 |
|---|---:|---:|---:|
| 总能量 | `7.97e-11 kJ/mol` | `3.86e-12` | `rtol<=1e-9` ✅ 远优于门槛 |
| 单原子力（最大） | `6.69e-07 kJ/mol/nm` | `2.33e-07` | `rtol<=1e-7` ⚠️ **略超门槛**（约 2.3 倍） |

## 过程中发现并修复的一个测试脚手架 bug（不是插件/参考实现的 bug）

生成 `water_ca_cluster` fixture 的第一版把离子残基命名为 `"CA"`
（PDB 风格），但 v2.6 的 `build_phase_cwld_metadata()` 用
`residue_name in ION_RESNAME_ALIASES.values()` 判断离子——它检查的是别名
解析**之后**的规范名（`"Ca2+"`），不会自动把 `"CA"` 映射过去；这一步由调用方
先跑 `normalize_ion_resnames(topology)`（生产管线 `build_1ckk_system()` 确实
这么做了，我的测试脚手架没有）。结果是 Ca 在 `setup_cwld_lips_system` 里被
错误分类进 solute/ligand 分支，又被 `FAST_ACTIVE_DENSITY` 默认网关整体清零
（`dens_source=0`），导致水响应机制在两次对照（`enable_water_response=True`
vs `False`）里给出逐位相同的能量——一度看起来像插件/参考实现有严重 bug
（32% 能量误差、140% 力误差），实际上是 Ca 从未真正成为过 density source。
用 20+ 分钟的逐层排查（trivial constant computed value → 手写最小 density_expr
→ 直接 dump 真实 CustomGBForce 的 per-particle 参数）才定位到问题，修好后
（残基直接命名为 `"Ca2+"`）两个实现几乎完美吻合（见证据 2）。这顺带印证了
`openmm_localcwld.metadata` 内部做别名解析而不依赖调用方预处理 topology
（见 `metadata.py` 模块文档）这个设计决策是有实际价值的——它对这一类调用方
疏忽更健壮，但也意味着**它和 v2.6 的行为在"caller 没有 normalize resnames"
这个前提下会不一致**，测试/对照时必须让两边站在同一条件上（都已 normalize），
而不是拿"更健壮"当借口。

## 建议（不是最终决定，需要主 agent 签署）

- 两组证据都表明：解析 kernel 在能量和绝大多数力分量上极其接近 tabulated
  kernel（差 6-12 个数量级），唯一的例外是极小一部分粒子对——那些配对距离
  恰好落在 `r_env` 边界附近几个 Å 以内——会因为 spline ringing 产生量级
  `1e-2`（绝对，dK/dr）的局部偏差，这在本次 10-粒子测试里已经让整体最大力
  相对误差略超过 `rtol<=1e-7` 的门槛（约 2.3 倍）。
- 这不是"analytic kernel 错了"，而是"tabulated kernel 本身在边界处有一个
  真实存在、大小可控的数值伪影"；两种实现哪个该被信任为"正确"取决于产品
  目标：如果目标是跟 v2.6 生产结果逐位复现（哪怕伪影也复现），应该实现
  `V26Tabulated1024` 兼容模式；如果目标是更干净的物理模型（消除已知的
  spline 边界伪影），analytic kernel 反而更可取，但那属于"改进"而不是
  "复现"，需要显式承认这一点。
- 本次证据只覆盖 1 个手工合成的 10 粒子体系和纯 1D 核扫描，还没有在真实
  1CKK 尺度（数万粒子，边界附近的粒子对数量会显著更多）上验证这个边界
  误差累积后是否仍然可忽略。在主 agent 签署此 DEC 之前，建议至少再跑一次
  基于真实 1CKK 构型子集（例如裁剪出 Ca 附近 2-3 nm 的局部环境）的同类
  对照，而不是仅凭这一个合成体系下结论。
- 执行 agent（包括写这份文档的我）无权自行拍板，只能提供证据；请主 agent
  依据以上证据和第 9/25 节门槛表决定 v0.1 默认用哪种 kernel，并把决定写回
  本文件的"状态"字段（改成"冻结：analytic"或"冻结：V26Tabulated1024"）。
