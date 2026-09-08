# LCWLD-030 report: Reference Python、fixtures 与 Golden

- 状态：REVIEW（等待主 agent / 用户签署为 DONE）
- 工单：LCWLD-030
- 前置：LCWLD-020（DONE，见 `LCWLD-020-report.md`）
- 日期：2026-08-28

## 修改文件

- `openmm-localcwld/python/localcwld_reference/__init__.py`（新增）
- `openmm-localcwld/python/localcwld_reference/reference.py`（新增，核心交付物）
- `openmm-localcwld/python/tests/test_reference_fixtures.py`（新增，7 个 fixture 的回归测试）
- `openmm-localcwld/tests/fixtures/*.json`（7 个 fixture，全部新增）
- `openmm-localcwld/tests/fixtures/README.md`、`sha256sums.txt`
- `openmm-localcwld/docs/decisions/DEC-002-density-kernel.md`（从 PENDING 更新为"证据已收集"）
- `openmm-localcwld/docs/decisions/DEC-030-two-particle-fixture.md`（新增，手算 fixture 溯源）
- `openmm-localcwld/docs/reports/generate_lcwld_030_fixtures.py`、
  `generate_water_ca_cluster_fixture.py`、`dec002_kernel_comparison.py`、
  `dec002_kernel_comparison_result.json`（生成器/证据脚本，均可重跑）

未修改 `test_lips_vs_pmeV2.6.py` 或任何 DCD/CSV/PNG。

## 实现摘要

`localcwld_reference.py` 是计划第 18/31/32 节数学规格的逐行 Python 移植：
显式 `O(N^2)` 双重循环（不优化，不并行），float64，四阶段结构
（Pass A density → Pass B Q/dQ → Pass C pair energy/direct force/lambda →
Pass D chain force），ZMM closure ell=1/2/3 解析导数，OpenMM 风格的 reduced
triclinic 最小镜像。**不依赖 OpenMM**——只需要 numpy，方便脱离 OpenMM 环境
单独审计/移植到 C++。

## 七个 fixture 全部完成

| Fixture | 覆盖点 |
|---|---|
| `two_particle_directional` | 计划第 46 节手算示例，独立标量算术复核 |
| `three_particle_chain` | 多 pair 同时存在时 lambda 正确累加；有限差分收敛 h=1e-4→1e-6: 3.19e-5→3.18e-7→1.87e-8 |
| `same_residue` | 计划第 49 节：same-residue 只清零 density edge，不清零 Local pair energy |
| `excluded_pair` | 显式 exclusion 让 density/energy/direct/chain 全部精确为 0 |
| `pbc_cross_boundary` | 1nm 盒子跨边界最小镜像 + 沿晶格矢量整体平移不变性 |
| `zmm_orders` | ell=1/2/3 的 U/dU/dr，多个 r 点 + 有限差分 + r→rc 处平滑归零验证 |
| `water_ca_cluster` | 1 Ca2+ + 3 水的合成体系，`localcwld_reference`（解析核）vs 真实
  v2.6 `CustomGBForce`（1024 点 tabulated 核，隔离 force group 静态求值，
  全程无 `Integrator.step()`）交叉验证 |

## 测试证据

```bash
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python3.12 -m pytest \
  openmm-localcwld/python/tests/ -v
```

结果：**22 passed**（15 条 LCWLD-020 的 metadata 测试 + 7 条新的 fixture 回归
测试，0 skip/xfail）。

## DEC-002（density kernel: analytic vs tabulated）现在有真实证据

两组独立证据（细节见 `DEC-002-density-kernel.md`）：

1. 10,000 点纯数值扫描（直接查询真实 OpenMM `Continuous1DFunction`，非猜测插值算法）：`K(r)` 绝对误差 `~2e-6`；`dK/dr` 在 `r≈r_env` 边界处有约 `1e-2`
   量级的 spline ringing 伪影（v2.6 生产环境里真实存在，不是本次测量误差）。
2. 真实 10 原子 water+Ca 体系：能量相对误差 `3.86e-12`（远优于
   `rtol<=1e-9` 门槛），力最大相对误差 `2.33e-7`（略超 `rtol<=1e-7` 门槛，
   约 2.3 倍，可能是边界伪影在多粒子情形下累积所致）。

**结论未由本工单单方面拍板**——`DEC-002` 状态改为"证据已收集，待主 agent
签署"，并明确建议在真实 1CKK 尺度上再跑一次同类对照后再定案。

## 调试过程中的一个重要发现（已修复，是测试脚手架问题，不是插件代码问题）

第一版 `water_ca_cluster` 交叉验证显示能量误差 32%、力误差 140%，一度像是
`localcwld_reference` 有严重 bug。排查约 20-30 分钟后定位：测试脚手架把
Ca 残基命名为 `"CA"`（PDB 风格），但 v2.6 的 `build_phase_cwld_metadata()`
用 `residue_name in ION_RESNAME_ALIASES.values()` 判断离子——检查的是别名
解析**之后**的规范名，不会自动把 `"CA"` 映射成 `"Ca2+"`；这一步依赖调用方
先跑 `normalize_ion_resnames(topology)`（生产管线 `build_1ckk_system()` 确实
这么做了，测试脚手架没有）。结果 Ca 被错误分类进 solute 分支又被
`FAST_ACTIVE_DENSITY` 清零，导致水响应机制"看起来"完全不工作。修复方式：
测试脚手架直接把残基命名为规范名 `"Ca2+"`。修复后两个实现几乎完美吻合
（见上表）。

这也顺带验证了 `openmm_localcwld.metadata` 内部做别名解析（不依赖调用方
预处理 topology）这个设计决策的价值——它在这次事故里本来会更健壮（不受这个
命名疏忽影响），但对照测试时必须让两边站在同一前提下，不能拿"更健壮"当
借口跳过统一前提。

## 未解决问题

- `DEC-002` 需要主 agent 签署最终结论（analytic 还是 `V26Tabulated1024`）。
- `pbc_cross_boundary` 只测了正交盒；triclinic 盒仍是 TODO（计划第 47 节）。
- `same_residue`/`excluded_pair` 的"source weight 减半"等计划第 46 节末尾提到
  的变体仍未数值化（该节原文已标注为定性描述，非本工单范围）。
- `water_ca_cluster` 只是一个合成构型，不是真实 1CKK 局部环境的裁剪片段。

## 是否修改 API/公式/容差

否。
