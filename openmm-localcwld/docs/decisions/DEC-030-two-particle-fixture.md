# DEC-030 (supporting note): two_particle_directional fixture provenance

- 状态：信息性记录（不是需要签署的正式决策），支持 `tests/fixtures/two_particle_directional.json`
- 工单：LCWLD-030（fixture 设计部分，本次先只交付这一个手算 fixture）
- 日期：2026-08-28

## 做了什么

把计划第 46 节"手算双粒子 Golden 示例"里的公式，用不依赖 OpenMM 的纯 Python 标量算术
（`tanh`/`pow`，见下方命令）独立重算了一遍，跟计划里发布的数值逐位比对：

```bash
python3 - <<'PY'
# 见对话记录 / 可重新粘贴复现：K(r)、density、tanh、Q、dQ/ddensity、U2、dU2/dr、
# pair energy、direct/chain dE/dr、finite-difference 收敛。
PY
```

结果：`K(r)`、`density[0]`、`tanh(kD/rho0)`、`y`、`Q0`、`Q1`、`dQ0/dDensity0`、
`U2(r)`、`dU2/dr`、`pair energy`、`dE/dr direct`、`dE/dQ0`(lambda0)、`K'(r)`、
`dE/dr chain`、`dE/dr total`、四个步长的中心差分收敛值，全部与计划文档给出的数值
在 float64 精度下一致（有限差分的符号约定是 `-(E(r+h)-E(r-h))/(2h)`，是"沿
`unit_0_to_1` 方向作用在粒子 0 上的力"的符号，数值大小与 `dE/dr total` 一致，
这一点和计划原文的符号说明一致）。额外算出了计划原文没有写出的 `lambda1
= dE/dQ1 = C*Q0*U2(r) = -389.13908009514734`，一并写入 fixture。

这不是"跑 MD"或"跑 CustomGB 对照"——只是标量公式的独立复核，因此符合
"assistant 只写/修脚本，不在本项目里执行 MD/analysis"的边界。

## 没做什么 / 明确的缺口

- **LCWLD-030 要求的另外六个 fixture**（`three_particle_chain`、
  `same_residue`、`excluded_pair`、`pbc_cross_boundary`、`water_ca_cluster`、
  `zmm_orders`）尚未生成。它们依赖真正的 `localcwld_reference` Python 参考实现
  跑显式循环，以及用 v2.6 的 `CustomGBForce` 交叉核对——这两者都需要一个装了
  OpenMM 的环境，本 sandbox 没有（见 `DEC-001-toolchain.md`）。
- 计划第 46 节末尾要求的四个变体（same-residue、explicit exclusion、
  source weight 减半、sink 侧 source weight 应无关）目前只在 fixture 的
  `variants_required_by_plan_section_46` 字段里做了**定性**描述，没有重新
  手算出数值，已在 fixture 里明确标成 `TODO`，不得被下一个 agent 误当作
  "已经算好可以直接当 golden 用"。
- `DEC-002`（analytic vs tabulated density kernel）仍然是 PENDING，本 fixture
  用的是解析 `K(r)`，还没跟 v2.6 的 1024 点 `Continuous1DFunction` 对照过。

## 下一步（需要用户在装有 OpenMM 的环境执行）

1. 用 `openmm_localcwld.metadata.build_particle_parameters()` 在几个小体系
   （纯水、水+离子、ASP/LYS 残基、脂质头基）上跑一遍，和
   `test_lips_vs_pmeV2.6.py:build_phase_cwld_metadata()` 的九个数组逐元素对比。
2. 用对比通过后的 metadata，写 `localcwld_reference.py`（float64、显式循环、
   不调用任何待开发的 C++ 插件），生成剩下六个 fixture 并接入 `CustomGBForce`
   交叉验证。
3. 完成后把 `DEC-002` 从 PENDING 改成有结论的状态。
