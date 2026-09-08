# LCWLD-020 report: 独立 Python 元数据构建器

- 状态：REVIEW（等待主 agent / 用户签署为 DONE）
- 工单：LCWLD-020
- 日期：2026-08-28

## 修改文件

- `openmm-localcwld/python/openmm_localcwld/__init__.py`（新增）
- `openmm-localcwld/python/openmm_localcwld/metadata.py`（新增）
- `openmm-localcwld/python/tests/test_metadata.py`（新增，15 个合成体系单测）

未修改 `test_lips_vs_pmeV2.6.py` 或任何 DCD/CSV/PNG。

## 实现摘要

`build_particle_parameters(system, topology, *, a_q2, ca_source_weight,
enable_water_response, enable_solute_polarization, fast_active_density,
dpolar_o)` 返回冻结的 `ParticleParameters` dataclass（九个数组，顺序见计划
17.2 节）。逐条对照 `test_lips_vs_pmeV2.6.py:build_phase_cwld_metadata()`
（约 242-379 行）实现，细节：

- `residue_id` = `residue.index`（不是残基名/链 ID）。
- 水 O 只有在残基恰好含 1 个 O + 2 个 H 时才成为 source；这个隐式 shape gate
  被保留。
- 离子命名走 `ION_RESNAME_ALIASES`，但**不修改**调用者的 `Topology` 对象
  （v2.6 是原地改 `residue.name`；这里改成本地解析别名，分类结果相同，
  且不产生副作用）。
- `qref2`（charge_mod 用）严格只从"active water source"（水 O 且
  `dens_source>0`）取平均，逐级回退到全体水、再到全体系。
- 新增了一个 v2.6 没有暴露成参数的模块级常量 `FAST_ACTIVE_DENSITY`
  （默认 `True`），实现为 `fast_active_density` 关键字参数，默认值与 v2.6
  一致。

## 发现并记录的两处计划文档 / 参考实现不一致（已写入 PLAN-ERRATA）

1. 计划第 17.3 节说 `enable_solute_polarization=False` 时"溶质可以继续是
   source，但不得成为响应 sink"，但 v2.6 代码里整段溶质分类循环（含
   `dens_source` 赋值）都嵌在 `if enable_solute_polarization:` 内部——关闭后
   溶质完全不是 source。本实现按**代码**字面行为实现，已在
   `PLAN_LocalCWLDForce_OpenMM_Plugin.md` 第 54 节 Errata 列表追加条目。
2. `FAST_ACTIVE_DENSITY`（默认 True，把蛋白/配体的 C/H 排除在 density 之外）
   没有出现在计划 17.1/17.3 的冻结表格里，但确实改变哪些原子会是
   `dens_source=1`。同样已追加 Errata 条目。

这两条都只是"记录矛盾"，不代表由本工单单方面决定谁对——需要主 agent 确认。

## 测试证据

### 1) 合成体系单测（不需要真实 1CKK/力场文件）

```bash
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python3.12 -m pytest \
  openmm-localcwld/python/tests/test_metadata.py -v
```

结果：**15 passed in 0.79s**（全部通过，0 skip，0 xfail）。

覆盖：水响应开/关、残缺水分子（1 个 H）不作为 source、离子分类+别名解析+
`ca_source_weight` 覆盖+不修改输入 topology、ASP 带电侧链权重、
`fast_active_density` 默认/关闭两种行为、`enable_solute_polarization=False`
的记录性行为、脂质头基 source vs 尾链惰性、`charge_mod`/`qref2` 只用水 O、
`charge_mod` 上下界 clamp、`residue_id`==`residue.index`、
`LIGAND_CHARGED_SOURCE_WEIGHT` 分支（当前冻结常量下不可达，用
monkeypatch 单独验证分支本身正确）、非法长度的 `ParticleParameters` 抛异常、
输入 System/Topology 不被修改。

### 2) 与 v2.6 的真实逐元素对照（合成体系，10 个 case）

脚本：`docs/reports/lcwld-020-parity-check-synthetic.py`（纯验证脚本，
不是 pytest 套件的一部分，保留下来方便复现）。对每个 case 分别调用 v2.6 的 `build_phase_cwld_metadata()`（先
`normalize_ion_resnames()`）与本模块的 `build_particle_parameters()`，
逐元素比较 `qbase/charge_mod/dpolar/is_polar/dens_source/dens_sink/
source_class_weight/static_phase/residue_id`（对应 v2.6 的
`qbase/charge_mod_array/dpolar/is_polar/dens_source/dens_sink/
source_class_weight/static_phase/mol_ids`）。

10 个 case：纯水(响应开)、纯水(响应关)、残缺水、离子(+`ca_source_weight`
覆盖)、ASP、ASP(`fast_active_density=False`，无 v2.6 对应，仅自检)、LYS、
脂质头基、`enable_solute_polarization=False`、混合体系(2 水+Na+ASP,
`a_q2=0.5`)。

结果：**9/9 有 v2.6 对应的 case 全部 `max_abs_diff == 0`**（逐位相同，不是
容差内接近）；第 10 个 case（`fast_active_density=False`）按预期与 v2.6 不同
（因为 v2.6 无法表达这个新参数），只做自检，通过。

### 3) 与 v2.6 的真实 1CKK 体系对照（31358 原子，最强证据）

脚本：`docs/reports/lcwld-020-parity-check-1ckk.py`。直接调用
`test_lips_vs_pmeV2.6.py::build_1ckk_system()`（PDBFixer 修复 + Amber19SB/
TIP3P 力场 + 1.5 nm 水盒 + 0.15 M NaCl，**不含任何积分/Context/生产步**），
用得到的真实 `(system, topology)` 分别跑 v2.6 的
`build_phase_cwld_metadata()` 和本模块的 `build_particle_parameters()`
（默认参数：`a_q2=0.5, ca_source_weight=2.0, enable_water_response=True,
enable_solute_polarization=True, fast_active_density=True`）。

```
n_particles = 31358
[qbase] OK (exact match, 31358 particles)
[charge_mod] OK (exact match, 31358 particles)
[dpolar] OK (exact match, 31358 particles)
[is_polar] OK (exact match, 31358 particles)
[dens_source] OK (exact match, 31358 particles)
[dens_sink] OK (exact match, 31358 particles)
[source_class_weight] OK (exact match, 31358 particles)
[static_phase] OK (exact match, 31358 particles)
[residue_id vs mol_ids] OK
OVERALL 1CKK PARITY: EXACT MATCH
```

这满足计划 LCWLD-020 的完成定义："新模块与 v2.6 对同一 1CKK system 的九个
数组逐元素一致"。

## 环境

- `/home/ruigengji/miniforge3/envs/openmm_dev`，OpenMM 8.5.2，Python 3.12.13，
  numpy 2.4.3，pytest 9.1.1（详见 `docs/decisions/DEC-001-toolchain.md`）。
- 只做了系统构建（`build_1ckk_system()`）和纯 Python 元数据计算，**没有**
  创建 Integrator/Context，**没有**跑任何积分步，不属于"跑 MD"。

## 未解决问题

- 上面两条 PLAN-ERRATA 需要主 agent 拍板，再决定要不要更新计划正文的
  17.3 节措辞，以及是否要把 `fast_active_density` 补进冻结常量表。
- `dpolar_o`（v2.6 参数名 `dpolar_O`）目前只在关键字参数里保留了默认值
  `-0.15`，未做单独的覆盖值单测（计划本身也没要求覆盖它）。

## 是否修改 API/公式/容差

否。只做了纯抽取和一处非破坏性的实现方式调整（ion 别名解析改成非原地
方式，分类结果不变）。
