# LocalCWLDForce：局部环境响应 OpenMM 原生插件开发计划

> 状态：提案 / 开发路线图  
> 日期：2026-08-13  
> 当前参考实现：`test_lips_vs_pmeV2.6.py`  
> 暂定产品名：**LocalCWLDForce**

## 1. 命名决定

正式名称建议使用 **LocalCWLDForce**，中文称为“局部 CWLD 力插件”或“局部环境响应力插件”。

不建议把正式名称写成“LocalGBForce”或“局部 GB 插件”，因为 GB 通常指 Generalized Born 隐式溶剂模型，容易让使用者误以为本项目在计算 Born 半径或传统 GB/GBSA 溶剂化能。本项目虽然目前借用 OpenMM `CustomGBForce` 表达多阶段计算，但物理核心是：

1. 在有限半径内计算局部环境密度；
2. 由局部密度生成响应电荷 `Q`；
3. 通过有限截断的 ZMM/CWLD 静电项计算能量和力；
4. 保留 `dQ/dR` 链式响应力。

因此，`LocalCWLDForce` 同时表达了“局部”“CWLD 模型”和“它是 Force、不是 Integrator”这三件关键事实。

## 2. 项目目标

开发一个可被 OpenMM 动态加载的原生 Force 插件，以局部邻居表和 GPU kernel 替代当前通用 `CustomGBForce` 路径，在不改变物理模型的前提下显著降低 CWLD 力的计算成本。

目标调用形式示意：

```python
from openmm_localcwld import LocalCWLDForce

cwld = LocalCWLDForce(
    r_env=0.35,       # nm
    cutoff=1.20,      # nm
    zmm_order=2,
    rho0=13.5,
    k_polar=0.8,
    q_delta_clamp=0.20,
)

for params in particle_parameters:
    cwld.addParticle(*params)

for i, j in exclusions:
    cwld.addExclusion(i, j)

system.addForce(cwld)
```

积分仍使用 OpenMM 标准稳定积分器，例如 `LangevinMiddleIntegrator`。插件不负责时间推进。

## 3. 当前基线与问题

当前精确参考实现位于：

- 局部环境和粒子分类：`test_lips_vs_pmeV2.6.py:242`
- 普通/MTS 积分器选择：`test_lips_vs_pmeV2.6.py:389`
- CWLD `CustomGBForce`：`test_lips_vs_pmeV2.6.py:1063`
- CPU/KDTree 实验实现：`test_lips_vs_pmeV2.6.py:1225`
- 平台选择和性能测试：`test_lips_vs_pmeV2.6.py:1269`

已知事实：

- `CustomGBForce` 路径是当前可信的精确参考实现，但约占总步时的 81%。
- Python/KDTree 路径存在 GPU→CPU 坐标同步、Python 循环和参数回传开销，实际比精确路径更慢，不能作为生产实现。
- MTS 2:1 和 4:1 已出现严重升温，说明 CWLD 响应力不能被当作低频慢力更新。
- 当前脚本把模型、体系构建、运行调度、分析和绘图混在一个大型文件中，不适合直接作为插件接口。

## 4. “局部”的精确定义

这里的局部是**空间有限支撑和稀疏响应集合**，不是“每隔若干步才更新”，也不是“只计算盒子中某一块固定区域”。

插件必须实现两种局部范围：

| 范围 | 默认值 | 用途 |
|---|---:|---|
| `r_env` | 0.35 nm | 计算响应位点周围的局部密度 |
| `cutoff` / `rc` | 1.20 nm | 计算 CWLD/ZMM pair energy 和 force |

局部稀疏性包括：

- 仅 `dens_sink != 0` 的粒子需要累积环境密度；
- 仅 `is_polar != 0` 且响应开关有效的粒子需要改变 `Q`；
- 仅 `dens_source != 0` 的粒子对环境密度有贡献；
- 固定电荷粒子的 `Q=qbase`，不需要执行响应公式；
- 分子内被排除的 pair 不进入密度或 CWLD pair 计算；
- 使用周期性边界条件和邻居表，禁止全体系 `O(N²)` 两两扫描。

CWLD 的 `Q` 和响应力仍应每个积分步计算。邻居表可以按 OpenMM 的安全 skin/rebuild 机制复用，但不能用低频 Q 更新替代正确的逐步力计算。

## 5. 必须保持一致的数学语义

第一版插件的任务是复现，不是重新发明模型。以下语义必须与 v2.6 一致。

### 5.1 局部密度

对于响应粒子 `i`：

```text
dens_i = Σ_j [
    dens_sink_i
  × dens_source_j
  × source_class_weight_j
  × charge_mod_j
  × density_kernel(r_ij)
  × different_residue_id(i, j)
]
```

其中 `density_kernel(r)` 在 `r < r_env` 内为 `(1-(r/r_env)^2)^2`，在范围外为零。注意当前变量虽然叫 `mol_id`，实际赋值为 `residue.index`。所以 v2.6 的精确语义是“相同 residue ID 不计入 density”，不是一般意义上的“相同 molecule 不计入”。第一版必须保持这一行为；以后若改成真正的 molecule ID，必须作为独立物理模型变更处理，不能混入性能移植。

### 5.2 响应电荷

```text
Q_i = qbase_i + q_delta_clamp × tanh(
    static_phase_i × is_polar_i × dpolar_i
    × tanh(k_polar × dens_i / rho0)
    / q_delta_clamp
)
```

参数单位、单双精度行为和极端输入下的数值稳定性都需要明确测试。

### 5.3 Pair energy

```text
U_ij = ONE_4PI_EPS0 × [
    Q_i Q_j U_ell(r_ij)
  - qbase_i qbase_j (1/r_ij - 1/rc)
]
```

`U_ell` 第一版至少支持当前使用的 `ell=1,2,3`，默认 `ell=2`。插件负责 CWLD 修正项；配套 Python builder 负责把原始 `NonbondedForce` 设置成与参考实现一致的 cutoff/reaction-field 语义，避免重复计算或漏算基线静电。

### 5.4 Force 与链式导数

必须包含：

- pair energy 对距离的直接导数；
- `Q_i(dens_i(R))` 和 `Q_j(dens_j(R))` 带来的全部链式导数；
- 密度源粒子和响应粒子两侧的反作用力；
- 周期性最小镜像和 exclusion 语义。

仅让能量一致、但忽略 `dQ/dR` 的实现不合格，即使短轨迹表面稳定也不能进入生产阶段。

## 6. 推荐架构

OpenMM 官方的 Force 插件结构由公共 `Force` API、`ForceImpl`、抽象 kernel，以及各 Platform 的 kernel factory/实现组成。第一版建议包含：

```text
openmm-localcwld/
├─ CMakeLists.txt
├─ README.md
├─ LICENSE
├─ include/
│  └─ LocalCWLDForce.h
├─ openmmapi/
│  ├─ LocalCWLDForce.cpp
│  ├─ LocalCWLDForceImpl.cpp
│  ├─ LocalCWLDForceImpl.h
│  └─ LocalCWLDKernels.h
├─ platforms/
│  ├─ reference/
│  └─ common/                 # CUDA/OpenCL/HIP 可共用的 Common Compute 路径
├─ python/
│  ├─ openmm_localcwld/
│  │  ├─ __init__.py
│  │  ├─ builder.py
│  │  └─ parameters.py
│  └─ wrappers/
├─ serialization/
├─ tests/
│  ├─ unit/
│  ├─ regression/
│  └─ performance/
└─ examples/
   └─ 1ckk_minimal.py
```

OpenMM 插件加载、Force/ForceImpl/kernel factory 的基本结构遵循官方开发文档：<https://docs.openmm.org/latest/developerguide/>。

### 6.1 Kernel 数据流

建议先采用清晰的多阶段 kernel，再根据 profiling 决定是否融合：

```text
OpenMM neighbor list / tile list
        │
        ├─ Kernel A: r_env 内稀疏 dens 累积
        │
        ├─ Kernel B: dens → Q
        │
        ├─ Kernel C: rc 内 CWLD/ZMM energy + direct force
        │
        └─ Kernel D: dQ/dR 链式响应力回传
```

优化原则：

- 优先复用 OpenMM Common Compute 的 pair/tile 基础设施；
- 使用一个 `rc` 邻居结构，并在密度 kernel 中额外判断 `r < r_env`，避免维护两个昂贵邻居表，除非 profiling 证明双表更快；
- 将固定 per-particle 参数保持在设备端；
- 禁止每步主机同步 Q 或坐标；
- 能量未请求时允许跳过能量归约，但力结果必须相同；
- 支持 OpenMM mixed precision，double/reference 用于验证。

## 7. 公共 API 边界

第一版 `LocalCWLDForce` 建议暴露：

### 全局参数

- `r_env`
- `cutoff`
- `rho0`
- `k_polar`
- `q_delta_clamp`
- `zmm_order`
- `ONE_4PI_EPS0`
- 是否启用可选 Q penalty（默认关闭）

### 每粒子参数

- `qbase`
- `charge_mod`
- `dpolar`
- `is_polar`
- `dens_source`
- `dens_sink`
- `source_class_weight`
- `static_phase`
- `molecule_id`

### 结构参数

- exclusions
- cutoff method，仅第一版支持 `CutoffPeriodic`
- force group

### 更新语义

- Context 创建前允许自由配置；
- 第一版只保证少量全局参数可在 Context 内更新；
- per-particle 参数更新若代价和正确性尚未验证，应明确要求重新初始化 Context；
- 提供 XML serialization 和 Python binding round-trip 测试。

粒子分类和生物化学规则不放入 C++ Force。水、离子、蛋白、配体和脂质的识别由 Python `builder.py` 完成，Force 只接收数值参数。这样插件不会被 1CKK 或某套残基命名绑死。

## 8. 分阶段实施计划

### 阶段 0：冻结参考模型

任务：

- 从 v2.6 抽取纯 Python 参数构建与参考计算模块；
- 固定 `a_q2=0.5`、`ca_source_weight=2.0` 的默认参考配置，但允许显式覆盖；
- 保存几个小体系的输入、粒子参数、exclusions、参考 dens/Q/energy/force；
- 为实验输出增加配置指纹，杜绝仅凭同名 DCD/CSV 静默复用。

验收：

- 不跑长轨迹即可重建相同的每粒子参数；
- 参考 fixture 可在 CPU double precision 下稳定复现；
- fixture 包含水盒、离子、水+Ca²⁺、小肽以及裁剪的 1CKK 局部环境。

### 阶段 1：Reference CPU 正确性原型

任务：

- 实现 `LocalCWLDForce`、`ForceImpl` 和 Reference kernel；
- 先使用清晰的 CPU 实现验证所有导数和 exclusion；
- 增加 C++ 与 Python API、序列化支持；
- 接入现有 OpenMM System builder。

验收：

- dens 和 Q 与 Python/CustomGB 参考逐粒子一致；
- energy 与参考一致；
- analytic force 通过中心有限差分；
- 平移不改变能量，总力接近零；
- 周期边界跨盒 pair 与参考一致；
- serialization 前后结果一致。

### 阶段 2：GPU/Common Compute 实现

任务：

- 实现 CUDA 优先的 Common Compute kernel；
- 设备端保存参数、dens、Q 和必要的导数中间量；
- 使用邻居 tile/cell list 和 active source/sink mask；
- 消除每步 GPU↔CPU 同步；
- 支持 single、mixed、double（以平台能力为准）。

验收：

- CUDA mixed/double 对 Reference/CustomGB 的 energy 和 force 在预定容差内；
- 不同原子排序、盒尺寸和活性位点比例下均通过；
- CUDA 不可用时明确报错或选择已实现平台，不允许静默改变物理模型。

### 阶段 3：短轨迹动力学回归

任务：

- 同一初始状态、同一随机种子分别运行 CustomGB 与 LocalCWLDForce；
- 做最小化、短 NVE、NVT 和当前标准生产流程；
- 比较温度、势能、约束误差、RMSD、Ca-water RDF 和响应电荷分布。

验收：

- NVE 测试没有插件特有的系统性漂移；
- NVT 温度分布与参考一致；
- 短时间确定性轨迹在混沌分离前与参考吻合；
- 统计指标差异落在重复轨迹噪声内；
- 不使用 MTS 为性能结果“加分”。

### 阶段 4：性能优化与基准

基准体系至少包含：

- 小水盒，用于观察固定开销；
- 1CKK 当前体系，用于直接业务比较；
- 更大蛋白/膜体系，用于验证扩展性；
- 不同响应位点比例和不同 `r_env/rc` 组合。

记录：

- ns/day；
- 每步 wall time；
- density、Q、pair、chain-force 各 kernel 时间；
- 邻居表构建占比；
- GPU occupancy、带宽和同步；
- 峰值显存；
- 相对 PME 和 CustomGB 的开销。

进入生产候选版本的建议门槛：

- 1CKK 上相对当前 CustomGB 至少 **2×** 加速；
- 不出现 CPU/KDTree 路径中的逐步主机同步；
- 插件加入后的总模拟速度明显优于现有约 131 ns/day 基线；
- 所有正确性门槛先于速度门槛，不能用放宽物理语义换取性能。

### 阶段 5：科学回归与发布

任务：

- 重新运行 PME / CustomGB / LocalCWLDForce 的统一短矩阵；
- 选择关键配置进行 3 seeds 回归；
- 至少在第二个独立 Ca²⁺ 体系上验证；
- 建立版本化参数文件、环境锁、安装包和用户文档；
- 发布 Linux + CUDA 为第一支持组合，随后再决定 OpenCL/HIP/Windows。

验收：

- 1CKK 的已冻结科学结论不因实现替换而改变；
- 第二体系无明显数值或物理异常；
- 从干净环境可安装、加载、运行最小示例；
- 生成结果记录插件版本、OpenMM 版本、平台、精度和参数指纹。

## 9. 测试矩阵与初始容差

具体容差应通过 Reference double 的量级测试校准，不能盲目固定。建议初始门槛：

| 项目 | Reference/double 初始目标 | CUDA mixed 初始目标 |
|---|---:|---:|
| 单粒子 `dens` | `rtol <= 1e-10` | `rtol <= 1e-5` |
| 单粒子 `Q` | `rtol <= 1e-10` | `rtol <= 1e-5` |
| 总能量 | `rtol <= 1e-9` | `rtol <= 1e-5` |
| 单原子力 | `rtol <= 1e-7` | `rtol <= 1e-4` |
| 有限差分力 | 与步长扫描的收敛区间一致 | 以 double 结果为基准 |

必须覆盖的边界情况：

- `r = 0` 防护和极近距离；
- `r` 位于 `r_env` 内外及边界附近；
- `r` 位于 `rc` 内外及边界附近；
- 无 source、无 sink、无 polar 粒子；
- same-residue 对 density edge 的屏蔽，以及显式 exclusion 对 density 与 Local pair 的屏蔽；
- 正负 `dpolar`、极端 density、tanh 饱和；
- 单价/二价离子的不同 source weight；
- 多 residue、跨周期边界和非正交/triclinic 盒；
- 粒子重排序后结果不变。

## 10. 明确不做的内容

第一阶段不做：

- 不开发新的积分器；
- 不恢复 MTS 或低频 Q 更新；
- 不把 Python/KDTree reporter 包装成“插件”；
- 不改变 `a_q2`、`ca_source_weight` 或响应公式；
- 不将 PME 近似成“真值”来重新拟合参数；
- 不在 C++ 层硬编码 1CKK、残基名或某个力场；
- 不承诺第一版跨所有平台；
- 不在缺少 force derivative 测试时运行长生产轨迹。

## 11. 主要风险与对策

### 风险 A：链式力实现错误

这是最高风险。对策是先完成 Reference CPU、有限差分和 CustomGB 三方一致性，再写 GPU；每个优化都必须跑 force regression。

### 风险 B：重复或漏算静电基线

当前实现通过修改原生 `NonbondedForce` 并添加 CWLD correction 形成总能量。对策是将这套组合封装在唯一的 Python builder 中，同时对“基线”“修正”“总能量”分别测试。

### 风险 C：邻居表加速收益低于预期

对策是阶段 2 先做可测量的多 kernel 正确实现，使用 profiler 决定融合、双邻居表、原子压缩列表等优化，不凭直觉重写。

### 风险 D：数值一致但动力学偏移

对策是设置 NVE/NVT 动力学门槛，并保留 CustomGB 参考路径，至少完成 3-seed 科学回归后才替换默认实现。

### 风险 E：OpenMM 版本和 ABI 兼容

对策是明确支持的 OpenMM 主版本，使用 CMake package config，CI 中构建支持矩阵，并在插件加载时报告版本不兼容，而不是静默运行。

## 12. 建议里程碑

| 里程碑 | 交付物 | 完成定义 |
|---|---|---|
| M0 参考冻结 | 独立 reference 模块和 fixtures | dens/Q/E/F 可重复生成 |
| M1 CPU 原型 | Reference `LocalCWLDForce` | API、序列化、有限差分全部通过 |
| M2 GPU Alpha | CUDA/Common Compute kernel | 与 Reference/CustomGB 一致，无主机逐步同步 |
| M3 动力学 Beta | NVE/NVT/短生产回归报告 | 无插件特有漂移或升温 |
| M4 性能候选 | 1CKK 与大体系 benchmark | 1CKK 至少 2× CustomGB |
| M5 v0.1 发布 | 包、文档、示例、版本化参数 | 干净环境可安装运行，3-seed 回归通过 |

## 13. 开工前的第一批具体任务

1. 新建独立源码仓库或至少独立子目录，避免继续在 `test_lips_vs_pmeV2.6.py` 中扩展。
2. 将 `build_phase_cwld_metadata()`、ZMM closure 和参考 dens/Q/energy 计算抽成无模拟副作用模块。
3. 为 4–5 个小体系生成 double precision golden fixtures。
4. 写出总能量拆分和 force finite-difference 测试。
5. 搭建 OpenMM Force plugin 的 CMake/API/Reference kernel 骨架。
6. Reference 全部通过后，再启动 Common Compute/CUDA 实现。

## 14. 最终完成标准

只有同时满足以下条件，`LocalCWLDForce` 才可替换当前 `CustomGBForce` 默认路径：

- 数学语义与 v2.6 参考实现一致；
- `dQ/dR` 链式力经过有限差分验证；
- 不使用低频响应更新或不稳定 MTS；
- 1CKK 和至少一个外部体系通过动力学/科学回归；
- 1CKK 性能达到预定加速目标；
- 环境、参数、代码版本和输出均可追踪；
- 原有 CustomGB 实现仍保留为回归 oracle，而不是立即删除。

这份计划的核心原则是：**模型保持逐步精确，计算通过空间局部性和 GPU 稀疏化加速。**

---

# 第二部分：面向执行 Agent 的实施手册

> 本部分是实施时的权威操作说明。若前面的路线图与本部分存在粒度差异，以本部分的接口冻结、任务依赖和验收门禁为准。任何 agent 都不得在自己的工单中改变物理公式、默认参数或验证阈值。

## 15. Agent 执行协议

每个执行 agent 在开始前必须完成下面五件事：

1. 阅读自己工单的“前置依赖”“允许修改文件”“禁止修改文件”和“完成定义”。
2. 检查前置工单的测试是否已经通过；没有通过就停止，不得自行绕过。
3. 只修改工单列出的文件。若必须修改范围外文件，应把原因提交给主 agent，由主 agent重新划分所有权。
4. 实现后运行工单规定的最小测试，并记录完整命令、平台、精度模式和结果。
5. 交付时报告：修改文件、实现摘要、测试证据、未解决问题、是否改变公共 API 或数值结果。

统一禁止事项：

- 不得修改本计划中的公式来让测试通过。
- 不得放宽容差、删除失败测试或把失败改成 skip。
- 不得启用 MTS、低频 Q 更新或 Python reporter 更新路径。
- 不得把 `q_driver`、动态 phase/interface 规则带进 exact 插件。
- 不得以 PME、AMOEBA 或新拟合值替换冻结参数。
- 不得提交 build 目录、二进制、DCD、缓存或本机路径。
- 不得让 CUDA 不可用时静默回退 CPU 并把结果记为 CUDA。
- 不得在单元测试中启动长时间 1CKK 模拟。

每张工单的状态只能是：`BLOCKED`、`READY`、`IN_PROGRESS`、`REVIEW`、`DONE`。只有所有硬性验收通过才可标记 `DONE`。

## 16. 冻结词汇和标识符

所有实现统一使用以下名称，避免不同 agent 各自命名：

| 概念 | 冻结名称 |
|---|---|
| 项目/发行名 | `openmm-localcwld` |
| C++ Force 类 | `OpenMM::LocalCWLDForce` |
| C++ API 库 | `OpenMMLocalCWLD` |
| 抽象 kernel | `CalcLocalCWLDForceKernel` |
| Reference 插件库 | `OpenMMLocalCWLDReference` |
| Common Compute 代码前缀 | `LocalCWLDCommon` |
| CUDA 插件库 | `OpenMMLocalCWLDCUDA` |
| Python 包 | `openmm_localcwld` |
| Python builder | `build_local_cwld_system()` |
| 参考 Python 包 | `localcwld_reference` |
| 结果 schema | `localcwld-result-v1` |

不要使用 `LocalGBForce`、`LipsForce`、`GBIntegrator` 或其他临时名称。若未来品牌名改变，应由单独的全局重命名工单完成。

## 17. v0.1 物理语义冻结表

### 17.1 全局常量和默认值

| 名称 | 默认值 | 单位 | 来源/说明 |
|---|---:|---|---|
| `r_env` | 0.35 | nm | density 有限支撑半径 |
| `rc` | 1.20 | nm | LocalCWLD pair cutoff |
| `r_on` | 0.90 | nm | 仅用于配套原生 `NonbondedForce` switching，不属于 LocalCWLD pair 公式 |
| `rho0` | 13.5 | 当前模型密度标度 | 保持 v2.6 |
| `k_polar` | 0.8 | 无量纲组合系数 | 保持 v2.6 |
| `q_delta_clamp` | 0.20 | e | Q 响应软上限 |
| `ONE_4PI_EPS0` | 138.935458 | kJ mol⁻¹ nm e⁻² | 静电常数 |
| `zmm_order` | 2 | 整数 | 允许 1、2、3 |
| `a_q2` | 0.5 | 构建期参数 | 只用于计算 `charge_mod`，不是 Force runtime global |
| `ca_source_weight` | 2.0 | 构建期参数 | 二价离子 source weight |
| `water_source_weight` | 0.5 | 构建期参数 | 只有水 O 是 source |
| `ion_source_weight` | 2.0 | 构建期参数 | 单价离子 |
| `q_penalty_enabled` | false | — | v0.1 默认关闭 |
| `q_penalty_k` | 180.0 | 与参考表达式一致 | 仅显式启用时使用 |

`a_q2`、水/离子/残基分类、`charge_mod` 和粒子 mask 都在 Python builder 构建 Context 前计算。低等级 agent 不得把它们错误地实现为每步重新计算的 GPU global。

### 17.2 每粒子参数顺序

C++ API、序列化、Python wrapper、fixture 和设备数组必须统一使用以下顺序：

1. `qbase`：基础电荷，单位 e。
2. `charge_mod`：构建期密度源修饰值。
3. `dpolar`：响应幅度和符号。
4. `is_polar`：响应 mask/权重。
5. `dens_source`：是否贡献密度。
6. `dens_sink`：是否接收密度。
7. `source_class_weight`：作为 source 时的类别权重。
8. `static_phase`：响应开关。
9. `residue_id`：v2.6 中的 `mol_id`，实际必须传 `residue.index`。

C++ 中 `residue_id` 使用整数类型；序列化时写整数。其余使用 `double`。不得在设备端根据 residue 名重新分类。

### 17.3 元数据分类精确规则

Python builder 必须逐条复现：

- 水残基名：`HOH`、`WAT`、`SOL`、`TP3`。
- 水 O：始终是 source，`dens_source=1`、`source_class_weight=0.5`。
- 水 H：永远不是 source。
- 开启 water response 时，O/H/H 均为 sink 和 polar；O 的 `dpolar=-0.15`，两个 H 各为 `+0.075`。
- 不要强行施加每个水分子的瞬时 `ΔQO+ΔQH1+ΔQH2=0`；当前 exact 模型没有这一约束。
- 单价离子是 source，权重 `2.0`；Ca/Mg/Zn 是 source，权重默认 `2.0`，但必须允许 builder 显式覆盖。
- 脂质头部 O/N/P/S 是 source，权重 `0.3`；脂质分类仍由 Python 层完成。
- 蛋白/配体普通极性 O/N/S 的 `dpolar` 分别为 `0.012/0.012/0.015`，source weight `1.0`。
- ARG `NE/NH1/NH2`、LYS `NZ`、ASP `OD1/OD2`、GLU `OE1/OE2`、HIS `ND1/NE2` 使用 charged-site 权重 `1.5`。
- `enable_solute_polarization=False` 时，溶质可以继续是 source，但不得成为响应 sink。
- `phase_ref_*`、nearest-distance phase、`q_driver` 只属于分析或旧 fast 路径，不进入 LocalCWLDForce。

### 17.4 charge modifier

构建期计算：

```text
active_water_source = water_mask AND dens_source > 0
qref2 = mean(qbase² over active_water_source)
```

若没有 active water source，则依次回退到所有水粒子、再回退到全体粒子。然后：

```text
charge_mod_i = clip(1 + a_q2 × (qbase_i² - qref2), 0.25, 2.5)
```

TIP3P 正常情况下 `qref2` 只能由水 O 计算，不能把 H 混入平均。

## 18. 可直接编码的数学规格

定义：

```text
A_i = static_phase_i × is_polar_i × dpolar_i
B_i = dens_source_i × source_class_weight_i × charge_mod_i
S_i = dens_sink_i
sameResidue(i,j) = residue_id_i == residue_id_j
```

### 18.1 Density pass

```text
K(r) = (1 - (r/r_env)²)²,  0 <= r < r_env
K(r) = 0,                  r >= r_env

dens_i = S_i × Σ_j [ B_j × K(r_ij) × I(not sameResidue(i,j)) × I(not excluded(i,j)) ]
```

方向性是硬约束：`j` 的 source 参数贡献给 `i` 的 sink。不得改成 `B_i B_j`，不得用 target 的 `source_class_weight_i`，也不得为了并行方便把 density 公式对称化。

### 18.2 Q pass

```text
t_i = tanh(k_polar × dens_i / rho0)
y_i = A_i × t_i / q_delta_clamp
Q_i = qbase_i + q_delta_clamp × tanh(y_i)
```

对应导数：

```text
dQ_i/ddens_i =
    A_i × (k_polar/rho0)
    × (1 - tanh(y_i)²)
    × (1 - t_i²)
```

如果 `S_i=0` 或 `A_i=0`，则 `dens_i` 或导数自然为零，`Q_i=qbase_i`。不得用硬 clip 替换嵌套 `tanh`。

### 18.3 ZMM closure

```text
U1(r) = 1/r - 3/(2rc) + r²/(2rc³)
U2(r) = 1/r - 15/(8rc) + 5r²/(4rc³) - 3r⁴/(8rc⁵)
U3(r) = 1/r - 35/(16rc) + 35r²/(16rc³)
        - 21r⁴/(16rc⁵) + 5r⁶/(16rc⁷)
```

实现时必须同时写出并测试解析 `dU/dr`；禁止数值差分用于生产 kernel。

### 18.4 Pair energy 和直接力

对每个未排除且 `r < rc` 的无序 pair：

```text
E_ij = C × [ Q_i Q_j Uell(r) - qbase_i qbase_j (1/r - 1/rc) ]
C = 138.935458
```

直接径向导数：

```text
dE_ij/dr = C × [ Q_i Q_j dUell/dr + qbase_i qbase_j/r² ]
```

LocalCWLD pair 本身不再套 `r_on` switching。`r_on` 只属于配套原生 NonbondedForce。

### 18.5 链式响应力

先计算每个粒子的 adjoint：

```text
lambda_i = dE/dQ_i
         = C × Σ_j[Q_j Uell(r_ij)]
```

若启用 Q penalty，再加：

```text
lambda_i += q_penalty_k × (Q_i - qbase_i)
```

定义：

```text
g_i = lambda_i × dQ_i/ddens_i
K'(r) = -4r/r_env² × (1 - r²/r_env²),  r < r_env
```

每个 density 邻居 pair 对径向能量导数的附加贡献为：

```text
chain_dEdr(i,j) = [g_i S_i B_j + g_j S_j B_i] × K'(r_ij)
```

仅当 residue 不同、pair 未排除且 `r < r_env` 时存在。最终 force 必须同时包含 direct 和 chain 两部分。这个四阶段结构可以作为 Reference 和 GPU 实现的共同算法：

1. density accumulation；
2. Q 与 `dQ/ddens`；
3. pair energy/direct force 与 `lambda`；
4. density backprop chain force。

### 18.6 Q penalty

仅显式启用时：

```text
E_penalty_i = 0.5 × q_penalty_k × (Q_i - qbase_i)²
```

它不仅增加能量，也通过 `lambda_i` 进入链式力。只加能量而漏掉响应力属于错误。

## 19. CustomGB 数值兼容决策门

当前 v2.6 使用 1024 点 `Continuous1DFunction` 表示 density kernel，而原生插件最自然的实现是解析多项式。这两者可能产生很小的插值差异。

在 `LCWLD-030` fixture 完成后，主 agent 必须生成 `DEC-002-density-kernel.md`：

- 比较 10,000 个距离点上的 tabulated-v2.6 与 analytic 值/导数；
- 比较所有小体系的 dens/Q/E/F；
- 若误差低于 Reference 门槛，v0.1 使用解析 kernel；
- 若误差超过门槛，v0.1 必须实现 `V26Tabulated1024` 兼容模式，并把 analytic 模式标为实验性；
- 执行 agent 无权自行选择或悄悄更新 golden。

## 20. 公共 API 冻结草案

`LocalCWLDForce.h` 第一版至少包含：

```cpp
class LocalCWLDForce : public Force {
public:
    enum NonbondedMethod { CutoffPeriodic = 0 };

    LocalCWLDForce();

    int addParticle(double qbase, double chargeMod, double dpolar,
                    double isPolar, double densSource, double densSink,
                    double sourceClassWeight, double staticPhase,
                    int residueId);
    int getNumParticles() const;
    void getParticleParameters(int index, double& qbase, double& chargeMod,
                    double& dpolar, double& isPolar, double& densSource,
                    double& densSink, double& sourceClassWeight,
                    double& staticPhase, int& residueId) const;
    void setParticleParameters(int index, ...same fields...);

    int addExclusion(int particle1, int particle2);
    int getNumExclusions() const;
    void getExclusionParticles(int index, int& particle1, int& particle2) const;
    void setExclusionParticles(int index, int particle1, int particle2);

    double getEnvironmentCutoff() const;
    void setEnvironmentCutoff(double distance);
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    int getZMMOrder() const;
    void setZMMOrder(int order);
    double getRho0() const;
    void setRho0(double value);
    double getKPolar() const;
    void setKPolar(double value);
    double getChargeDeltaClamp() const;
    void setChargeDeltaClamp(double value);
    bool getUseQPenalty() const;
    void setUseQPenalty(bool enabled);
    double getQPenaltyStrength() const;
    void setQPenaltyStrength(double value);

    void updateParametersInContext(Context& context);
    bool usesPeriodicBoundaryConditions() const override;
protected:
    ForceImpl* createImpl() const override;
};
```

API 规则：

- `r_env > 0`、`rc > 0`、`r_env <= rc`。
- `zmm_order` 只允许 1/2/3。
- `rho0 > 0`、`q_delta_clamp > 0`。
- 所有浮点参数必须 finite；禁止静默修正 NaN、负 cutoff 或错误粒子索引。
- Context 初始化时，Force 粒子数必须等于 System 粒子数。
- 重复 exclusion、self exclusion 的处理方式必须写入 API 文档并有测试；建议拒绝 self exclusion，规范化 pair 顺序，并拒绝重复项。
- `updateParametersInContext()` v0.1 只更新每粒子数值，不允许改变粒子数或 exclusion 拓扑；全局参数是否可 Context 更新必须逐项测试后承诺。
- `usesPeriodicBoundaryConditions()` 返回 true。
- `r_on` 不进入 Force API，由 Python builder 配置原生 NonbondedForce。
- `a_q2` 不进入 Force API，由 Python builder 预计算 `charge_mod`。
- 静电常数 v0.1 固定为 138.935458；若为测试暴露 setter，必须注明非生产用途并序列化。

## 21. 推荐源码树和文件所有权

```text
openmm-localcwld/
├─ CMakeLists.txt
├─ README.md
├─ LICENSE
├─ cmake/
│  ├─ FindOpenMM.cmake
│  └─ EncodeKernelFiles.cmake
├─ openmmapi/
│  ├─ include/openmm/LocalCWLDForce.h
│  ├─ include/openmm/LocalCWLDKernels.h
│  ├─ include/openmm/internal/LocalCWLDForceImpl.h
│  ├─ include/openmm/internal/windowsExportLocalCWLD.h
│  └─ src/
│     ├─ LocalCWLDForce.cpp
│     └─ LocalCWLDForceImpl.cpp
├─ serialization/
│  ├─ src/LocalCWLDForceProxy.cpp
│  ├─ src/LocalCWLDSerializationProxyRegistration.cpp
│  └─ tests/TestSerializeLocalCWLDForce.cpp
├─ platforms/
│  ├─ reference/
│  │  ├─ src/ReferenceLocalCWLDKernels.h
│  │  ├─ src/ReferenceLocalCWLDKernels.cpp
│  │  ├─ src/ReferenceLocalCWLDKernelFactory.cpp
│  │  ├─ src/ReferenceLocalCWLDPlugin.cpp
│  │  └─ tests/TestReferenceLocalCWLDForce.cpp
│  ├─ common/
│  │  ├─ src/CommonLocalCWLDKernels.h
│  │  ├─ src/CommonLocalCWLDKernels.cpp
│  │  ├─ src/LocalCWLDForceInfo.h
│  │  ├─ src/LocalCWLDForceInfo.cpp
│  │  └─ kernels/localCWLD.cc
│  └─ cuda/
│     ├─ src/CudaLocalCWLDKernelFactory.cpp
│     ├─ src/CudaLocalCWLDPlugin.cpp
│     └─ tests/TestCudaLocalCWLDForce.cpp
├─ python/
│  ├─ localcwld.i
│  ├─ setup.py.in
│  ├─ openmm_localcwld/
│  │  ├─ __init__.py
│  │  ├─ builder.py
│  │  ├─ metadata.py
│  │  ├─ manifest.py
│  │  └─ reference.py
│  └─ tests/
├─ tests/
│  ├─ fixtures/
│  ├─ unit/
│  ├─ integration/
│  └─ benchmarks/
├─ schemas/localcwld-result-v1.schema.json
└─ examples/1ckk_minimal.py
```

每个工单必须独占自己列出的文件。生成的 SWIG wrapper、encoded kernel source、build tree 和安装 staging 不进入源码仓库。

## 22. 工单依赖图

```text
LCWLD-000 决策与环境冻结
   ├── LCWLD-010 仓库/CMake 骨架
   ├── LCWLD-020 Python 元数据模块
   └── LCWLD-030 Golden fixtures 与参考计算
          │
LCWLD-010 ─┼── LCWLD-040 C++ Public API
LCWLD-030 ─┘          │
                      ├── LCWLD-050 ForceImpl/Kernel contract
                      ├── LCWLD-060 XML serialization
                      └── LCWLD-070 Reference CPU kernel
                                │
                                ├── LCWLD-080 Python SWIG/builder 接入
                                ├── LCWLD-090 Common Compute host
                                │        └── LCWLD-100 GPU device kernels
                                │                 └── LCWLD-110 CUDA 注册/加载
                                └── LCWLD-120 数值与有限差分门禁
                                                   │
                         LCWLD-110 ─────────────────┤
                                                   ├── LCWLD-130 NVE/NVT 回归
                                                   ├── LCWLD-140 1CKK smoke/science
                                                   └── LCWLD-150 性能 benchmark
                                                            │
                                        LCWLD-060/080/120/140/150
                                                            └── LCWLD-160 CI/打包/发布
```

并行规则：只有依赖图中没有上下游关系、且文件所有权不重叠的工单可以并行。Reference kernel、Common host 和 CUDA kernel 不得在数学规格尚未冻结时同时开工。

## 23. 标准工单模板

主 agent 派发工单时必须复制下面模板并填完整：

```text
工单：LCWLD-xxx 标题
状态：READY
目标：一句话、可验证
前置依赖：列出 DONE 工单
允许修改：精确文件列表
禁止修改：精确文件/模块
输入：fixture、接口版本、公式章节
步骤：编号操作
必须运行：完整命令
交付物：文件和报告
完成定义：所有硬性断言
失败处理：什么情况下返回 BLOCKED
```

Agent 回报模板：

```text
状态：REVIEW / BLOCKED
修改文件：...
实现内容：...
执行命令：...
测试结果：通过数/失败数/耗时
数值差异：dens/Q/E/F 最大绝对和相对误差
未解决问题：...
是否修改 API/公式/容差：必须回答否；若是则不能进入 REVIEW
```

## 24. 逐张实施工单

### LCWLD-000：环境与设计决策冻结

目标：在写 C++ 前固定 OpenMM 版本、平台范围、编译器、目录位置和两个关键兼容决策。

允许修改：

- `openmm-localcwld/docs/decisions/DEC-001-toolchain.md`
- `openmm-localcwld/docs/decisions/DEC-002-density-kernel.md`（先创建待实验状态）
- `openmm-localcwld/docs/decisions/DEC-003-platform-scope.md`

步骤：

1. 在计算节点记录 OpenMM、Python、CUDA toolkit、驱动、CMake、C++ 编译器版本。
2. 将 v0.1 最低支持目标暂定为 Linux + Reference + CUDA；OpenCL/HIP/Windows 仅构建可扩展边界，不承诺发布。
3. 决定代码放在 `/home/ruigengji/L-IPS/2.5/openmm-localcwld` 子目录还是独立仓库。默认先放子目录，成熟后迁出。
4. 记录 OpenMM CMake target、插件目录、Python module 名称，禁止根据旧 `simtk.openmm` 文档猜测。
5. 标记 `DEC-002` 为 `PENDING LCWLD-030`，不得提前选择 analytic/tabulated。

完成定义：三份决策文档包含日期、结论、证据命令、支持/不支持范围；任何未知项显式写 `UNKNOWN`，不能留给实现者暗自假设。

### LCWLD-010：仓库和 CMake 最小骨架

前置：LCWLD-000。

允许修改：根 `CMakeLists.txt`、`cmake/*`、`README.md`、`LICENSE`、`.gitignore` 和各子目录空 CMake 文件。禁止实现任何物理计算。

步骤：

1. 创建第 21 节目录树。
2. 配置选项：`LOCALCWLD_BUILD_REFERENCE`、`LOCALCWLD_BUILD_CUDA`、`LOCALCWLD_BUILD_PYTHON`、`BUILD_TESTING`。
3. 查找 OpenMM；找不到时 configure 必须给出清晰的 `OPENMM_DIR` 提示。
4. 创建 API library 占位 target `OpenMMLocalCWLD`。
5. 安装 public headers 到 `include/openmm`，API 库到 `lib`，平台插件最终安装到 `lib/plugins`。
6. 设置 C++17、Windows export 宏、版本号和 install RPATH；不得硬编码 `/home/ruigengji`。
7. `.gitignore` 排除 `build*`、`stage`、SWIG/CUDA 生成物、pycache、DCD、作业日志。

验收命令：

```bash
cmake -S . -B build -DOPENMM_DIR=$OPENMM_PREFIX \
  -DLOCALCWLD_BUILD_REFERENCE=OFF \
  -DLOCALCWLD_BUILD_CUDA=OFF \
  -DLOCALCWLD_BUILD_PYTHON=OFF
cmake --build build --parallel
cmake --install build --prefix stage
```

完成定义：configure/build/install 成功；源码树没有生成物；CUDA 关闭时不要求 CUDA toolkit。

### LCWLD-020：独立 Python 元数据构建器

前置：LCWLD-000。允许修改 `python/openmm_localcwld/metadata.py` 和对应单测；禁止修改 v2.6 原脚本。

公开函数建议：

```python
@dataclass(frozen=True)
class ParticleParameters:
    qbase: np.ndarray
    charge_mod: np.ndarray
    dpolar: np.ndarray
    is_polar: np.ndarray
    dens_source: np.ndarray
    dens_sink: np.ndarray
    source_class_weight: np.ndarray
    static_phase: np.ndarray
    residue_id: np.ndarray

def build_particle_parameters(system, topology, *, a_q2=0.5,
    ca_source_weight=2.0, enable_water_response=True,
    enable_solute_polarization=True) -> ParticleParameters: ...
```

步骤：

1. 从 v2.6 复制并最小化分类逻辑，保持常量集合和回退规则。
2. 把 `mol_ids` 对外重命名为 `residue_id`，数值仍为 `residue.index`。
3. 检查每个数组长度等于 `system.getNumParticles()`，所有浮点 finite。
4. 给水 O/H、Na/Cl/Ca、ASP/LYS、普通 O/N/S、脂质头和小配体写逐粒子断言。
5. 覆盖 aq0、aq0.5、water response off、solute response off、charge_mod 上下 clamp。

完成定义：新模块与 v2.6 对同一 1CKK system 的九个数组逐元素一致；测试报告每个数组最大差值；不得依赖当前工作目录。

### LCWLD-030：Reference Python、fixtures 与 Golden

前置：LCWLD-020。允许修改 `reference.py`、`tests/fixtures/*`、fixture builder 和测试。

必须建立：

- `two_particle_directional.json`：一个 source、一个 sink。
- `three_particle_chain.json`：同时产生 direct 与 chain force。
- `same_residue.json`：相同 residue density 为零。
- `excluded_pair.json`：explicit exclusion。
- `pbc_cross_boundary.json`：跨盒最小镜像。
- `water_ca_cluster.json`：TIP3P + Ca²⁺。
- `zmm_orders.json`：ell=1/2/3。

Reference API 必须能返回 `dens`、`Q`、`dQ_ddens`、`lambda`、`energy_direct`、`energy_penalty`、`force_direct`、`force_chain` 和总量。使用 float64；代码保持显式循环以便审计，不优化。

步骤：

1. 对每个 fixture 固定粒子顺序、box、参数、exclusions 和 SHA256 manifest。
2. 用解析公式实现 reference，不调用待开发插件。
3. 用 v2.6 CustomGBForce 对照相同 fixture。
4. 在 10,000 个距离点比较解析 density kernel 与 1024 点表格参考，完成 `DEC-002`。
5. Golden JSON 写生成器版本和公式版本，禁止手改数值。

完成定义：所有 fixture 可离线重建；重复生成 hash 相同；与 CustomGB 的差异报告完成；主 agent签署 `DEC-002` 后才可 DONE。

### LCWLD-040：C++ Public API

前置：LCWLD-010、030。只允许修改 `LocalCWLDForce.h/.cpp`、export header 和 API tests。

步骤：

1. 严格实现第 20 节 API，不加入分类、拓扑或设备状态。
2. 所有 setter 做 finite/range/index 检查。
3. `addParticle()` 返回连续索引。
4. exclusion 规范化为 `(min,max)`，拒绝 self 和重复 pair。
5. `usesPeriodicBoundaryConditions()` 返回 true。
6. `createImpl()` 暂时可以链接到 LCWLD-050 的声明，但不能自行实现 kernel。
7. 测试默认值、合法 round trip、非法参数异常和 copy/clone 行为。

完成定义：API tests 通过；公开头文件不包含 CUDA/OpenCL header；所有 public 方法有单位和更新语义注释。

### LCWLD-050：ForceImpl 与抽象 Kernel contract

前置：LCWLD-040。允许修改 `LocalCWLDKernels.h`、`LocalCWLDForceImpl.h/.cpp`。

Kernel contract：

```cpp
class CalcLocalCWLDForceKernel : public KernelImpl {
public:
    static std::string Name();
    virtual void initialize(const System&, const LocalCWLDForce&) = 0;
    virtual double execute(ContextImpl&, bool includeForces,
                           bool includeEnergy) = 0;
    virtual void copyParametersToContext(ContextImpl&,
                           const LocalCWLDForce&) = 0;
};
```

步骤：

1. `ForceImpl::initialize()` 验证粒子数和 periodic box 支持，再请求 kernel。
2. `calcForcesAndEnergy()` 尊重 force group、includeForces、includeEnergy。
3. `updateParametersInContext()` 只转发允许更新的每粒子参数。
4. 不在 ForceImpl 中写数值算法。
5. 写 fake kernel 测试验证 initialize/execute/update 的调用次数和参数。

完成定义：force group 不包含时贡献为零；includeEnergy=false 时返回值不被误加；错误 Context/粒子数给清晰异常。

### LCWLD-060：XML Serialization v1

前置：LCWLD-040。允许修改 serialization 文件与测试。

步骤：

1. 创建 `LocalCWLDForceProxy`，schema version 固定为 1。
2. 序列化 name、force group、所有 global、所有粒子参数和 exclusions。
3. 反序列化时验证 version、字段、索引和参数范围。
4. 注册 proxy，保证重复加载不会重复注册崩溃。
5. 测试 ell=1/2/3、penalty on/off、非默认 global、多个 exclusions。

完成定义：XML round-trip 后所有 getter 完全相同，Reference energy/force 在容差内；未知 version 必须失败而非忽略。

### LCWLD-070：Reference CPU Kernel

前置：LCWLD-030、050。只允许修改 `platforms/reference/*`。

实现步骤必须按固定顺序：

1. 初始化时复制参数和 exclusion set。
2. 每次 execute 获取位置、box 和 force buffer。
3. Pass A：有向 source→sink 累积 density。
4. Pass B：计算 Q 和 `dQ/ddens`。
5. Pass C：每个无序 pair 计算 energy、direct force 和双方 lambda。
6. penalty on 时累积能量和 lambda。
7. Pass D：每个 density pair 计算 chain force。
8. 仅在请求 energy 时累积能量，但请求 force 时始终完成必要中间量。
9. 使用 OpenMM 周期性最小镜像并支持合法 triclinic 盒；临时 orthorhombic-only 原型必须明确拒绝 tilted box，且不得进入 v0.1 release。

硬性测试：

- Reference vs Python fixture 的 dens/Q/E/F。
- source 与 sink 交换方向性。
- same-residue 的 density edge 为零；显式 exclusion 的 density、Local pair energy 和 force 均为零。
- PBC 平移不变。
- 总力接近零。
- 中心有限差分覆盖 source、sink 和普通固定 Q 粒子。

完成定义：CPU double 的 dens/Q 接近机器精度；能量与力达到第 9 节门槛；ASan/UBSan 无错误。

### LCWLD-080：Python SWIG 与系统 Builder

前置：LCWLD-020、040、060、070。允许修改 `python/*` 和 Python tests。

`build_local_cwld_system()` 必须：

1. 使用 XML serialization 复制输入 System，不原地修改。
2. 找到唯一 `NonbondedForce`，否则报错。
3. 调用 metadata builder 得到九个数组。
4. 将原 NB 设置为 `CutoffPeriodic`、`rc=1.2 nm`、RF dielectric=1、switch on、`r_on=0.9 nm`。
5. 创建 LocalCWLDForce，添加全局值、每粒子参数。
6. 将原 NB 每个 exception 的 `(p1,p2)` 复制为 Local exclusion；不复制 exception charge/sigma/epsilon。
7. 返回新 System 和可序列化 manifest；不创建 Integrator，不运行模拟。

SWIG 必须导入当前 OpenMM 的 SWIG headers，包名固定 `openmm_localcwld`。测试 clean import、API、XmlSerializer 和 Reference Context。

完成定义：同一个输入 System 构建两次结果一致；原 System 未变化；Local + modified NB 与 v2.6 总能量/力一致；错误输入清晰失败。

### LCWLD-090：Common Compute Host 层

前置：LCWLD-050、070。允许修改 `platforms/common/src/*`，禁止写设备数学。

步骤：

1. 通过 `ComputeContext`/`ComputeArray` 分配九个参数、dens、Q、dQ、lambda 和 exclusion 数据。
2. 添加 `LocalCWLDForceInfo`，保证 OpenMM 粒子重排序时参数映射正确。
3. 编译/绑定四个 device kernel：density、Q、pair/lambda、chain。
4. 参数永久留设备；execute 不读取 positions/Q 回 host。
5. `copyParametersToContext` 只上传允许更新的数据，并 invalidate 必要缓存。
6. 不包含 CUDA 或 OpenCL 专属 header。

完成定义：common host target 可在 CUDA 关闭时编译；静态检查无 `cuda.h`/`cl.h`；参数重排测试通过。

### LCWLD-100：Common/GPU Device Kernels

前置：LCWLD-030、090。只允许修改 `platforms/common/kernels/localCWLD.cc` 和 kernel 单测。

执行顺序：

1. 先写一线程一 sink 的 density kernel，正确优先。
2. 写 Q/dQ kernel。
3. 写 pair energy/direct force/lambda kernel；lambda 累积必须线程安全。
4. 写 chain force kernel。
5. 使用 Common Compute 宏和 `real/mixed` 类型；禁止 vendor-only 语法。
6. 先用简单全 pair 实现验证，再单独优化到 neighbor tile；优化提交不得和首次正确实现混在同一工单。
7. 每次优化前后运行固定 parity tests。

完成定义：2/16/128 粒子在 single/mixed/double 支持范围内与 Reference 一致；重复执行无竞态随机漂移；GPU 每步无 host synchronization。

### LCWLD-110：CUDA Plugin 注册与加载

前置：LCWLD-090、100。允许修改 `platforms/cuda/*` 和 CUDA CMake。

步骤：

1. 创建 CUDA kernel factory，注册 `CalcLocalCWLDForceKernel::Name()`。
2. 导出 `registerPlatforms()` 和 `registerKernelFactories()`。
3. 安装动态库到 OpenMM plugin directory。
4. 测试显式加载和 `OPENMM_PLUGIN_DIR` 自动加载。
5. 检查 `Platform.getPlatformByName('CUDA')` 后真实 platform 名和 device 属性。
6. CUDA 不存在时 configure 允许显式关闭；CUDA lane 中缺 CUDA 必须 fail/skip-with-reason，不能跑 CPU 冒充。

完成定义：CUDA Context 能创建，energy/force 与 Reference parity 通过，XML deserialize 后仍能运行。

### LCWLD-120：P0 数值测试总门禁

前置：LCWLD-070；CUDA 部分另依赖 LCWLD-110。允许修改 tests，不允许改实现公式。

必须实现并逐一命名：

- `test_metadata_particle_classification`
- `test_charge_mod_qref2_water_oxygen_only`
- `test_density_directionality`
- `test_same_residue_density_zero`
- `test_exclusion_zero_contribution`
- `test_q_nested_tanh_and_clamp`
- `test_zmm_orders_value_and_derivative`
- `test_rf_cancellation_term`
- `test_force_finite_difference_source`
- `test_force_finite_difference_sink`
- `test_force_finite_difference_penalty`
- `test_pbc_minimum_image`
- `test_translation_and_permutation_invariance`
- `test_serialization_roundtrip`
- `test_context_parameter_update`
- `test_reference_cuda_parity`

有限差分使用 `h=1e-4,1e-5,1e-6 nm` 扫描，必须展示收敛区间。近零分量用绝对误差，非零分量同时检查绝对和相对误差。任何 P0 失败都阻断后续动力学。

### LCWLD-130：NVE/NVT 动力学回归

前置：LCWLD-120 全通过。允许修改 integration tests 和报告脚本。

NVE：4–8 粒子 fixture，Verlet `dt=0.0005 ps`，10,000 steps；记录总能量相对漂移和 slope。硬门槛：相对漂移 ≤`1e-4`，slope ≤`1e-5 kJ/mol/ps/particle`，无 NaN/Inf。

NVT：LangevinMiddle，300 K、1/ps、0.002 ps、seed 1234/5678、100 ps；后 80 ps 平均温度 285–315 K，无持续漂移。不得使用 MTS。

失败排查顺序固定：有限差分 → cutoff/r_env 边界 → chain force → seed/precision → integrator。禁止通过增大 thermostat 摩擦掩盖力错误。

### LCWLD-140：1CKK 集成和科学回归

前置：LCWLD-080、120、130。允许修改 `examples/1ckk_minimal.py`、integration/science tests、manifest；禁止覆盖现有 DCD/CSV。

PR smoke：固定 `1CKK.pdb` hash；minimize≤200 iterations，NVT 1000、NPT 1000、production 10,000 steps；临时输出目录；seed=1234；LocalCWLD 默认参数；普通 LangevinMiddle。

Nightly：PME、aq0-w1、aq0-w2、aq0.5-w2，各 3 seeds。Release 才运行冻结的 5 ns 方案。

硬门槛：构建、最小化、运行成功；E/F finite；短跑温度 300±10 K，RMSD<1.0 nm；所有输出 metadata 完整。科学指标超过 golden 3σ 时阻断发布但不得自动改参数或 golden。

### LCWLD-150：性能 Benchmark

前置：LCWLD-110、120。允许修改 benchmarks 和报告，不允许改物理参数。

体系：合成 N=64/256/1024/4096 与 1CKK。每项 warmup=1000、measure=5000、重复 5 次取 median。记录 Context/JIT、steady-state、ns/day、kernel 分解、显存、device、precision。

同机比较：

1. PME/modified NB 无 Local force（理论上界）。
2. v2.6 CustomGB exact。
3. LocalCWLD Reference。
4. LocalCWLD CUDA mixed/double。

硬门槛：正确性先通过；GPU 不能出现每步 host copy；性能不得比同机 CustomGB 慢；v0.1 发布目标为 1CKK ≥2× CustomGB。未达到 2× 可标记“正确性 alpha”，不能标记“生产性能版”。

优化工单必须由 profiler 证据驱动，单独记录 kernel 时间占比；禁止通过降低 Q 更新频率取得速度。

### LCWLD-160：CI、结果 Schema、打包和发布

前置：所有前置 P0；release 还依赖 140/150。

CI lanes：

- `lint-build`: configure、C++/Python lint、无生成物。
- `unit-reference`: LCWLD-120 的 CPU double，目标 ≤5 min。
- `integration-reference`: NVE/NVT 和短 1CKK。
- `cuda-nightly`: mixed/double parity、动力学、benchmark。
- `science-nightly`: 3 seeds 差异报告，不覆盖 golden。
- `abi-matrix`: 支持的 OpenMM/Python 版本和 serialization。
- `release-gate`: 全 P0、science、performance、clean install。

每个运行的 JSON sidecar 必填：代码 SHA/dirty、插件/ABI/schema 版本、OpenMM/Python 版本、platform/device/precision、PDB/拓扑 hash、粒子参数 hash、box、global、exclusion 数、integrator/dt/friction/T/P/barostat、seed、run mode、报告间隔、UTC 时间、输出 hash。禁止写凭据。

发布门禁：clean environment 安装；plugin load；Python import；fixture/API/XML/Reference/CUDA smoke；CHANGELOG；兼容矩阵；包与 fixture SHA256。有限差分、NVE、PBC、serialization 任一失败不得发布。

## 25. 测试阈值速查

| 检查 | CPU Reference/double | CUDA mixed 初始门槛 |
|---|---:|---:|
| dens/Q | `rtol 1e-10` | `rtol 1e-5` |
| energy | `rtol 1e-9` | `rtol 1e-5` |
| force parity | `rtol 1e-7` | `rtol 1e-4` |
| finite difference | `abs 1e-5` 且 `rel 1e-4` | `abs 5e-4` 且 `rel 1e-3` |
| 总力 | `norm <=1e-8` | `norm <=1e-5` |
| PBC 等价 energy | `1e-10` | `1e-6` |
| NVE 相对漂移 | `<=1e-4` | 经 double 对照校准 |
| NVT 后段均温 | 285–315 K | 285–315 K |
| 性能 | 正确性 oracle | 不慢于 CustomGB；目标 ≥2× |

若 analytic density 与 v2.6 tabulated reference 的误差天然超过表中阈值，只能通过 `DEC-002` 明确兼容模式和新阈值；执行 agent 不得自行放宽。

## 26. 常见失败与固定排查顺序

### dens 不一致

1. 检查 target 是否 `dens_sink=1`。
2. 检查 source 是否 `dens_source=1`。
3. 检查只使用 source 的 class weight 和 charge_mod。
4. 检查 `residue_id`，不是 chain ID 或真正 molecule ID。
5. 检查 exclusion 是否也从 density 中排除。
6. 检查水 H 没有成为 source。
7. 检查 PBC 和 `r<r_env`。

### Q 不一致

1. 先确认 dens。
2. 检查 `static_phase*is_polar*dpolar`。
3. 检查内外两层 tanh。
4. 检查 q_delta_clamp 在外层乘、内层除。
5. 检查 qbase 最后相加，未被缩放。

### Energy 不一致但 Q 一致

1. 检查 ell 系数和 cutoff。
2. 检查 pair 是否重复计数。
3. 检查 RF cancellation 永远是 `(1/r-1/rc)`。
4. 检查 Local pair 没有额外 switching。
5. 检查 exclusions。
6. 检查 penalty 开关。

### Force 不一致但 energy 一致

1. 检查 `F=-dE/dx` 符号。
2. 检查 source 和 sink 两侧 chain force。
3. 检查 lambda 是否包含所有 pair 和 penalty。
4. 检查 `K'(r)` 的负号与 `r` 因子。
5. 用 source-only 位移 fixture 验证是否漏 `dQ/dR`。
6. 检查原子力累积竞态。

### 动力学升温

1. 立即确认未启用 MTS/低频 Q。
2. 回跑有限差分和 NVE。
3. 检查 cutoff/r_env 附近连续性。
4. 检查 CUDA mixed 与 Reference parity。
5. 不得先调 thermostat、dt 或参数掩盖问题。

### 性能差

1. 确认真的是 CUDA platform。
2. 分开 JIT 和 steady-state。
3. 查每步 host-device copy/sync。
4. profile 四个 passes。
5. 检查 inactive source/sink 是否仍做全量工作。
6. 再考虑 kernel fusion、compact active lists、neighbor tile；每次只做一种优化并跑完整 parity。

## 27. 主 Agent 验收清单

主 agent 合并任何工单前必须确认：

- 修改范围与工单所有权一致。
- 没有顺手修改公式、参数或旧实验脚本。
- 新 public API 有文档和非法输入测试。
- 数值报告同时给绝对/相对最大误差，不能只写“测试通过”。
- GPU 报告包含真实 platform/device/precision。
- 没有生成物、大轨迹、本机绝对路径或秘密。
- 失败测试没有被删、skip 或放宽。
- 工单的前置依赖确实 DONE。
- 对应文档、测试和实现一起更新。

## 28. 建议的首轮派发批次

第一批只派三个互不冲突的工单：

1. `LCWLD-000`：环境与决策冻结。
2. `LCWLD-020`：Python metadata 抽取，但暂不接插件。
3. `LCWLD-030` 的 fixture 设计部分；参考计算要等 020 的字段稳定后完成。

随后按顺序：

```text
010 → 040 → 050 → 070 → 080
030 ────────────────┘
070 → 090 → 100 → 110
070/110 → 120 → 130 → 140/150 → 160
```

不要第一天就让多个 agent 同时写 API、Reference 和 CUDA。最先需要冻结的是元数据、Golden 和 C++ API；GPU 优化必须最后建立在已经通过有限差分的 Reference 上。

## 29. 可复制的首张执行提示词

```text
你负责工单 LCWLD-020：独立 Python 元数据构建器。
工作区：/home/ruigengji/L-IPS/2.5。
只允许创建或修改：
- openmm-localcwld/python/openmm_localcwld/metadata.py
- openmm-localcwld/python/tests/test_metadata.py
不得修改 test_lips_vs_pmeV2.6.py 或任何 DCD/CSV/PNG。

先阅读 PLAN_LocalCWLDForce_OpenMM_Plugin.md 的第 15–18、24 节。
严格复现 test_lips_vs_pmeV2.6.py:242-380 的 exact 元数据语义。
特别注意：mol_id 实际是 residue.index；水 H 不是 source；qref2 优先只取 active water O；不要引入 q_driver 或 dynamic phase。

实现 frozen ParticleParameters dataclass 和 build_particle_parameters()。
为水、离子、ASP/LYS、普通极性原子、脂质头、配体、开关矩阵和 charge_mod clamp 写单测。
完成后报告修改文件、测试命令、通过数量、九个数组与原函数的逐元素最大误差。不得改变公式或容差。
```

## 30. 文档完成判定

这份计划现在可以作为低等级 agent 的实施输入。开始编码前仍必须由主 agent 完成 `LCWLD-000`，并确认目标计算节点的 OpenMM/CUDA 工具链；工具链未知不是执行 agent 自行猜测 ABI 或安装路径的理由。

## 31. Reference CPU 实现逐行级伪代码

本节用于 `LCWLD-070`。实现者应先照此写出朴素、可审计的 `O(N²)` Reference 版本。任何空间优化、SIMD 或线程化都留到 Reference 正确性通过以后。

### 31.1 建议的内部结构

```cpp
struct ParticleData {
    double qbase;
    double chargeMod;
    double dpolar;
    double isPolar;
    double densSource;
    double densSink;
    double sourceClassWeight;
    double staticPhase;
    int residueId;
};

struct PairKey {
    int first;
    int second;
};

struct Workspace {
    std::vector<double> density;
    std::vector<double> charge;
    std::vector<double> dChargeDDensity;
    std::vector<double> lambda;
};
```

`Workspace` 必须在 kernel 对象中复用，不能每步重复申请；但 Reference 初版可以先追求清晰。每次 execute 开始必须将 `density` 和 `lambda` 清零。`charge` 与 `dChargeDDensity` 每个粒子必须重写，不能依赖上一步值。

exclusion 建议初始化为：

```cpp
std::unordered_set<uint64_t> exclusions;
uint64_t key(int i, int j) {
    int a = std::min(i, j);
    int b = std::max(i, j);
    return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
}
```

禁止在内层循环线性扫描 exclusion vector。

### 31.2 公共 pair 几何辅助函数

Reference 中只允许一个函数负责 pair displacement，density、direct 和 chain 三处都调用它：

```cpp
PairGeometry geometry(i, j, positions, box) {
    // delta 必须统一定义为 position[j]-position[i] 的最小镜像。
    Vec3 delta = minimumImage(positions[j]-positions[i], box);
    double r2 = dot(delta, delta);
    if (r2 <= 0)
        throw OpenMMException("LocalCWLDForce encountered zero pair distance");
    double r = sqrt(r2);
    Vec3 unit = delta/r;  // 从 i 指向 j
    return {delta, unit, r, r2};
}
```

若决定允许重合粒子，则必须在独立决策中定义能量/力；v0.1 建议明确抛错，避免 `1/r` 产生 NaN。

### 31.3 execute 顶层流程

```cpp
double execute(context, includeForces, includeEnergy) {
    load positions and periodic box;
    resize workspace if needed;
    fill(density, 0);
    fill(lambda, 0);

    computeDensity();
    computeChargesAndDerivatives();
    double energy = computePairEnergyDirectForceAndLambda(
        includeForces, includeEnergy);
    energy += computePenaltyAndLambda(includeEnergy);
    if (includeForces)
        computeChainForce();

    if (!includeEnergy)
        return 0.0;
    return energy;
}
```

即使 `includeEnergy=false`，若请求 force，Pass C 仍必须计算 `lambda`，否则 chain force 缺失。若 `includeForces=false` 且 `includeEnergy=true`，可以跳过 direct/chain force 累积，但仍要算 Q 和 energy。若二者均 false，可尽早返回零。

### 31.4 Pass A：Density

```cpp
for i in [0, N):
    if particles[i].densSink == 0:
        density[i] = 0
        continue

    sum = 0
    for j in [0, N):
        if i == j: continue
        if particles[j].densSource == 0: continue
        if particles[i].residueId == particles[j].residueId: continue
        if excluded(i,j): continue

        geom = geometry(i,j)
        if geom.r >= rEnv: continue

        x = geom.r/rEnv
        kernel = (1-x*x)*(1-x*x)
        sourceAmplitude = particles[j].densSource
                        * particles[j].sourceClassWeight
                        * particles[j].chargeMod
        sum += sourceAmplitude*kernel

    density[i] = particles[i].densSink*sum
```

虽然许多 mask 是 0/1，实现中仍按浮点乘法保留一般权重语义。不要用 `bool` 截断非整数值。

Pass A 结束后的 debug 断言：

- 所有 density finite。
- `densSink=0` 的 density 必须严格为零。
- 没有 source 的 fixture 全部为零。
- 交换 source/sink 参数后结果按方向改变。

### 31.5 Pass B：Q 和导数

```cpp
for i in [0, N):
    A = staticPhase[i]*isPolar[i]*dpolar[i]
    t = tanh(kPolar*density[i]/rho0)
    y = A*t/qDeltaClamp
    outer = tanh(y)

    Q[i] = qbase[i] + qDeltaClamp*outer
    dQdDens[i] = A*(kPolar/rho0)
                 *(1-outer*outer)
                 *(1-t*t)
```

结束断言：

- Q、导数 finite。
- `abs(Q-qbase) <= qDeltaClamp + tolerance`。
- `A=0` 时 Q 必须等于 qbase，导数必须为零。
- 不要用 `1/cosh²`，大输入下容易 overflow；使用 `1-tanh²`。

### 31.6 Pass C：Pair energy、direct force、lambda

```cpp
energy = 0
for i in [0, N):
    for j in [i+1, N):
        if excluded(i,j): continue
        geom = geometry(i,j)
        if geom.r >= rc: continue

        U, dUdr = zmmClosureAndDerivative(order, geom.r, rc)
        shiftedRF = 1/geom.r - 1/rc

        pairEnergy = C*(Q[i]*Q[j]*U
                     - qbase[i]*qbase[j]*shiftedRF)
        if includeEnergy:
            energy += pairEnergy

        // 无论 includeEnergy 如何，只要 includeForces=true 就需要 lambda。
        lambda[i] += C*Q[j]*U
        lambda[j] += C*Q[i]*U

        if includeForces:
            dEdr = C*(Q[i]*Q[j]*dUdr
                    + qbase[i]*qbase[j]/geom.r2)
            forceOnI = dEdr*geom.unit
            forces[i] += forceOnI
            forces[j] -= forceOnI
```

注意：如果调用者请求 energy 而不请求 force，penalty energy 仍要加入；如果只请求 force，lambda 必须算。为减少分支错误，Reference 初版允许总是计算 lambda。

### 31.7 Penalty

```cpp
if useQPenalty:
    for i:
        delta = Q[i]-qbase[i]
        if includeEnergy:
            energy += 0.5*qPenaltyStrength*delta*delta
        lambda[i] += qPenaltyStrength*delta
```

Penalty 的 lambda 必须在 Pass D 之前加入。

### 31.8 Pass D：Density chain force

用无序 pair 同时处理 `j→i` 与 `i→j` 两个有向 density 贡献：

```cpp
for i in [0, N):
    for j in [i+1, N):
        if excluded(i,j): continue
        if residueId[i] == residueId[j]: continue

        geom = geometry(i,j)
        if geom.r >= rEnv: continue

        x2 = geom.r2/(rEnv*rEnv)
        dKernelDr = -4*geom.r/(rEnv*rEnv)*(1-x2)

        Bi = densSource[i]*sourceClassWeight[i]*chargeMod[i]
        Bj = densSource[j]*sourceClassWeight[j]*chargeMod[j]
        gi = lambda[i]*dQdDens[i]
        gj = lambda[j]*dQdDens[j]

        dEdrChain = (gi*densSink[i]*Bj
                   + gj*densSink[j]*Bi)*dKernelDr
        forceOnI = dEdrChain*geom.unit
        forces[i] += forceOnI
        forces[j] -= forceOnI
```

这里最容易出现四类错误：

1. 只算 `gi*Bj`，漏掉反向 `gj*Bi`。
2. 又乘一次 `densSink`，而 density 或导数中已错误重复包含。
3. `K'(r)` 漏掉负号或漏掉 `r`。
4. force unit vector 定义与 Pass C 不一致，导致 direct 正确、chain 符号相反。

### 31.9 Reference 强制诊断模式

Reference 测试构建可以提供非公开的 test hook，输出四个 workspace 数组和 direct/chain force 分解。生产 public API 不暴露这些内部数组，避免 ABI 被调试接口锁死。建议测试 hook 只在 `LOCALCWLD_BUILD_TESTING=ON` 时编译。

诊断 CSV/JSON 每个粒子至少包含：

```text
index, residue_id, qbase, density, Q, dQ_ddensity,
lambda, force_direct_x/y/z, force_chain_x/y/z
```

## 32. ZMM closure 解析导数清单

实现者不得临场手推。冻结导数为：

```text
dU1/dr = -1/r² + r/rc³

dU2/dr = -1/r² + 5r/(2rc³) - 3r³/(2rc⁵)

dU3/dr = -1/r² + 35r/(8rc³)
          - 21r³/(4rc⁵) + 15r⁵/(8rc⁷)
```

单测必须在随机 `r∈(0.05,rc)` 上与高精度中心差分对照，并在 `r→rc` 检查 `U(rc)` 与 `dU/dr(rc)`。不得只测默认 ell=2。

## 33. 设备数据布局与 Kernel ABI

本节用于 `LCWLD-090/100`。第一版设备数组建议采用 SoA，而不是 `struct ParticleData[]`：

```text
qbase[N]                real/mixed
chargeMod[N]
dpolar[N]
isPolar[N]
densSource[N]
densSink[N]
sourceClassWeight[N]
staticPhase[N]
residueId[N]            int

density[N]              mixed
dynamicCharge[N]        mixed
dChargeDDensity[N]      mixed
lambda[N]                mixed
forceBuffer[3N]          使用 OpenMM 规定的定点/原子累积格式
```

原因：各 pass 访问字段子集不同，SoA 更容易合并访问，也方便 OpenMM particle reordering。所有参数数组必须随 Context 重排；`residueId` 也不能遗漏。

### 33.1 Kernel A：`computeLocalCWLDDensity`

输入：positions、box、邻居 tile/exclusion、`rEnv²`、source/sink 数组、residueId。输出：density。

启动前：density 必须清零。推荐每个 sink thread 累积本地 sum 后单写，避免 density atomic；若使用 tile pair 同时处理多个 sink，则需要安全累积。

硬性行为：

- `r² < rEnv²`，边界比较与 Reference 统一。
- 同 residue 跳过。
- exclusions 跳过。
- 使用 source `j` 的三个乘数。
- 若用无序 pair tile，必须分别判断并累积 `j→i` 和 `i→j`。

### 33.2 Kernel B：`computeLocalCWLDCharge`

输入：density 和 per-particle response arrays；输出 Q、dQ/dDensity，顺便将 lambda 清零。

每个粒子独立，无 atomic。固定 Q 粒子也必须写 Q=qbase 和 derivative=0，防止读取旧 step 数据。

### 33.3 Kernel C：`computeLocalCWLDPair`

输入：positions、Q/qbase、closure globals、exclusions；输出 direct force、可选 energy、lambda。

lambda 有并发累积，初版可使用 atomic。force 累积必须遵循 OpenMM Common Compute 的固定点 force buffer 规则，不能自己建立未经 Context 合并的 float force 数组。Energy 使用平台推荐的 reduction buffer。

如果 `includeEnergy=false`，可禁用 energy atomic/reduction；但只要需要 force，lambda 仍必须计算。

### 33.4 Kernel D：`computeLocalCWLDChain`

输入：positions、lambda、dQ/dDensity、source/sink 参数；输出 chain force。使用与 Kernel A 相同的 density pair 条件。不得读 host Q。

### 33.5 固定 launch 顺序

```text
clear density/energy/lambda as required
A density
B Q + dQ + clear lambda
C pair direct + lambda + optional energy
C2 penalty + lambda（可融合进 B 或独立，但结果需一致）
D chain force
```

不能把 D 提到 C 前面，因为它依赖完整 lambda。可以在 profiling 后融合 B/penalty 或 C/penalty；不得融合到难以逐阶段诊断后再首次验证。

## 34. 邻居表策略分三步实施

### Step 1：正确性 GPU 原型

允许简单全 pair，仅用于 2–128 粒子 fixture。禁止拿它跑 1CKK 性能并宣布失败或成功。

### Step 2：单一 `rc` tile list

使用 `rc=1.2 nm` 的邻居 tile。Kernel C 处理全部 tile；Kernel A/D 在同一 tile 内额外判断 `r<r_env`。优点是只维护一套邻居结构，先作为默认方案。

### Step 3：性能实验

仅在 profiler 证明 A/D 扫描大量无效 `rc` pair 后，实验第二套 `r_env` compact list 或 active source/sink list。必须分别报告：

- 构建第二邻居表的成本；
- A/D kernel 节省；
- 内存增加；
- 总 step time 改善；
- 不同体系/活性位点比例的收益。

没有端到端收益就撤回，不保留复杂路径。

## 35. OpenMM 重排序与 exclusion 规则

`LocalCWLDForceInfo` 必须让 OpenMM 知道哪些粒子参数决定等价性。重排序后所有九个数组、force buffer index、exclusion index 和 residue ID 必须一致映射。

最小重排序测试：

1. 构建具有重复/不同参数的 16 粒子系统。
2. 记录 Reference E/F。
3. 在允许重排序的平台运行多个 step 触发 reorder。
4. 比较 E/F 与重排前和 Reference。
5. 更新一个粒子参数，再比较新 Context 与 `updateParametersInContext()`。

exclusion 的单一事实来源是 Force 对象的规范化 pair 列表。设备端不得另从 bond topology 推断 exclusions。

## 36. 数值误差报告标准

每个 parity 测试失败时必须打印足够信息，不得只有 `assert false`：

```text
fixture=
platform=
precision=
particle_count=
zmm_order=
r_env/rc=
max_abs_density_error= at particle
max_rel_density_error= at particle
max_abs_Q_error= at particle
max_abs_energy_error=
max_abs_force_error= at particle/component
max_rel_force_error= at particle/component
reference_value=
actual_value=
seed=
```

相对误差分母统一为：

```text
max(abs(reference), abs(actual), scale_floor)
```

建议 `scale_floor`：density/Q `1e-12`，energy `1e-10 kJ/mol`，force `1e-8 kJ/mol/nm`。近零值必须以绝对误差判定。

## 37. Gradient 检查操作规程

每个 fixture 的有限差分执行：

1. 选远离 `r_env`、`rc` 边界的坐标作为主 gradient test。
2. 对每个可移动粒子的 x/y/z 分量分别计算。
3. 使用 `h=1e-4,1e-5,1e-6 nm`。
4. 计算 `F_fd=-(E(x+h)-E(x-h))/(2h)`。
5. 至少两个相邻 h 的误差应下降或进入平台噪声平台。
6. 单独运行 direct-only fixture 和 response-on fixture，确认 chain 项确实被测试。
7. cutoff 边界测试不与普通 gradient 容差混在一起。

若 gradient 失败，禁止立即调 h 或容差；先输出 direct/chain 分解和 density/Q/lambda。

## 38. Builder 的系统能量组合审计

LocalCWLDForce 单独能量不是完整物理模型。`build_local_cwld_system()` 必须留下可审计 manifest：

```json
{
  "base_nonbonded": {
    "method": "CutoffPeriodic",
    "cutoff_nm": 1.2,
    "reaction_field_dielectric": 1.0,
    "switching": true,
    "switch_distance_nm": 0.9
  },
  "localcwld": {
    "zmm_order": 2,
    "r_env_nm": 0.35,
    "cutoff_nm": 1.2,
    "particle_parameter_sha256": "...",
    "exclusion_sha256": "..."
  }
}
```

审计测试必须按 force group 分解：

- 原系统 PME energy；
- 修改后 NB energy；
- LocalCWLD correction；
- 修改后 NB + LocalCWLD 总能量；
- v2.6 CustomGB 对照总能量。

测试目的不是要求新模型等于 PME，而是确保新插件组合与 v2.6 组合一致，且没有 RF cancellation 双算/漏算。

## 39. 可复制的工单派发卡包

下面每张卡已经压缩为可直接交给执行 agent 的文本。主 agent 派发时只需补充实际工作树、目标分支和已完成依赖，不要删减红线。

### 派发卡 LCWLD-000

```text
你负责 LCWLD-000：工具链与设计决策冻结。只允许修改 docs/decisions/DEC-001..003。不得写插件代码或安装软件。记录目标计算节点的 OpenMM/Python/CMake/C++/CUDA/driver、插件路径和精度支持；未知项写 UNKNOWN。确认 v0.1 平台范围、源码位置、analytic-vs-tabulated 决策待办。交付三份决策文档、证据命令和阻塞项。不得猜 ABI。
```

### 派发卡 LCWLD-010

```text
你负责 LCWLD-010：CMake/仓库骨架。依赖 LCWLD-000 DONE。只拥有根 CMake、cmake/、README、LICENSE、.gitignore 和空子目录 CMake；不得实现公式。要求 CUDA/Python/Reference 均可显式开关，CUDA 关闭时 CPU-only configure/build/install 成功。不得硬编码个人或集群路径。交付 configure/build/install 命令、安装树和无生成物证明。
```

### 派发卡 LCWLD-020

使用第 29 节完整提示词。

### 派发卡 LCWLD-030

```text
你负责 LCWLD-030：Python float64 reference 与 golden fixtures。依赖 metadata 字段冻结。只拥有 reference.py、fixture builder、tests/fixtures 和 DEC-002 的实验数据；不得修改 C++ 或 v2.6。实现显式循环，输出 dens/Q/dQ/lambda/direct/chain/E/F。建立七个指定 fixture、manifest、SHA256；对照 v2.6 CustomGB，并完成 analytic-vs-tabulated 10,000 点报告。Golden 只能由生成器写，不能手改。交付最大绝对/相对误差和主 agent 待签署决策。
```

### 派发卡 LCWLD-040

```text
你负责 LCWLD-040：LocalCWLDForce C++ public API。依赖 CMake 和 fixture contract。只修改 public header/source、export header、API tests；不得写数值 kernel。严格实现第20节方法、参数验证、粒子/exclusion round-trip、PBC声明。拒绝 NaN、无效 cutoff/order/index、self/duplicate exclusion。公共 header 不得包含平台 header。交付 API tests 和单位文档；不得新增未经计划批准的方法。
```

### 派发卡 LCWLD-050

```text
你负责 LCWLD-050：ForceImpl 与 CalcLocalCWLDForceKernel contract。只修改三个 internal/kernel 文件和 fake-kernel tests。ForceImpl 只校验/转发，不计算物理。正确处理 force group、includeForces/includeEnergy 和 updateParametersInContext。用 fake kernel 证明 initialize/execute/update 调用。不得接触 Reference/CUDA 实现。
```

### 派发卡 LCWLD-060

```text
你负责 LCWLD-060：XML serialization v1。只修改 serialization 源和测试。序列化所有 global、九个粒子字段、exclusions、force name/group，显式 version=1；拒绝未知版本和非法数据；保证 proxy 注册幂等。不得序列化 Context、ForceImpl、workspace 或设备缓存。交付 XML round-trip getter 比较和 Reference E/F 比较。
```

### 派发卡 LCWLD-070

```text
你负责 LCWLD-070：Reference CPU kernel。先阅读第18、31、32节。只修改 platforms/reference。按 Pass A density、B Q/dQ、C E/direct/lambda、Penalty、D chain 顺序实现朴素 double O(N²)。统一最小镜像和 force unit-vector 约定。禁止优化、线程化、冻结Q或改变公式。运行全部 fixture、PBC、exclusion、总力和 source/sink finite-difference。交付逐阶段最大误差及 ASan/UBSan 结果。
```

### 派发卡 LCWLD-080

```text
你负责 LCWLD-080：Python SWIG 和 build_local_cwld_system。只修改 python/。Builder 必须复制 System，不原地修改；设置唯一 NonbondedForce 为 CutoffPeriodic/RF=1/switch=.9/cutoff=1.2；用 metadata 添加 Local force；只复制 exception pair 为 Local exclusions。不得创建 Integrator、运行MD或依赖cwd。交付 clean import、XML、Reference Context、原System未变和 v2.6 总E/F parity。
```

### 派发卡 LCWLD-090

```text
你负责 LCWLD-090：Common Compute host。只修改 platforms/common/src。分配/上传 SoA 参数和四个 workspace 数组，注册 ForceInfo 以支持重排序，绑定四个 kernel，保证每步不回读 host。不得写设备数学或包含 CUDA/OpenCL header。交付 CUDA关闭时的编译、参数更新和重排序测试。
```

### 派发卡 LCWLD-100

```text
你负责 LCWLD-100：Common device kernels。只修改 localCWLD.cc 和其测试。先实现2–128粒子全pair正确性版本，再另提优化。四kernel顺序固定；Q/derivative每步覆盖；lambda安全累积；force用OpenMM规定buffer；不得host同步或vendor-only语法。运行single/mixed/double parity和重复竞态测试。未通过Reference parity不得做neighbor优化。
```

### 派发卡 LCWLD-110

```text
你负责 LCWLD-110：CUDA factory/plugin/load。只修改 platforms/cuda及CUDA CMake。注册冻结kernel名，导出标准注册函数，安装到lib/plugins，测试显式与OPENMM_PLUGIN_DIR加载。报告真实CUDA platform/device/precision；CUDA缺失时明确BLOCKED/SKIP，严禁CPU回退冒充。交付CUDA E/F parity和XML反序列化运行。
```

### 派发卡 LCWLD-120

```text
你负责 LCWLD-120：P0数值门禁。只修改tests，不得改实现、golden或阈值。实现第24节列出的16个命名测试，finite difference用三个h并报告收敛。失败必须输出fixture/platform/precision/index/component/reference/actual/abs/rel。任何P0失败均交付FAIL证据，不能skip或放宽容差。
```

### 派发卡 LCWLD-130

```text
你负责 LCWLD-130：NVE/NVT回归。前置P0全通过。只修改integration tests/report。NVE与NVT使用冻结dt、seed和步数；禁止MTS。记录每步/分块E、T、maxF、Q范围和platform。若升温，按有限差分→边界→chain→precision顺序排查，不得调thermostat掩盖。交付原始CSV、摘要JSON和阈值判定。
```

### 派发卡 LCWLD-140

```text
你负责 LCWLD-140：1CKK smoke/science。只使用临时输出，不读取或覆盖仓库旧DCD/CSV。固定PDB hash、参数、seed和普通Langevin。先跑短smoke；nightly/release长作业需主agent授权。输出必须带manifest sidecar。科学指标超golden只报告和阻断，不自动改参数/golden。交付PME/CustomGB/Local对照及所有输入输出hash。
```

### 派发卡 LCWLD-150

```text
你负责 LCWLD-150：性能benchmark。正确性必须先PASS。只修改benchmarks/report，不改模型、更新频率、dt或精度来取巧。同机测no-Local、CustomGB、Reference、CUDA；warmup1000、measure5000、重复5次median。验证真实device，分开JIT/steady state，记录kernel、VRAM和host sync。未到2x标alpha，不能改阈值。
```

### 派发卡 LCWLD-160

```text
你负责 LCWLD-160：CI/schema/package/release gate。不得修公式来使CI通过。建立lint、unit-reference、integration、cuda-nightly、science-nightly、ABI、release lanes；缺硬件明确SKIP且P0不视为release-ready。验证clean install/import/plugin load/XML/smoke、metadata schema、包hash和rollback。只有零open P0且证据可重放才可签署release。
```

## 40. 通用失败证据格式

任何失败也是交付物。统一创建安全的 YAML：

```yaml
status: FAIL            # FAIL | BLOCKED | SKIP
severity: P0
work_item: LCWLD-120
command: "ctest --test-dir build -R finite_difference --output-on-failure"
cwd: "/absolute/worktree"
git:
  sha: "..."
  dirty: true
runtime:
  os: "..."
  compiler: "..."
  python: "..."
  openmm: "..."
  platform: "Reference"
  device: "CPU"
  precision: "double"
inputs:
  - path: "tests/fixtures/three_particle_chain.json"
    sha256: "..."
    seed: 1234
expected:
  value: 1.234
  unit: "kJ/mol/nm"
  tolerance_abs: 1.0e-5
actual:
  value: 1.240
  delta_abs: 0.006
  delta_rel: 0.00486
exit_code: 1
elapsed_seconds: 0.42
first_bad_particle: 1
first_bad_component: "x"
log_path: "artifacts/LCWLD-120/failure.log"
minimal_reproduction: "..."
classification: "code" # code|fixture|environment|dependency|flaky|unknown
suspected_layer: "chain-force"
next_action: "compare lambda and Kprime for pair 0-1"
next_owner: "LCWLD-070 owner"
```

聊天或公开报告不得包含 token、私钥或完整本机凭据。必要环境只记录版本和安全 hash。

## 41. 工单 Review 和交接模板

```markdown
# Review LCWLD-XXX

- Owner:
- Reviewer:
- Dependencies confirmed DONE:
- Owned files:
- Out-of-scope files touched: none / list

| Requirement | Threshold | Actual | Evidence | Status |
|---|---:|---:|---|---|
| ... | ... | ... | path + SHA256 | PASS/FAIL/BLOCKED |

## Reproduction

```text
exact commands with exit codes and elapsed time
```

## Numerical evidence

- max abs/rel dens error:
- max abs/rel Q error:
- energy error:
- max force particle/component/error:
- finite-difference h convergence:
- platform/device/precision:

## Decision

APPROVE / REQUEST_CHANGES / BLOCKED

## Handoff

- Modified files:
- Artifacts and hashes:
- Known limitations:
- Unverified platforms:
- Next work item/owner:
- API/formula/golden/tolerance changed: NO
```

实现者不得审批自己的 API、golden、阈值或 release。此类变更需要主 agent 和另一名 reviewer；旧/新数值、原因和重跑命令必须一起保留。

## 42. Git 与提交拆分规则

建议每张工单至少拆成：

1. `test:` fixture/失败测试或 contract 测试；
2. `feat:` 最小实现；
3. `docs:` API/decision/evidence；
4. `perf:` 仅在正确性提交之后独立优化。

禁止将 golden 更新、公式变化和性能优化混在一个提交。禁止提交 DCD、build、stage、`.so/.dll`、SWIG 生成物、CUDA 编码源、`__pycache__`。

每个 PR/变更集说明：

```text
Work item:
Scientific semantics changed: no
Public API changed: yes/no
Golden changed: yes/no
Platforms actually tested:
Commands and results:
Maximum numerical differences:
Performance before/after:
Unverified assumptions:
Rollback method:
```

## 43. 结果与 Artifact 目录规范

所有测试和 benchmark 输出写入源码树外或被忽略的：

```text
artifacts/<work-item>/<UTC-run-id>/
├─ manifest.json
├─ commands.txt
├─ environment.json
├─ stdout.log
├─ stderr.log
├─ metrics.json
├─ result.csv
├─ failure.yaml            # 仅失败时
└─ sha256sums.txt
```

`manifest.json` 至少包含：schema version、git SHA/dirty、fixture hash、参数 hash、exclusion hash、OpenMM/plugin版本、platform/device/precision、seed、单位、命令、UTC时间和输出 hash。

同名 artifact 目录已存在时必须新建 run ID，不得覆盖。聚合器发现 manifest 不匹配时必须非零退出，禁止静默拼接。

## 44. API/ABI/Schema 版本规则

- Public API 使用 semantic version。
- C++ ABI major 与 OpenMM 支持矩阵绑定；ABI 不兼容必须在加载时清晰失败。
- XML `LocalCWLDForce` schema v1 独立于包版本。
- Result schema 使用 `localcwld-result-v1`。
- Particle parameter 顺序改变属于 ABI/schema breaking change。
- 默认公式、closure 或 residue exclusion 语义改变属于 scientific-model change，即使 API 没变也必须升模型版本。
- 性能优化且 E/F 在冻结容差内不升模型版本。

建议在 manifest 同时写：

```json
{
  "package_version": "0.1.0",
  "abi_version": 1,
  "force_schema_version": 1,
  "result_schema_version": "localcwld-result-v1",
  "model_semantics_version": "v2.6-compatible-1"
}
```

## 45. 发布与回滚操作清单

发布候选必须依次完成：

1. clean checkout/configure/build/install；
2. API/serialization/Reference P0；
3. CUDA P0（若该发布宣称 CUDA 支持）；
4. NVE/NVT；
5. 1CKK smoke；
6. science regression；
7. benchmark 与内存；
8. clean Python environment import/example；
9. 包、fixture、报告 SHA256；
10. known limitations 和兼容矩阵；
11. 上一稳定版本的安装包与回滚说明。

立即阻断/回滚条件：

- dens/Q/E/F、有限差分、PBC 或 exclusion P0 回归；
- NVE 异常漂移、NVT 插件特有升温或 NaN；
- ABI/schema 无声不兼容；
- CUDA 实际回退 CPU；
- 同硬件 steady-state 性能退化超过 20%；
- artifact 无法重放或 metadata 缺失；
- 包含凭据或不可接受的本机路径。

核心 P0 不允许 waiver。非核心 P1 waiver 必须写批准者、原因、补偿检查和到期日期。

## 46. 手算双粒子 Golden 示例

这个例子用于最早期 smoke，帮助实现者在完整 fixture 框架之前发现公式、方向和符号错误。它不是科学体系。

参数：

```text
particle 0: qbase=-0.8, sink=1, source=0,
            dpolar=-0.15, isPolar=1, staticPhase=1,
            sourceClassWeight=0, chargeMod=1, residueId=0
particle 1: qbase=+2.0, sink=0, source=1,
            dpolar=0, isPolar=0, staticPhase=0,
            sourceClassWeight=2.0, chargeMod=1, residueId=1
r=0.2 nm, r_env=0.35 nm, rc=1.2 nm,
ell=2, k_polar=0.8, rho0=13.5, qclip=0.2,
C=138.935458, no exclusion, no Q penalty
```

float64 参考值：

```text
K(r)                  = 0.4535610162432318
density[0]            = 0.9071220324864636
density[1]            = 0
tanh(kD/rho0)         = 0.05370366156716889
outer argument y      = -0.04027774617537666
Q[0]                  = -0.8080511958960022
Q[1]                  = 2.0
dQ0/dDensity0         = -0.008848889303412277
U2(r)                 = 3.466194058641975
dU2/dr                = -24.71547067901234
pair energy           = 147.95822647637223 kJ/mol
dE/dr direct          = -7.9566560344210195 kJ/mol/nm
dE/dQ0 (adjoint)      = 963.1545181086034 kJ/mol/e
K'(r)                 = -4.398167430237401 nm^-1
dE/dr chain           = 74.96982244683544 kJ/mol/nm
dE/dr total           = 67.01316641241442 kJ/mol/nm
```

若 unit vector 定义为从 particle 0 指向 particle 1，则：

```text
F0 = +(dE/dr total) * unit01
F1 = -F0
```

中心差分 `dE/dr`：

```text
h=1e-3  -> 67.01632775194355
h=1e-4  -> 67.01319802473904
h=1e-5  -> 67.01316672490520
h=1e-6  -> 67.01316641510857
```

必须再构造以下变体：

- 两粒子 `residueId` 相同：density 和全部 Local pair contribution 都为零还是仅 density？注意：同 residue 条件只写在 density expression；Local pair exclusion 由 explicit exclusions 控制。因此**相同 residue 只令 density contribution 为零，不能自动令 Local pair energy 为零**。若它们同时是原 NB exception，才因 Local exclusion 而 pair energy 也为零。
- 添加 explicit exclusion：density、Local pair energy、direct/chain force 全部为零。
- 将 particle 1 的 source weight 从 2 改 1：density 减半，但 pair Q-Q 仍通过改变后的 Q 间接变化。
- 将 particle 0 的 source weight 改大：它不是 source，不应改变当前方向的 density。

这一段纠正一个常见误读：`same residue` 条件属于 density computed value；不能凭直觉扩展为全部 Local pair exclusion。

## 47. PBC 与盒形支持的冻结要求

为与 OpenMM `CutoffPeriodic CustomGBForce` 参考语义一致，生产 v0.1 应支持 OpenMM 合法的周期盒，包括 triclinic。Reference 优先调用 OpenMM 已有的周期 displacement 辅助逻辑，Common Compute 使用平台提供的 periodic delta 宏/参数。

不允许只读取 box 对角线并声称通用 PBC。若早期 Reference 里程碑临时只支持 orthorhombic：

1. 必须在 `DEC-003` 标为临时限制；
2. 遇到 tilted box 明确抛错；
3. 不得进入 v0.1 release；
4. CUDA 与 Reference 最终必须有同样支持范围。

triclinic 测试至少使用一个上三角盒，比较：原构型、平移一个完整 lattice vector、不同 periodic image 的 E/F。

## 48. `includeEnergy/includeForces` 真值表

| includeEnergy | includeForces | 必须执行 | 可跳过 | 返回 |
|---|---|---|---|---|
| false | false | 无 | A/B/C/D 全部 | 0 |
| true | false | A、B、C energy、penalty energy | direct force、lambda、D | energy |
| false | true | A、B、C direct+lambda、penalty lambda、D | energy reduction | 0 |
| true | true | A、B、C 全部、penalty、D | 无 | energy |

Reference 可以为清晰多算少量中间值；GPU 必须最终避免不必要的 energy reduction，但不能因 `includeEnergy=false` 漏掉 lambda。

内部 adjoint 建议命名 `chargeGradient`，不要在 public API 中叫 `lambda`，避免与 ABFE alchemical lambda 混淆。

## 49. Same-residue 与 Explicit-exclusion 语义表

| 条件 | Density edge | Local pair E/direct | Chain force |
|---|---|---|---|
| `i==j` | 无 | 不枚举 self | 无 |
| same residue, not excluded | 无 | 有，若 `r<rc` | 无该 density edge |
| different residue, excluded | 无 | 无 | 无 |
| different residue, not excluded, `r<r_env` | 有向 edge 按 masks | 有 | 有，若响应导数非零 |
| different residue, not excluded, `r_env<=r<rc` | 无 | 有 | 无 |
| `r>=rc` | 无 | 无 | 无 |

所有实现与测试都以此表为准。计划前文出现的“同 residue/self 与 exception pair 均无 Local pair”表述若有，必须以本表纠正：同 residue 本身不会调用 `addExclusion()`，只影响 density；真正 Local pair exclusion 来自原 NonbondedForce exception 列表。

## 50. 逐文件 Definition of Done

| 文件/模块 | 完成定义 |
|---|---|
| `metadata.py` | 与 v2.6 九数组逐元素一致；无 cwd 依赖 |
| `reference.py` | 返回阶段诊断；golden/FD/PBC 通过 |
| `LocalCWLDForce.h/.cpp` | API/校验/单位文档/非法输入测试通过 |
| `LocalCWLDForceImpl.*` | kernel 转发、force group、update contract 通过 |
| serialization | schema v1 全字段 round-trip；未知版本失败 |
| Reference kernel | 四 pass、double parity、FD、triclinic、sanitizer 通过 |
| SWIG | clean env import，API/XML/Context 可用 |
| builder | 不修改输入；NB+Local组合与 v2.6一致；manifest完整 |
| Common host | SoA、reorder、无vendor header、无每步host回读 |
| device kernels | precision parity、无竞态、四pass诊断一致 |
| CUDA plugin | factory/load/install/真实device/parity 通过 |
| P0 tests | 所有命名测试 PASS；零 skip/block |
| NVE/NVT | 固定协议门槛通过；无MTS/NaN/升温 |
| 1CKK | smoke+metadata；release science regression通过 |
| benchmark | 同机、同精度、5次median；目标和实际均报告 |
| CI/package | clean build/install/import/load/rollback可重放 |

## 51. 静态代码审查清单

Reviewer 不运行代码前先搜索：

```text
MTS
FastQUpdateReporter
q_driver
cpu_kdtree
getPositions / host download in execute
cudaMemcpy device-to-host in per-step path
simtk.openmm hard-code
/home/
D:\\
TODO physics
magic 0.35 / 1.2 / 13.5 / 0.8 / 138.935458 duplicates
```

需要逐项回答：

- 是否只有 builder 负责生物化学分类？
- 是否所有设备参数支持 OpenMM reorder？
- density 是否有向，且 same-residue 仅作用于 density？
- explicit exclusions 是否同时作用 density 和 Local pair？
- Q 是否双 tanh？
- direct force RF cancellation 符号是否正确？
- chain force 是否包含两个方向的 density edge？
- F-only 是否仍计算 chargeGradient？
- penalty 是否同时影响 energy、chargeGradient 和 chain force？
- Local pair 是否错误使用了 `r_on` switch？
- builder 是否仍保留 PME，造成双算？
- Context 更新是否可能改变粒子数/exclusion 而未 reinitialize？
- CUDA 报告是否可能由 CPU fallback 生成？

## 52. 动态诊断开关设计

生产 API 不暴露内部 dens/Q workspace，但测试构建需要诊断。建议：

- CMake `LOCALCWLD_ENABLE_TEST_DIAGNOSTICS=ON`；
- 只在测试 library 导出 `getLocalCWLDDiagnostics(Context&)`；
- release 包关闭该符号或标记 internal；
- diagnostics 调用允许显式同步，但 benchmark 不得调用；
- 输出 model/ABI/schema 和 step index，避免把不同 Context 数据混合。

Diagnostics 用于层级定位，不是 production reporter，也不能每 N 步回传 Q 参与动力学。

## 53. 性能优化 Backlog（正确性后才派发）

按风险从低到高：

1. compact active sink/source index arrays；
2. 参数压缩：0/1 masks 使用小类型，但保持数值语义；
3. 使用单一 rc tile 并跳过 inactive lane；
4. fuse Q 与 penalty 初始化；
5. 优化 lambda atomic/reduction；
6. fuse pair/direct/lambda 的 tile 访问；
7. 为 r_env 建 compact neighbor list（必须证明端到端收益）；
8. CUDA graph/launch 减少（若 OpenMM架构允许）；
9. mixed precision 局部提升：density/Q/lambda 使用 mixed；
10. 多 GPU/多 Context 并行属于后续工作，不进入 v0.1。

每个优化工单必须交付：before/after E/F parity、五次 median、kernel timeline、VRAM、目标 GPU 型号、回滚提交。不能同时做两项导致无法归因。

## 54. 文档内部冲突处理规则

计划很长，未来可能出现旧段落与新发现冲突。优先级固定：

1. 明确标记的 Decision Record；
2. 第 17–18、46–49 节的冻结语义表；
3. LCWLD-030 生成的 golden fixture contract；
4. 第 24 节工单；
5. 前半部分路线图和示意文字。

发现冲突时创建 `PLAN-ERRATA` 小节，写旧表述、正确表述、影响工单和修订日期。不得由执行 agent默默选择一个版本。

### 当前 Errata 记录

- `mol_id` 的确切含义是 `residue.index`，已统一对外称 `residue_id`。
- same-residue 条件只出现在 density expression；它不会自动排除 Local pair energy。Local pair exclusions 来自显式 `addExclusion()`，即原 NonbondedForce exceptions。
- `r_on` 只用于修改后的原生 NonbondedForce switching；LocalCWLD pair 不使用 `r_on`。
- v2.6 虽注册了 `rho0` global，但 Q expression 通过字符串嵌入 `k_polar/rho0` 数值；插件以冻结公式数值语义为准，不以无实际引用的 global 形态为准。
- **（LCWLD-020 发现，2026-08-28）** 第 17.3 节写"`enable_solute_polarization=False` 时，溶质可以继续是 source，但不得成为响应 sink"，但 `test_lips_vs_pmeV2.6.py:build_phase_cwld_metadata()` 里，蛋白/配体逐原子设置 `dpolar/is_polar/dens_source/dens_sink/static_phase` 的整段循环（约 330-349 行）全部嵌套在 `if enable_solute_polarization:` 内部——也就是说关闭 solute polarization 时，v2.6 的实际行为是溶质完全不作为 density source，而不是"继续是 source 但不响应"。`openmm-localcwld/python/openmm_localcwld/metadata.py::build_particle_parameters()` 已按**代码**的字面行为实现（`enable_solute_polarization=False` ⇒ 溶质 `dens_source` 也归零），并在 `test_enable_solute_polarization_false_zeroes_solute_source_too` 里显式测试、注释说明这一矛盾。这条 errata 只记录矛盾，不代表"哪个是对的"已经决定——如果生产上确实需要"溶质继续是 source"的行为，需要主 agent 显式决定这是否算物理语义变更（第 44 节：默认公式/语义改变即使 API 不变也要升 `model_semantics_version`），不能由后续 agent 随手改。
- **（LCWLD-020 发现，2026-08-28）** v2.6 有一个模块级常量 `FAST_ACTIVE_DENSITY`（默认 `True`），在蛋白/配体分类循环里对`is_active_heavy = elem in ("O","N","S") or is_charged_site` 取反后 `continue`——也就是说默认情况下，蛋白/配体的碳、氢原子完全不参与 density（`dens_source` 也保持 0），不仅仅是不响应。这个开关没有出现在第 17.1（全局常量表）或 17.3（分类精确规则）的冻结列表里。`metadata.py` 把它实现成一个新增的builder 期参数 `fast_active_density`（默认 `True`，与 v2.6 一致），并在 `test_fast_active_density_default_excludes_carbon` 里同时测试默认值和关闭后的行为差异（关闭后蛋白碳原子会以 `POLAR_SOURCE_WEIGHT` 变成 density source）。需要主 agent 确认是否要把 `fast_active_density` 补进第 17.1 表格，以及 v0.1 是否允许它被 builder 调用方覆盖。

## 55. Result Manifest 具体字段

`schemas/localcwld-result-v1.schema.json` 至少要求以下结构：

```json
{
  "schema_version": "localcwld-result-v1",
  "run_id": "UTC timestamp + random suffix",
  "work_item": "LCWLD-140",
  "git": {"sha": "...", "dirty": false},
  "software": {
    "package_version": "0.1.0",
    "abi_version": 1,
    "model_semantics_version": "v2.6-compatible-1",
    "python": "...",
    "openmm": "..."
  },
  "hardware": {
    "platform": "CUDA",
    "device_name": "...",
    "device_index": "0",
    "precision": "mixed",
    "driver": "..."
  },
  "inputs": {
    "topology_sha256": "...",
    "positions_sha256": "...",
    "particle_parameters_sha256": "...",
    "exclusions_sha256": "...",
    "particle_count": 0,
    "box_vectors_nm": [[0,0,0],[0,0,0],[0,0,0]]
  },
  "model": {
    "r_env_nm": 0.35,
    "rc_nm": 1.2,
    "zmm_order": 2,
    "rho0": 13.5,
    "k_polar": 0.8,
    "q_delta_clamp_e": 0.2,
    "q_penalty_enabled": false,
    "q_penalty_strength": 180.0,
    "a_q2_builder": 0.5,
    "ca_source_weight_builder": 2.0,
    "water_response": true,
    "solute_polarization": true
  },
  "protocol": {
    "integrator": "LangevinMiddleIntegrator",
    "dt_ps": 0.002,
    "temperature_K": 300,
    "friction_per_ps": 1.0,
    "barostat": "...",
    "seed": 1234,
    "steps": 10000,
    "report_interval": 5000
  },
  "outputs": [
    {"path": "...", "sha256": "...", "bytes": 0, "units": "..."}
  ],
  "timing": {"start_utc": "...", "end_utc": "...", "elapsed_s": 0},
  "status": "PASS"
}
```

Schema 必须拒绝：缺 seed、NaN/Infinity、未知单位、空 hash、负步数、未声明 precision、CUDA platform 却没有 device。Python JSON 写出时需把 NumPy scalar 转为标准类型。

## 56. Benchmark CSV 固定列

```text
run_id,git_sha,package_version,openmm_version,platform,device,precision,
system_name,particle_count,active_source_count,active_sink_count,
r_env_nm,rc_nm,zmm_order,warmup_steps,measure_steps,repetition,
context_create_s,first_step_s,steady_step_us,ns_per_day,
density_kernel_us,q_kernel_us,pair_kernel_us,chain_kernel_us,
neighbor_build_us,host_sync_count,peak_vram_mb,peak_rss_mb,
energy_max_abs_error,force_max_abs_error,status
```

统计规则：

- 保留五次原始 repetition，不只保存 median。
- 报告 median、MAD、min/max；不只挑最好一次。
- JIT/Context 时间不混入 steady state。
- `host_sync_count` 应由 profiler 或 instrumentation 支持，不能凭观察填 0。
- 不同硬件的数据不直接计算加速比；主要比值必须同机、同 precision、同输入 hash。

## 57. CMake Target 依赖图

```text
OpenMMLocalCWLD                    # public API + ForceImpl + serialization
├── OpenMM
├── TestLocalCWLDAPI
└── TestSerializeLocalCWLDForce

OpenMMLocalCWLDReference           # runtime plugin
├── OpenMMLocalCWLD
├── OpenMMReference
└── TestReferenceLocalCWLDForce

LocalCWLDCommonObjects             # common host/device encoded source
├── OpenMMLocalCWLD
└── OpenMM common compute interfaces

OpenMMLocalCWLDCUDA                # runtime plugin
├── OpenMMLocalCWLD
├── LocalCWLDCommonObjects
├── OpenMMCUDA
└── TestCudaLocalCWLDForce

_openmm_localcwld                  # Python extension
├── OpenMMLocalCWLD
├── OpenMM Python/SWIG ABI
└── Python tests
```

构建脚本不得让 public API library 依赖 CUDA。Reference 和 CUDA 都是可独立安装的 runtime plugin。Serialization proxy 放在总能被 Python/C++ API 加载的库，避免未加载平台插件时 XmlSerializer 不认识 Force。

## 58. Plugin 加载 Smoke 流程

C++：

```cpp
vector<string> failures = Platform::loadPluginsFromDirectory(pluginDir);
ASSERT(failures.empty());
Platform& ref = Platform::getPlatformByName("Reference");
Context context(system, integrator, ref);
```

Python：

```python
from openmm import Platform
from openmm_localcwld import LocalCWLDForce

failures = Platform.loadPluginsFromDirectory(plugin_dir)
assert not failures, failures
names = [Platform.getPlatform(i).getName()
         for i in range(Platform.getNumPlatforms())]
assert "Reference" in names
```

注意插件为现有 Platform 注册新 kernel，不会出现名为 `LocalCWLD` 的新 Platform。Smoke 应创建含 LocalCWLDForce 的 Context，单纯 import 成功不足以证明 kernel factory 已注册。

错误消息至少区分：

- API library 未找到；
- serialization proxy 未注册；
- plugin dynamic library 加载失败；
- 指定 Platform 没有 `CalcLocalCWLDForceKernel` factory；
- OpenMM ABI/version 不匹配；
- CUDA runtime/device 不可用。

## 59. 预期异常文本规范

低等级 agent 容易写出含糊的 `invalid argument`。建议错误包含 Force、字段和值：

```text
LocalCWLDForce: environment cutoff must be finite and > 0 nm; received ...
LocalCWLDForce: cutoff must be >= environment cutoff; received ...
LocalCWLDForce: zmmOrder must be one of {1,2,3}; received ...
LocalCWLDForce: particle count (...) does not match System particle count (...)
LocalCWLDForce: exclusion cannot reference the same particle (...)
LocalCWLDForce: duplicate exclusion (...,...)
LocalCWLDForce: updateParametersInContext cannot change particle count or exclusions
LocalCWLDForce: periodic box is singular
LocalCWLDForce: zero interparticle distance for non-excluded pair (...,...)
LocalCWLDForce: no kernel implementation registered for platform ...
```

测试只断言稳定的关键片段，不锁死完整编译器措辞。

## 60. CI 伪配置与执行预算

```yaml
jobs:
  unit-reference:
    budget: 5 minutes
    requires: [OpenMM, C++ compiler, Python]
    runs: [API, metadata, golden, serialization, Reference parity, finite difference]
  integration-reference:
    budget: 15 minutes
    runs: [NVE, NVT, 1CKK-short]
  cuda-parity:
    schedule: nightly
    requires: [CUDA device]
    runs: [precision matrix, PBC, exclusion, FD subset, race repeat]
  cuda-benchmark:
    schedule: nightly
    requires: [dedicated GPU]
    runs: [synthetic scaling, 1CKK, profiler summary]
  science-regression:
    schedule: manual/release
    requires: [authorized compute allocation]
    runs: [frozen multi-seed matrix]
  package-smoke:
    runs: [clean install, plugin load, import, XML, minimal context]
```

CI 规则：

- PR 不启动 5 ns 任务。
- GPU shared runner 的性能只做趋势，不做严格 release 数值；release benchmark 使用固定节点。
- CUDA job 找不到 device 时标记基础设施失败或明确 skip，不运行 CPU fallback。
- 所有失败上传 fixture、manifest、stdout/stderr、failure YAML。
- science job 不自动更新 golden。

## 61. 测试命令统一入口

建议提供顶层脚本/target，但脚本只能编排，不隐藏退出码：

```bash
cmake --preset reference-debug
cmake --build --preset reference-debug --parallel
ctest --preset reference-debug --output-on-failure

cmake --preset cuda-release
cmake --build --preset cuda-release --parallel
ctest --preset cuda-release -L cuda --output-on-failure

python -m pytest python/tests -q
python -m pytest tests/integration -m "not long" -q
python -m pytest tests/benchmarks --benchmark-only
```

CTest labels 固定：`api`、`metadata`、`serialization`、`reference`、`finite-difference`、`pbc`、`cuda`、`integration`、`science`、`benchmark`、`package`。

## 62. 最终实现完成后的用户接口示例

```python
from openmm import app, unit, LangevinMiddleIntegrator
from openmm_localcwld import build_local_cwld_system

base_system = forcefield.createSystem(
    topology,
    nonbondedMethod=app.PME,
    constraints=app.HBonds,
)

system, manifest = build_local_cwld_system(
    base_system,
    topology,
    a_q2=0.5,
    ca_source_weight=2.0,
    enable_water_response=True,
    enable_solute_polarization=True,
    r_env=0.35*unit.nanometer,
    cutoff=1.20*unit.nanometer,
    zmm_order=2,
)

integrator = LangevinMiddleIntegrator(
    300*unit.kelvin,
    1/unit.picosecond,
    0.002*unit.picoseconds,
)

simulation = app.Simulation(topology, system, integrator)
simulation.context.setPositions(positions)
```

示例必须明确：builder 会返回 System 副本；输入 `base_system` 不变；生产使用标准 LangevinMiddle；不需要也不应创建 GB/MTS 特殊积分器。

## 63. 计划文档当前成熟度

本计划已覆盖：模型 contract、解析公式和导数、Reference 算法、GPU 数据流、API/ABI、序列化、builder、fixtures、有限差分、PBC、exclusion、动力学、1CKK、benchmark、CI、派工卡、review、artifact 和回滚。

下一步继续写文档的收益已经低于执行 `LCWLD-000/020/030` 获取真实工具链与 golden 证据的收益。后续新增内容应来自实际实现发现，通过 Decision Record 或 Errata 进入，而不是继续凭空扩展接口。
