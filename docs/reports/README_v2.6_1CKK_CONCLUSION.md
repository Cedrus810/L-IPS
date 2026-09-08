# L-IPS v2.6：1CKK 验证总结与当前结论

更新时间：2026-07-15（**2026-09-01 追加重大保留意见，见下方警告框**）

> ## ⚠ 2026-09-01：本文件的参数选择结论存在一个未解决的混杂，暂不可作为冻结依据
>
> **四个配置不是在同一个 pair closure 下跑的。** 查 2.4 作业日志与 legacy 快照定的性：
>
> | 配置 | 轨迹日期 | closure |
> | --- | --- | --- |
> | `PME` / `CWLD_aq0_ca_w1` / `CWLD_aq0_ca_w2` | 2026-07-11 | **ell=1**（v2.6 原始 RF-like） |
> | `CWLD_aq0p5_ca_w2` ← **胜出的那个** | 2026-07-14 | **ell=2**（ZQuad） |
>
> 依据：全部 7 份 `L-IPS.o*` 里只有 `L-IPS.o10793`（07-14，跑 aq0p5_ca_w2 三个 seed）
> 打印了 `ZMM closure order ell=2`；其余都没有这行，因为 ZMM closure 是后加的——
> `legacy_versions/MTS_test_lips_vs_pmeV2.6.py`（2026-07-06）里根本没有 `zmm_uclosure`，
> 用的是硬编码的 `(Q1*Q2-qbase1*qbase2)/r + Q1*Q2*(r²/(2rc³)-1.5/rc) + qbase1*qbase2/rc`，
> 代数上就是 ell=1。
>
> **为什么不能忽略**：`closure_force_audit.py`（2026-09-01，静态力审计，零 MD）
> 在真实 1CKK 构型上测得 ell=1 相对 ell=2 的力误差比为——总体 0.980，
> 但 **Ca 第一壳 0.844、第二壳 0.788、蛋白重原子 0.789**。
> 差异最大的位置恰好就是本文件用来判优的观测量（Ca-water RDF 峰/AUC、配位数、salt bridge）
> 所在的区域。所以不能默认「closure 差异可忽略」。
>
> **当前状态**：`a_q2=0.5, ca_source_weight=2.0` 的参数差异与 closure 阶数差异被混在一起，
> **该选择的归因不成立**，需要在同一 closure 下重比后才能谈冻结。
> 详见 `STAGE1_CLOSURE_KILLSWITCH_REPORT.md` §11 与 `PLAN_PSWF_Closure_Family.md`。
>
> 以下原文保留未改，作为混杂被发现前的记录。

## 一句话结论

基于同一协议下的 5 ns production × 3 seeds，当前建议将
`a_q2=0.5, ca_source_weight=2.0` 作为 **1CKK 内部验证通过的暂定默认配方**。
旧 500 ps 单轨迹中“`a_q2=0.5` 导致严重 Ca-water 过度结构化”的现象没有在长轨迹中复现，
主要应归因于短轨迹采样噪声。新增的 500 ps `V2.3 charge_mod` 消融（见下文第 7 节）在机制方向上
进一步支持这一结论：`aq0.5` 的 salt bridge 最接近 PME，而 `aq0`/water-off 明显偏强，
说明水响应和 charge weighting 的作用方向正确；但该结果仍是单条 500 ps 轨迹，只作为机制证据，
不参与参数选择。不过，当前结果仍不足以证明该参数可无条件推广到其他体系；
冻结参数前还需要完成时间分块检查。项目中已有一条 1 ns AMOEBA 2018 极化力场轨迹，
它应当作为 PME 之外的第二参考轴；但由于协议、长度和指标尚未完全对齐，目前只能做初步三方比较，
不能直接并入 5 ns × 3-seed 的统计参数选择。随后需要完成同口径的极化参考和第二个 Ca²⁺ 体系外部验证。

## 数据与实验设计

本结论只使用当前根目录的 5 ns 长轨迹结果：

- `1ckk_v26_long_convergence_5ns_per_seed_metrics.csv`：逐 seed 指标。
- `1ckk_v26_long_convergence_5ns_summary.csv`：按配置汇总的 count/mean/std。
- `LONG5ns_*_seed*_1ckk.dcd`：production-only 轨迹。
- `LONG5ns_*_seed*_1ckk.csv`：production-only 状态数据和势能。

四个配置均使用 3 个随机种子，每条轨迹包含 100 ps 平衡和 5 ns 无约束 production；
DCD/状态文件只从 production 开始记录，因此没有混入受限平衡帧。

| 配置 | `a_q2` | `ca_source_weight` | 用途 | closure（2026-09-01 补） |
|---|---:|---:|---|---|
| PME | — | — | 基准 | ell=1 |
| CWLD_aq0_ca_w1 | 0.0 | 1.0 | Ca source weight 下界 | ell=1 |
| CWLD_aq0_ca_w2 | 0.0 | 2.0 | 未加权候选 | ell=1 |
| CWLD_aq0p5_ca_w2 | 0.5 | 2.0 | Charge-weighted 默认候选 | **ell=2** ⚠ |

⚠ 最后一列是 2026-09-01 补查出来的，写这份文档时并不知道——**四个配置的 closure 不一致**，
见文件顶部的警告框。原文下面这句「四个配置均使用 3 个随机种子……」在协议其余方面仍成立。

旧的 `1ckk_v26_replicate_matrix_analysis.md` 以及旧目录中的 500 ps 数据属于另一批结果，
不能与本批长轨迹混用。旧 500 ps 结果可用于说明采样风险，但不再用于最终参数选择。

项目中另有极化力场参考：

- `0706/amoeba2018.csv`：AMOEBA 2018，1 ns 单轨迹、1000 帧。
- `0706/1ckk_pme_cwld_amoeba_merged.csv`：旧版 PME/CWLD/AMOEBA 指标对照。

AMOEBA 数据是重要物理参考，但目前不是与长轨迹矩阵完全匹配的 replicate：它只有一条 1 ns 轨迹，
没有 Ca-water RDF、water residence、salt bridge 和有效蛋白偶极数据。因此下文单独列出可比指标，
不把它伪装成与 5 ns × 3-seed 等权的统计样本。

## 5 ns 核心结果

下表为 mean ± sample std，`n=3 seeds`。

| 指标 | PME | aq0-w1 | aq0-w2 | aq0.5-w2 |
|---|---:|---:|---:|---:|
| 温度 (K) | 300.326 | 300.263 | 300.310 | 300.329 |
| 密度 (g/mL) | 1.0217 | 1.0252 | 1.0290 | 1.0258 |
| 蛋白 RMSD (nm) | 0.306±0.008 | 0.285±0.010 | 0.301±0.009 | 0.312±0.010 |
| Ca-water RDF peak | 1.676±0.181 | 1.799±0.200 | 1.636±0.051 | 1.736±0.029 |
| Ca-water coordination @0.30 nm | 0.936±0.155 | 0.952±0.118 | 0.905±0.037 | 0.923±0.099 |
| Ca-protein-O coordination @0.30 nm | 6.723 | 6.632 | 6.582 | 6.597 |
| Ca-total-O coordination @0.30 nm | 7.659 | 7.584 | 7.486 | 7.520 |
| Ca-water first-peak AUC | 0.0393±0.0061 | 0.0401±0.0048 | 0.0379±0.0014 | 0.0390±0.0037 |
| Ca-water residence τ (ps) | 1402±520 | 2040±909 | 1559±707 | 1462±752 |
| 观察到 `t1/e` 的 seeds | 3/3 | 2/3 | 3/3 | 3/3 |
| Salt-bridge first-peak AUC | 2.464±0.195 | 2.124±0.249 | 2.129±0.530 | 2.382±0.463 |
| Salt contacts/frame @0.40 nm | 22.19±1.66 | 18.63±2.03 | 18.66±4.75 | 21.04±4.01 |
| 蛋白偶极矩 (D) | 291±112 | 364±22 | 288±70 | 352±76 |
| 势能线性 slope (kJ mol⁻¹ ns⁻¹) | -47.4 | -47.5 | -51.0 | -39.7 |

注意：`6.7` 左右的数值是 **Ca-protein-O coordination**，不是 Ca-water coordination。
Ca-water coordination 是 `0.9` 左右。二者不能混用。

## AMOEBA 2018 极化力场：初步三方比较

AMOEBA 原始文件中的 Ca 配位数是 4 个 Ca²⁺ 的总数；为与脚本输出的“每个 Ca 平均配位数”一致，
下表已除以 4。PME/CWLD 来自 5 ns × 3 seeds，AMOEBA 来自单条 1 ns 轨迹，故只能比较中心值和趋势。

| 可比指标 | PME（5 ns × 3） | CWLD aq0.5-w2（5 ns × 3） | AMOEBA 2018（1 ns × 1） |
|---|---:|---:|---:|
| 密度 (g/mL) | 1.0217 | 1.0258 | 1.0206 |
| 蛋白 RMSD (nm) | 0.306 | 0.312 | 0.191 |
| Ca-protein-O coordination / Ca @0.30 nm | 6.723 | 6.597 | 6.080 |
| Ca-carboxylate-O coordination / Ca @0.30 nm | 5.273 | 5.102 | 4.599 |

初步观察：

- CWLD 的 Ca-protein-O 和 carboxylate-O 配位均略低于 PME，方向上朝 AMOEBA 移动，但仍更接近 PME。
- 这可能说明 CWLD 的环境响应已产生一部分“降低固定电荷过强配位”的极化趋势；也可能只是协议/采样差异。
  在 AMOEBA 没有同长度、同初态、同分析口径 replicate 前，不能把该方向直接解释为物理改进。
- AMOEBA RMSD 包含从 0 开始的前 1 ns，时间窗和参考结构与 production-only 5 ns 统计不同，
  其较低均值不能用来判定 AMOEBA 更稳定或 CWLD 更差。
- CWLD 密度比 PME/AMOEBA 高约 0.4–0.5%，但当前 CWLD/PME production 为关闭 barostat 的 NVT，
  盒子来自较短 NPT 平衡；密度不是这一批实验的强判据。

### AMOEBA 参与当前参数选择后的明确结论

使用已经完成的 last-4ns 分析后，可比数据如下。AMOEBA 的两个 Ca 配位数仍按 4 个 Ca 归一化：

| 指标 | PME | aq0-w1 | aq0-w2 | aq0.5-w2 | AMOEBA 2018 |
|---|---:|---:|---:|---:|---:|
| 密度 (g/mL) | 1.0217 | 1.0252 | 1.0290 | 1.0258 | 1.0206 |
| Ca-protein-O / Ca @0.30 nm | 6.741 | 6.645 | **6.584** | **6.591** | 6.080 |
| Ca-carboxylate-O / Ca @0.30 nm | 5.290 | 5.153 | **5.083** | **5.091** | 4.599 |

AMOEBA 对当前选择提供的结论是：

- 它明确支持 `ca_source_weight=2.0`。w2 的 Ca-protein-O 和 carboxylate-O 配位比 w1 更接近
  AMOEBA 的低配位方向；aq0-w2 与 aq0.5-w2 在这两个指标上几乎相同。
- 现有 AMOEBA 文件 **不能区分 `a_q2=0` 与 `a_q2=0.5`**，因为它没有 Ca-water RDF/AUC、
  water residence、salt bridge 和可用偶极数据。不能假装 AMOEBA 已经对这些指标投票。
- `a_q2=0.5` 的选择来自 AMOEBA 不覆盖的长轨迹证据：last-4ns 中 aq0.5-w2 的 Ca-water peak
  `1.680 vs PME 1.679`、AUC `0.03816 vs 0.03915`、residence τ `1424 vs 1494 ps`、
  salt-bridge AUC `2.501 vs 2.505`、接触数 `22.26 vs 22.65`，均优于 aq0-w2 的综合匹配。

因此，把历史 AMOEBA 和当前长轨迹放在一起后的参数结论仍是：

```python
DEFAULT_A_Q2 = 0.5
CA_SOURCE_WEIGHT = 2.0
ENABLE_WATER_RESPONSE = True
ENABLE_SOLUTE_POLARIZATION = True
```

这里 `w2` 同时得到 AMOEBA 配位趋势和 PME 长轨迹动力学的支持；`aq0.5` 得到 PME 长轨迹
Ca-water/salt-bridge 的支持，且没有违背现有 AMOEBA 可比指标。这是当前全部历史证据的联合选择，
不是只按 PME 选择。

因此正确的比较框架不是“CWLD 只要复制 PME”，而是三层参考：

1. **PME 固定电荷基线**：检验数值稳定性、结构保持和现有 ABFE 工作流兼容性。
2. **AMOEBA 极化参考**：检验 CWLD 是否捕获合理的极化方向和环境响应，而非仅拟合固定电荷模型。
3. **实验数据**：当 PME 与 AMOEBA 给出不同趋势时，用实验结构、配位、交换动力学和自由能决定哪个更接近真实体系。

## 分项判断

### 1. 整体稳定性：通过

所有配置温度稳定在 300 K 附近，蛋白 RMSD 约 0.29–0.31 nm，没有 NaN、轨迹崩溃或明显结构失稳。
CWLD 没有表现出相对于 PME 独有的势能下降趋势；各组势能 slope 同量级。

这里的势能 slope 只用于检查 Langevin/NVT 轨迹是否存在异常非平稳趋势，
不能当作严格的 NVE 能量守恒测试。所有组仍有共同的缓慢势能松弛，说明 100 ps 平衡可能偏短，
需要用时间分块确认后半程结论是否稳定。

### 2. `ca_source_weight`：w2 优于 w1

w1 的静态配位并未被破坏，但其 residence τ 更长、方差更大，且仅 2/3 seeds 在 5 ns 内观察到
ACF 跨过 `1/e`。w2 的三个 seeds 均观察到 `t1/e`，其 Ca-water peak、AUC、配位和 residence
整体落在 PME 的运行间波动范围内。因此不再继续扫描 `ca_source_weight`，暂定 `2.0`。

### 3. `a_q2=0.5`：旧的“严重过极化”结论未复现

与 aq0-w2 相比，aq0.5-w2 只改变 `a_q2: 0 → 0.5`，属于单变量对照。长轨迹显示：

- Ca-water first-peak AUC：0.0390，几乎等于 PME 的 0.0393。
- Ca-water coordination：0.923，接近 PME 的 0.936。
- Residence τ：1462 ps，接近 PME 的 1402 ps；3/3 seeds 均观察到 `t1/e`。
- Salt-bridge AUC 和平均接触数的中心值均比 aq0-w2 更接近 PME。
- 没有出现旧单条 500 ps 轨迹中 peak `2.14 vs PME 1.61` 所暗示的灾难性过度结构化。

因此，旧的 500 ps 异常不能继续作为否决 `a_q2=0.5` 的证据。

不过，salt-bridge 的逐 seed 波动依然较大，aq0.5 相对 aq0 的改善尚未达到可称为严格统计显著的程度。
当前选择 aq0.5 的理由是它在 Ca hydration、residence 和 salt bridge 多项中心值上更均衡，
而不是已经证明它与 PME 等价。

### 4. 蛋白偶极矩：暂不作为否决 aq0.5 的主判据

aq0-w2 的偶极均值更接近 PME，而 aq0.5-w2 偏高。但 PME 自身跨 seed 标准差约 112 D，
而当前 `protein_dipole_mean_debye` 使用基础固定电荷 `qbase`，不是 CWLD 瞬时响应电荷。
该指标更接近构象代理，不能单独证明或否定极化响应是否正确。

### 5. 电荷响应：存在明确响应；近 Ca 抽样缺陷已修复，待重新验证

aq0.5 的 Ca 附近水氧响应约为 `ΔQO ≈ -0.014 e`，强于 aq0-w2 的约 `-0.008 e`，
说明 charge weighting 的确增强了局部响应。现有结构和动力学指标未显示这种增强已经造成明显过度配位。
500 ps `V2.3 charge_mod` 消融独立复现了同一响应方向（`ΔQO ≈ -0.0153 e`），支持该效应可复现。

此前 aq0.5 seed0 的 `ca_near_water_delta_QO_mean` 缺失。原因不是没有 Ca 第一水合层，而是
`compute_water_q_profile()` 先用固定 `linspace` 下采样水分子，可能漏掉靠近 Ca 的水。

**状态更新（2026-07-15）：已在代码中修复。** `compute_water_q_profile()` 现在每帧强制保留所有
`distance_to_ca <= NEAR_CA_CUTOFF_NM(0.35 nm)` 的水，再从其余 bulk water 中抽样，不会再漏采
Ca 第一水合层。修复后的重新分析由 `block_time_check_v26.py` 执行（同时完成第 2 步的时间分块检查），
待用户在计算节点上跑完后，需确认每个 CWLD 配置的 `ca_near_water_delta_QO_mean_count` 均为 3，
之前"只有 2 个有效 seeds"的限制才算解除。

### 6. Salt bridge：aq0.5 中心值改善，但不能过度解读

aq0.5-w2 的 salt-bridge AUC 和平均接触数比 aq0-w2 更接近 PME，但逐 seed 改善并不完全一致。
此外，`salt_bridge_contact_frame_fraction_0p40_nm` 在所有配置均为 1.0，已经饱和；
它只表示整条蛋白每一帧至少存在一个盐桥，不能区分模型，应停止作为选择指标。
后续使用 first-peak AUC、平均接触数或逐残基对 occupancy。

### 7. 500 ps `V2.3 charge_mod` 消融：补充机制证据（非参数选择依据）

在用户计算节点上单独跑的一次 500 ps `charge_mod` 消融矩阵（`1ckk_v23_charge_mod_ablation_metrics.csv`），
比较 PME、`aq0`、water-off 等配置与 `aq0.5`。核心观察：

- Salt bridge：aq0.5 最接近 PME（AUC 1.478 vs PME 1.520；接触数 13.38 vs 13.00）。
  aq0 和 water-off 明显偏强（AUC 约 2.02–2.07，接触数约 17.6–18.8），说明关闭水响应会让
  盐桥被系统性过度稳定，动态 water response 确实有必要。
- Ca-water peak：aq0.5 仍接近 PME（1.625 vs 1.527），但 AUC 和配位数偏高约 25%——
  与 5 ns × 3-seed"基本匹配 PME"的结论不同，再次印证单条 500 ps 波动很大，不能替代长轨迹结论。
- 所有配置本次 `t_1e_observed=False`，ACF 尾部仍在 0.61–0.80，因此本次结果不能提供任何
  residence τ 的动力学结论。
- 近 Ca 水氧响应 `ΔQO ≈ -0.0153 e`，与 5 ns 结果的约 `-0.014 e` 方向和量级一致，见第 5 节。

**证据层级（本次结论采用）：**

1. 5 ns × 3 seeds 长轨迹结果 —— 负责最终参数选择。
2. 本次 500 ps `charge_mod` 消融 —— 只用于解释 water response / charge weighting 的作用方向，
   不参与参数选择。
3. 旧的单条 500 ps 轨迹 —— 仅作为采样失败的反面案例，不再具有证据效力。

顺带修复了一个格式问题：`1ckk_v23_charge_mod_ablation_metrics.csv` 此前保存时索引列无列名，
现已改为 `metrics_df.to_csv(..., index_label="config")`，不影响本次数值结论。

## 当前参数决定

当前建议：

```python
DEFAULT_A_Q2 = 0.5
CA_SOURCE_WEIGHT = 2.0
```

状态：~~**provisional default / 1CKK 内部验证通过**~~
→ **2026-09-01 降级为「归因不成立，待同 closure 重比」**，见文件顶部警告框。
这个参数值目前仍是 `test_lips_vs_pmeV2.6.py` 里的默认值（未改动），但它现在没有有效的实验依据支撑。

这不是“已经完成通用力场验证”。目前只有一个蛋白体系、每配置三个 5 ns replicate，
且部分慢变量仍存在较大跨 seed 方差。已有 AMOEBA 数据提供了有价值的极化方向参考，
但其 1 ns 单轨迹协议尚未与 PME/CWLD 长轨迹统一，所以也不能据此宣称 CWLD 已达到 AMOEBA 级极化精度。

## 下一步

按优先级执行：

1. ~~修复 `compute_water_q_profile()`~~ **已完成（2026-07-15）**：每帧强制保留所有
   `distance_to_ca <= 0.35 nm` 的水，再从其余 bulk water 中抽样，避免近 Ca 电荷响应缺失。
2. 对现有轨迹做 5 个 1 ns block，并单独重算最后 4 ns；检查结论是否随时间块系统漂移。
   **脚本已就绪：`block_time_check_v26.py`**（复用 `analyze_1ckk()`，不重新跑 MD，只对已有
   `LONG5ns_*_seed*_1ckk.{dcd,csv}` 重新分析）——用户在计算节点上运行，输出：
   - `1ckk_v26_long_convergence_5ns_per_seed_metrics_QFIX.csv` / `_summary_QFIX.csv`
     （修复后的全 5 ns 重算，用于跟本文件第 35–57 行的原始表对比，确认除近 Ca 指标外其它数值不变）
   - `1ckk_v26_block_time_check_per_block_metrics.csv` / `_block_summary.csv`（5 x 1ns block）
   - `1ckk_v26_block_time_check_last4ns_metrics.csv` / `_last4ns_summary.csv`（丢弃首 1 ns 后）
   重点检查：aq0.5-w2 在最后 4 ns 的 Ca-water AUC/coordination/residence 是否仍接近 PME；
   salt-bridge 改善是否在各时间块持续存在；各指标是否存在单调时间漂移；
   修复后每个 CWLD 配置的 `ca_near_water_delta_QO_mean_count` 是否均为 3。
3. ~~若后 4 ns 仍保持相同排序，正式冻结~~ **2026-09-01 阻塞**：在解决顶部警告框那个
   closure 混杂之前不能冻结。最小重比方案：四个配置在**同一个** closure 下各跑 3 seed × 5 ns。
   若只想先救结论，至少把 `CWLD_aq0p5_ca_w2` 用 ell=1 补跑 3 seed，与另外三组同口径直接比。
   （成本参考：CWLD 每条 5 ns 约 2 小时，PME 约 10 分钟。）
4. 统一极化参考口径：尽可能从与 PME/CWLD 相同的初始结构出发，用相同温度、压力、时间窗、
   Ca 归一化和分析脚本重跑/重分析 AMOEBA。理想设计为 5 ns × 3 seeds；若成本过高，至少先做
   3 个独立 1 ns replicate，并明确它只能验证快结构量，不能验证 residence。
5. 在第二个独立 Ca²⁺ 结合体系上做外部验证；应进行 PME / 最终 CWLD / 极化力场三方比较，
   并尽量加入实验配位或动力学数据，而不是只把 PME 当作“真值”。
6. 如需可靠的水交换动力学常数，对 PME、最终 CWLD 和可负担的极化参考运行更长轨迹；5 ns 虽已观察到多数 `t1/e`，
   但 residence 的跨 seed 方差仍约为均值的 37–51%。
7. Salt bridge 若在外部体系仍系统偏低，应单独检查溶质 source weighting 或方向性修正，
   不再通过 Ca source weight 补偿。500 ps `V2.3 charge_mod` 消融的方向性证据（第 7 节）支持
   现有 water response 机制是必要的，但不能替代外部体系上的定量验证。

## 最终目标

最终目标不是让 CWLD 在 1CKK 上逐项拟合 PME，也不是逐体系重调参数。目标是得到一套冻结的、
可复现且可转移的 CWLD-L-IPS 参数与实现，使其：

- 保留 PME 基线的数值稳定性、蛋白结构稳定性和 ABFE 工作流可靠性；
- 在 Ca 配位、水响应、盐桥和介电环境变化上捕获与 AMOEBA/其他极化力场一致的合理极化趋势；
- 当 PME 与极化力场不一致时，以实验结构、动力学和自由能为最终裁判；
- 在多个蛋白/配体体系中无需重新调 `a_q2` 或 source weights；
- 在 ABFE benchmark 上不降低自由能精度和配体排序能力，同时提供设计预期的计算效率或物理改进。

最终验收物应包括：冻结参数和代码、1CKK 内部验证、同口径 AMOEBA 对照、第二个 Ca 体系验证、
多配体 ABFE benchmark、误差/收敛/性能报告，以及明确的适用范围与已知限制。

## 不应继续做的事情

- 不再根据单条 500 ps 轨迹选择参数。
- 不再混用旧目录和当前根目录结果。
- 不再只看 RDF 最大 bin 或 salt-bridge peak 作决定。
- 不再继续扫描 `ca_source_weight=1/1.5/2`；这一问题已有足够证据收敛到 w2。
- 在完成第二体系验证前，不宣称当前参数具有普适性或已经达到 production force-field 级别。
- 不再把 500 ps `V2.3 charge_mod` 消融（第 7 节）当作参数选择依据；它只解释方向/机制，
  结论仍以 5 ns × 3-seed 数据为准。
