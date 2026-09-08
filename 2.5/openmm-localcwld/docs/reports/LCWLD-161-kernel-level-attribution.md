# LCWLD-161：附带成本的机制、修复，以及 160 目标的重估

- 日期：2026-09-05
- 前置：LCWLD-160 计划（未开工）、LCWLD-150 完成（端到端 1.55x）
- 方法：`nsys` CUDA trace，按 MD 步切分、按臂分类、逐 kernel 求和
- 结论：**160 的 3.21x 目标可达 —— 但不在 pair kernel 这条路上。** 真因是
  `LocalCWLDForceInfo` 比较了 `residueId`，堵死了 OpenMM 的原子重排。
  改掉它：CWLD **4.83 → 1.52 趟**，端到端 **1.548x → 3.1–3.6x**，pair kernel 一行没动。

## 0. 先说这份报告没有推翻什么

160 的账**是对的**。两套独立方法互证：

    量                    墙钟臂减法      nsys GPU 时间
    CWLD / NonbondedForce    4.68 趟         4.66 趟
    CustomGB / Nonbonded     8.12 趟         7.93 趟
    NonbondedForce/CustomGB  0.1232          0.1261

基线也复现了：**e2e 1.575x，f 0.5765，比值散度 2.49%，PASS**（3 轮，1000/4000）。
「融合建表值 0.15x」也复现（buildEnvPairs 0.43 趟 ⇒ 1.55x → 1.65x）。

**被推翻的只有一件事：那 4.49 趟里有多少在 pair kernel 里。**

## 1. 测量

`nsys profile -t cuda`，200 warmup / 400 measure / 1 rep，2400 个 MD 步入 trace。
按 `integrateLangevinMiddlePart1` 切步，按步内出现的力 kernel 分臂。

    臂      步数    GPU us/步
    bare     600        97.0
    none     600       400.2
    new      600      1812.6
    gb       599      2804.4

    NonbondedForce 边际 = none - bare =  303.2 us/步   ← 一趟普通 rc 遍历
    CWLD 边际           = new  - none = 1412.4 us/步   = 4.66 趟
    CustomGB 边际       = gb   - none = 2404.2 us/步   = 7.93 趟

## 2. CWLD 那 1412 us/步 到底花在哪

    项                                us/步    趟     占比
    computePairAndAdjoint             751.4   2.48    53%
    buildEnvPairs                     131.0   0.43     9%
    computeChainForce                  39.1   0.13     3%
    computeDensity                     21.9   0.07     2%
    computeQ + clearBuffers             4.1   0.01     0%
    ── 我们自己的 kernel 合计          947.6   3.13    67%
    computeNonbonded      225 -> 551  +326.2  1.08    23%   ┐ 不在我们任何
    findBlocksWithInter    48 -> 176  +128.7  0.42     9%   ┘ kernel 里
    sortShortList2         16 ->  18    +1.4  0.00     0%
    ── 附带成本合计                    457.0  1.51    32%
    ────────────────────────────────────────────────────
    合计已归因                       1404.7          99.5%（边际 1412.4）

**pair kernel 是 2.48 趟，不是 4.49 趟。** 160 第 1 节把整个 4.49 趟当成 pair pass
（因为 density+chain 探针测出来 ~0，剩下的就都记在 pair pass 头上），
但臂减法/环境变量探针**分不开 pair kernel 和附带成本** —— 两者都随「CWLD 在不在」
一起出现一起消失。

## 3. 附带成本不是我们的 bug

    kernel                        bare    none     new      gb
    computeNonbonded               0.0   224.8   551.0   972.6
    findBlocksWithInteractions     0.0    47.6   176.3   170.4
    ──── 对照（不碰邻居表）────
    applySettleToPositions        24.7    25.3    27.4    26.6

    computeNonbonded 相对 none：  new 2.45x    gb 4.33x
    findBlocks       相对 none：  new 3.70x    gb 3.58x

OpenMM **自己的** CustomGBForce 把 `computeNonbonded` 撑得比我们更狠（4.33x vs 2.45x），
`findBlocks` 两者几乎一样（3.6x / 3.7x）。所以这是「第二个力共享邻居表」的通用代价，
不是 LocalCWLD 写坏了。

**时钟解释已排除**：对照 kernel `applySettleToPositions` 四臂 24.7/25.3/27.4/26.6，
纹丝不动。若是降频，它必须跟着一起慢。

**已排除的其它候选**：
* cutoff 不匹配 —— dump 里 NonbondedForce 和 CustomGBForce **都是 1.2 nm**，
  `LocalCWLDForce` 默认也是 1.2（`LocalCWLDForce.cpp:48`）。
* 重建频率 —— 三个臂都是**每 4-5 步一次**（数 `findBlocks` 的长启动得到），
  变的是每次重建的**耗时**，不是次数。
* 力组 —— benchmark 的 new 臂用默认力组 0，和 NonbondedForce 同组
  （力组 31 只在 Dynamics/Pathological 测试里用）。

### 3.1 机制：**是注册本身，不是我们的 kernel**（探针 D，实测）

探针 D（`LOCALCWLD_PROBE_NO_KERNELS=1`）：`initialize()` 原封不动 —— `addInteraction()`
和 `LocalCWLDForceInfo` 照常注册 —— 只让 `execute()` 直接返回，六个 kernel 一个不发。
**严格单变量**，且可以「加回去」验证（关掉环境变量就是生产路径）。

    对照（同一次 nsys 运行，new 臂与 none 臂相邻，各 400 步）
                            none 臂     new 臂(探针 D)    倍数    差值
    computeNonbonded         225.2        520.7         2.31x   +295.5
    findBlocksWithInter       47.9        195.8         4.09x   +147.9
    ─────────────────────────────────────────────────────────────────
    附带成本合计                                                +443.4 us/步

    kernel 全开时（第 2 节）                                    +455.0 us/步

**443.4 / 455.0 = 97%。** 附带成本几乎全部来自「这个力注册了共享邻居表」这件事，
跟我们的 kernel 跑不跑、跑多久无关。1.46 趟纯入场费。

含义：**pair kernel 优化到零也带不走这 1.46 趟**，它在 `execute()` 之外。

**探针存活性已验证**：banner 打印，且 new 臂 97.71 → 182.50 ns/day、f 0.4195 → 0.1629。
（不是空操作 —— 交接信记载本项目已有两次「变异测试通过时其实什么都没变异」。）

**机制的下一层未定。** 注册做了两件事，各自有单变量探针待测：
* 探针 E `LOCALCWLD_PROBE_NO_FORCEINFO` —— 不注册 `ComputeForceInfo`。
  嫌疑：`areParticlesIdentical` 把 `residueId` 也比了，于是**任何两个不同残基的
  原子都不可互换**，OpenMM 只在「相同分子」集合内部重排原子换局部性，
  这可能让重排基本失效。benchmark 的 CustomGBForce **也带 `mol_id` 逐粒子参数**，
  正好和它更狠的 4.33x 一致。
* 探针 F `LOCALCWLD_PROBE_NO_EXCLUSIONS` —— 只注册自排除。
  嫌疑：我们的排除集与 NonbondedForce 的不同，改变了 exclusionTiles / 邻居表的划分。

⚠ 上面这两条是**嫌疑，未验证**。

## 4. 这对 160 的目标意味着什么

`e2e = 1/((1-s) + s*f)`，实测 `s = 0.8624`，`f = CWLD趟数 / 7.93`。

    情形                                      CWLD 趟数    f       e2e
    现状                                        4.66     0.588   1.55x   【实测】
    pair kernel 2.48 -> 1.5                     3.68     0.464   1.86x
    pair kernel 2.48 -> 1.0                     3.18     0.401   2.07x
    pair kernel -> 0（Amdahl 上限，只动 pair）    2.18     0.275   2.67x  ←
    附带成本清零 + pair 1.5                      2.17     0.274   2.67x
    附带清零 + pair 1.0 + 建表融合                1.20     0.151   3.73x

**只动 pair kernel，端到端上限 2.67x。** 160 的 3.21x 目标（= CWLD 总共 1.5 趟）
低于附带成本 1.51 趟 + 建表 0.43 趟 + density/chain 0.20 趟 = **2.14 趟的地板**，
不碰附带成本就到不了。

## 5. 建议的下一步（按性价比）

1. **查清附带成本的机制。** 1.51 趟、32%，是最大的未审项，比 160 花篇幅讨论的
   buildEnvPairs（0.43 趟）大 3.5 倍。**且 ncu profile `computePairAndAdjoint`
   完全看不见它** —— 160 第 2 节「第一步上 ncu」会直接错过三分之一的成本。
   干净的单变量实验：加一个只注册 `addInteraction`、kernel 全空的假力，
   量「第二个力上表」的纯税；再加一个双 `NonbondedForce` 的臂做上界。
2. **pair kernel 仍然值得做**，但预期要下调：2.48 → 1.5 趟约值 **1.55x → 1.86x**，
   不是 160 说的 → 3.2x。160 第 3 节的嫌疑清单（mixed 能量累加、寄存器/LOCAL、
   伴随量原子操作）依然有效。
3. **`mixed` 能量累加有一条具体的、可静态验证的线索**：OpenMM 把
   `energyBuffer[...] += energy;` 包在 `#ifdef INCLUDE_ENERGY` 里并编译
   force / energy / forceEnergy 三个变体（`strings libOpenMMCUDA.so` 可读），
   动力学步里整条 fp64 依赖链被 DCE。我们是**无条件写**
   （`localCWLD.cc:641`），`execute()` 拿到了 `includeEnergy` 却从没用它选变体
   （`CommonLocalCWLDKernels.cpp:241`）。粗算只值 ~0.03 ms/步，**未实测**。
   验的时候先 `cuobjdump -sass` 确认 fp64 指令真的消失，别只测到少了一次写
   （这条验证方法来自 `abfe-ibs-cuda-04`）。

## 6. 遗留：ncu 看不见我们的 kernel

`ncu --kernel-name computePairAndAdjoint` 报 `No kernels were profiled`，
列出的 39 个 available kernel 全是 OpenMM 的，我们 6 个一个都不在，
而 **nsys 同一次运行能看到全部 6 个**。不是权限问题（`==PROF== Connected` 正常，
无 `ERR_NVGPUCTRPERM`；`RmProfilingAdminOnly: 1` 挡的是别的东西）。原因未查。

## 7. 复现

    export OPENMM_PLUGIN_DIR=$CONDA_PREFIX/lib/plugins       # 否则 "no Platform called CUDA"
    nsys profile -t cuda -o out ./build-cuda/platforms/cuda/BenchmarkCudaLocalCWLD \
        ./build-cuda/platforms/cuda docs/reports/dec005_1aay_dpolar 200 400 1
    nsys stats --report cuda_gpu_kern_sum out.nsys-rep      # 生成 out.sqlite
    # 然后按 integrateLangevinMiddlePart1 切步、按 computePairAndAdjoint /
    # computeN2Value / computeNonbonded 的有无分臂，逐 kernel 求和。

两次独立 trace（5/20 冷卡、200/400 热卡）附带成本分别为 1.29 趟和 1.51 趟。
本报告的数全部取自 200/400 那次。**没有改任何代码。**


---

# 第二部分：机制与修复（2026-09-05 同日）

## 8. 机制：`LocalCWLDForceInfo` 比较了 `residueId`

`areParticlesIdentical` 把 `residueId` 算进比较 ⇒ 任意两个不同残基的原子都不可互换
⇒ OpenMM 只在「相同分子」集合内部重排原子 ⇒ 体系里没有两个水分子相同
⇒ **重排等于不做** ⇒ 共享邻居表失去空间局部性 ⇒ 所有走这张表的 kernel 变慢，
**包括我们自己的 pair kernel**。

benchmark 里的 CustomGBForce 带 `mol_id` 逐粒子参数，同一个毛病，
正好对应它更狠的 4.33x。

### 探针（每个只改一件事，都能「加回去」验证）

    探针 D  注册但一个 kernel 不发     附带成本 +443.4 us/步（kernel 全开 +455.0）⇒ 97% 来自注册
    探针 E  完全不注册 ComputeForceInfo  附带成本 92% 消失
    探针 F  只注册自排除               **无法运行** —— OpenMM 抛
                                       "All Forces must have identical exceptions"

探针 F 那个失败**比计时数更硬**：OpenMM 强制所有力的排除集一致，所以我们的排除集
**不可能**与 NonbondedForce 不同，排除表这条因此被构造性地否定，不需要测。

### 探针 E 下的逐 kernel 变化（nsys，同一次运行，对照臂不变）

    kernel                          原来     探针E    倍数
    computePairAndAdjoint          751.4    349.2    0.46x   ← 我们自己的 kernel 也快一倍
    buildEnvPairs                  131.0     58.6    0.45x
    computeNonbonded               551.0    252.8    0.46x
    findBlocksWithInteractions     176.3     37.0    0.21x
    ──────────────────────────────────────────────
    CWLD 边际                     1412.4    480.4
    趟数                            4.66     1.59

对照：none 400.2→397.3，bare 97.0→96.0，gb 2804.4→2797.0（探针只影响 new 臂）。
爆炸检查：`computePairAndAdjoint` 全程平在 340–352 us 无漂移，体系没散，计时未被污染。

## 9. 修复

`areParticlesIdentical` 不再比较 `residueId`。

**安全论据**：kernel 只做 `residueId[atom2] == residueId[atom1]` 的相等判断、
**从不读它的值**（`localCWLD.cc:264,297`），物理只依赖残基**分区**。OpenMM 只搬整个
分子；排除表是 1-2/1-3，其传递闭包就是成键连通性；所以只要每个残基落在一个连通分量
里，重排就会把残基的原子整体搬走，分区在标签留在槽位的情况下依然成立。

**这个前提没有被假设，而是被检查**：`residuesFitInMolecules()` 用并查集实测。
不成立时**退回严格比较**（慢但正确），**不抛错** —— 两个合成测试体系里确实有
residue 0 跨分子的情况，把一个合法体系 refuse 掉比让它慢跑更糟。

`LOCALCWLD_PROBE_STRICT_FORCEINFO=1`（探针 G）强制走旧路径，用于同 binary A/B。

## 10. 结果：同一个 binary，一个变量，各 4 轮

    A  探针 G（旧的严格 ForceInfo）   e2e 1.548x   f 0.5899   比值散度 2.08%   PASS
    B  默认（本次修复）               e2e 3.391x   f 0.1843   比值散度 19.61%  NO VERDICT

    CWLD 趟数：4.83  ->  1.52     （nsys 独立测得 4.66 -> 1.59）

A 复现了 LCWLD-150 的基线（1.548 vs 1.575）。

### ⚠ B 为什么没有判定，以及该引用哪个数

不是被抢卡。全程 `nvidia-smi` 无第三方进程，卡温 47 → 69 → 73 °C，纯热降频。

    gb   57.96 -> 54.06 -> 51.00 -> 48.73    单调漂 18.9%
    new  178.78 -> 178.83 -> 177.17 -> 177.50        0.9%

`e2e = new/gb`，所以逐轮 **3.085 / 3.308 / 3.474 / 3.643**。

**轮转让比值免疫的前提是各臂一起漂，而这个前提被加速本身破坏了**：new 臂现在快一倍多、
发热少得多，不再跟着 gb 漂。这是新出现的测量问题，加轮数解决不了（漂移是单调系统性的，
不是噪声）。

**该引用的数**：`e2e 在 3.1x–3.6x 之间，取决于热状态`。保守下界 **3.075x**（最冷那轮，
两臂热状态最接近），是旧路径 1.548x 的 **1.99 倍**。
**不要引用中位数（3.391x / 3.365x）** —— 闸门拒绝了它。

**独立复现（2026-09-05 第二次 4 轮跑）**：逐轮 3.075 / 3.297 / 3.435 / 3.654，
CWLD 1.51 趟（前次 1.51/1.52），矩阵机时 14.4 GPU-h（前次 14.3）。两次几乎重合。

DEC-005 的判定（≥1.5x）不受影响：最保守的 3.085x 也是达标线的 2.06 倍。

### ⚠ benchmark 的诊断信息现在会误导

闸门对 B 打印过 `rate spread 17.6% (NOT monotonic -> suspect another process
on the GPU)`。**卡是干净的。** 触发「非单调」的是 `new` 臂在 0.9% 噪声内的一点抖动。

**已修（同日）**：方向变化要先过 2% 噪声门槛；并新增一类诊断 —— 各臂同步单调但
漂速差 > 5 个百分点时报 `monotonic but the arms drift at different rates`，
NO VERDICT 段改说 `Adding rounds will NOT help. Report a range across rounds,
or lock the clocks.`。真机验证输出：

    rate spread 18.7% (monotonic but the arms drift at different rates)
      gb drifts 18.7% over 4 rounds, new drifts 1.4%. Interleaving
      cancels drift only while the arms drift together; they no longer do.

判据边界由 `docs/reports/check_gate_diagnosis.py` 用三组真实数据钉住
（thermal / decoupled / 2026-09-04 真被抢卡那次 contention）。
**该脚本是 C++ 判据的副本，改阈值要改两处。**

## 11. 正确性

    ctest                                   7/7
    Dynamics 2000 步（约 8 次重排），快路径   p99 3.265e-03   超阈 146（地板 ~120）
    Dynamics 2000 步，探针 G 旧严格路径      p99 3.211e-03   超阈 140
    参照：文档记载正常 3.5e-3，陈旧表 18.6

快路径与旧路径**精度无差别**，余量约 280x 未损失。400 步只跨一次重排，
所以特意跑到 2000 步拿约 8 次重排的覆盖。

## 12. 这对 160 的重估

    情形                                   CWLD 趟数    e2e
    修复前                                    4.83     1.548x   【实测】
    修复后（本次）                             1.52     3.1-3.6x 【实测】
    再把 pair kernel 从 1.14 趟做到更低         ...      见下

修复后 CWLD 只剩 1.52 趟，**已经在 160 的「1.5 趟」目标上**。160 第 3 节的嫌疑清单
（寄存器/LOCAL 压力、伴随量原子操作、mixed 能量累加）**剩余空间已经很小**：
pair kernel 现在 349 us/步 ≈ 1.16 趟，做到 1.0 趟只值几个百分点。

**建议：160 可以关掉。** 下一步若还要性能，应重新做一次 kernel 级归因，
因为修复后的成本分布跟修复前完全不同。

一处需要顺带更新的估算：第 5.3 节那条 `mixed` 能量累加（OpenMM 编译 force/energy/
forceEnergy 三个变体并 DCE 掉 fp64 链，我们无条件写）粗算约 27 us/步。修复前它是
pair kernel 751 us 的 3.6%，**修复后是 349 us 的约 8%** —— 相对份额翻倍，
从「排不进前列」变成「剩下的东西里最大的一个」。仍然**未实测**，验的时候先
`cuobjdump -sass` 确认 fp64 指令真的消失（这条验证方法来自 `abfe-ibs-cuda-04`）。


---

# 第三部分：跨硬件验证（RTX 3090，2026-09-05）

`yayoi27.cluster.local`，RTX 3090 / 驱动 610.57.04，OpenMM `8.5.2.dev-36a30cb`
（`git_revision` 与开发机逐字一致，所以 kernel 依赖的 OpenMM 内部邻居表布局成立）。
独立构建树 `build-cuda-3090`，**ctest 7/7**。

## 13. 结果

    量                        2080 Ti          3090
    修复前 e2e                1.548x          1.454x
    修复后 e2e                3.1-3.6x        2.80-3.28x
    CWLD 趟数                 4.83 -> 1.51    5.18 -> 1.72
    修复后 f                  0.1843          0.2140
    s                         0.8644          0.8552

**趟数是跨硬件唯一可比的量**（ns/day 不可比：3090 绝对更快，new 230.21 vs 184.33）。
两台机器都是约 3.0-3.2 倍的趟数改善。

## 14. ⚠ 修复前的实现在 3090 上**不达标**

    B（探针 G，旧的严格 ForceInfo）  e2e 1.454x
    DEC-005 section 5: BELOW TARGET，margin -3.06%，比值散度 2.18%
    工具同时标注：margin 小于散度的 2 倍，未判定

2080 Ti 上是 1.548x（PASS），3090 上掉到 1.5x 线以下。**同一份代码，换台机器就翻面。**

这条比 3.05x 那个数值钱：LCWLD-150 的那个 PASS 是**硬件相关且勉强**的，只在开发用的
那张卡上成立。这次修复不只是"更快"，是把一个会随硬件翻车的结论变成了稳固的
（3090 上最保守的 2.80x 也是达标线的 1.87 倍）。

## 15. 闸门修复在未测过的硬件上正确工作

A 臂（两臂解耦）：

    monotonic but the arms drift at different rates
    gb drifts 17.3% over 4 rounds, new drifts 1.3%
    NO VERDICT -- ... Adding rounds will NOT help

B 臂（两臂都重、一起漂）：`ratio spread 2.18%` ⇒ **正常给出判定**。

同一套判据在一台从未调过的 GPU 上把"解耦"和"正常热漂"分得干净。
这比第 10.1 节用三组历史数据钉边界更有说服力。

## 16. 顺带修掉：`48.4 GPU-h` 是硬编码的 2080 Ti 基线

`BenchmarkCudaLocalCWLD.cpp` 里 `one 29x5ns matrix: 48.4 GPU-h -> %.1f h` 的 48.4
是写死的常量。在 3090 上照算等于把两台机器的数混在一起（3090 的 gb 臂本身就快
1.38 倍，真实基线不是 48.4）。已在输出里标明"只有加速比跨硬件可比，小时数不可比"。
**3090 上不要引用那个 15.9 h。**

## 17. 相对 PME 的位置（1AAY，2080 Ti 生产日志，dt=2 fs，5 ns，3 seed）

    PME                     663.2 / 660.2 / 657.7  ->  ~660 ns/day   100%
    CWLD 旧引擎              58.9 / 58.6 / 58.6                        8.9%
    CWLD 旧引擎（其它配置）   39.0 / 48.4                          5.9% / 7.3%
    AMOEBA                  5.8                                       0.9%

修复后按实测 3.1-3.6x 外推：**约 183-212 ns/day ⇒ PME 的 28-32%**。

**⚠ 上面这段已被 2026-09-05 的同机实测取代，见第 18 节。保留原文以记录当时的不确定性。**

⚠ **这是外推不是实测**，而且有一处对不上：benchmark 的 `none` 臂（去掉 CWLD、
保留 CutoffPeriodic 的 NonbondedForce）是 400 ns/day，比生产 PME 的 660 **还慢**，
方向反了。跨 harness 比数正是本项目反复翻车的地方（benchmark 自己的输出末尾就印着
这句警告）。**28-32% 只能当量级看。** 要准确的数，应在 benchmark 里加一条 PME 臂
直接量，而不是继续做跨 harness 的乘法。


## 18. 相对 PME：同机同 harness 实测（3090，2026-09-05）

`yayoi27` / RTX 3090，`lips.run.zn_job`，1AAY 32794 原子，dt=2 fs，
平衡 NVT 100 ps + NPT 200 ps，production 0.5 ns，单 seed。
产物在 `data/bench_3090/`（已标注为计时产物、非科学数据）。

    引擎                     production        占 PME     完整性
    PME                      497.2 ns/day      100%       跑完
    CWLD LocalCWLDForce      212.0 ns/day      42.6%      跑完
    CWLD CustomGBForce          41 ns/day       8.2%      未跑完（读数已稳定）

**CWLD 从 PME 的 8.3% 提到 42.6%。** 这是同一个节点、同一套 harness、
同一个体系的分子和分母，不再是跨 harness 的乘法。第 17 节那个 28-32% 的外推作废。

旧引擎那条取自运行中的读数（41.2 / 41.0，两次一致，已收敛），不是完成值：
0.5 ns production 预计要 20 分钟以上，跑到一半手动中止。
**「旧引擎慢到跑不完一次验证跑」这件事本身就是本项目存在的理由。**

平衡段（300 ps，三个引擎都跑了，可作旁证）：

    PME                            58 s
    LocalCWLDForce                128 s
    LocalCWLDForce + 旧 ForceInfo  398 s     （探针 G）
    CustomGBForce                 564-569 s

### ⚠ 未解决：生产 harness 的 5.15x 与 benchmark 的 3.05x 不一致

`212.0 / 41 = 5.2x`，而同一张 3090 上 benchmark 给的是 **3.05x**。差 69%。

顺着查发现一个更反常的：

    CustomGBForce 生产速度   2080 Ti (ligoff, 2026-09-03)  48.4 ns/day
                            3090   (ligoff, 2026-09-05)    41 ns/day

**更快的卡上反而更慢**，而 benchmark 的 gb 臂方向是对的（3090 75.47 vs 2080 Ti 54.78）。

所以问题不在插件，在**旧引擎于 3090 节点的生产 harness 里异常慢**。嫌疑是节点的
host 侧（CPU 或 NFS 写）而非 GPU —— CustomGBForce 是三趟 pass、kernel 启动次数远多于
插件，对 host 慢最敏感；而 benchmark 不写 dcd/csv，正好绕开 I/O。**未验证。**

判别实验（约 2 分钟，未做）：在 3090 上以 `--report-ps 1000`（几乎不写盘）重跑旧引擎。
速度跳回 ~70 则是 I/O；仍是 ~41 则是 host CPU。（未做，成本约 2 分钟）

**因此可以说**：「在 3090 节点的生产流程里，CWLD 从 PME 的 8.3% 提到 42.6%」。
**不可以说**：「插件比旧引擎快 5.2x」—— 那个数里混着节点效应。跨硬件可比的量
仍然是趟数（4.83→1.51、5.18→1.72）和 benchmark 的 3.05x。
