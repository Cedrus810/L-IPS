# LCWLD-160：把 pair kernel 从 4.5 趟做到 1.5 趟

- 日期：2026-09-04
- 前置：LCWLD-150 完成，端到端 **1.55x**（DEC-005 未达标线 1.5x，PASS）
- 目标：端到端 **约 3.2x**
- 状态：**已关闭（2026-09-05，用户决定）。从未开工，也不需要开工。**

  > **关闭理由**：`LCWLD-161` 用一个 `ComputeForceInfo` 的改动把端到端做到了
  > 3.1–3.6x，达成了本计划的 3.21x 目标，**没碰 pair kernel 一行**。
  >
  > 本计划有两处前提是错的，留在这里备查：
  > 1. 「pair pass = 4.5 趟」把 1.5 趟的附带成本（OpenMM 自己的
  >    `computeNonbonded` / `findBlocksWithInteractions` 被撑贵）算进了 pair pass。
  >    实际 pair kernel 是 2.48 趟，修复后 1.16 趟。
  > 2. 第 2 节「第一步是上 `ncu`」会直接错过真因 —— ncu profile
  >    `computePairAndAdjoint` **看不见 ForceInfo**。
  >
  > 仍然有效的部分：第 3 节嫌疑清单、第 4 节测量纪律、第 5 节工具、第 6 节不要做的事。
  > 若日后还要追性能，先重做一次 kernel 级归因：修复后的成本分布与修复前完全不同。
  >
  > 原状态：下面标了「实测」的是数，标了「嫌疑」的一条都没验证过。

## 1. 为什么只剩这一件事

以 CustomGBForce 为单位是坏单位（三趟不等价 pass 的和）。换成**一趟普通 rc 遍历**
（体系里的 `NonbondedForce`：CutoffPeriodic、同 1.2 nm、同一张邻居表，`ARM_BARE` 量出来的）：

    一趟普通 rc 遍历    = 0.1305 个 CustomGBForce     【实测】
    CustomGBForce 本身  = 7.7 趟（三趟 pass，均 2.6 趟/趟）【实测】
    CWLD 力（当前）     = 4.49 趟                     【实测】

    情形                                    f       端到端
    现状（4.49 趟）                       0.586    1.55x   【实测】
    融合建表（探针 C 直接测出）             0.519    1.70x   【实测】
    pair pass 降到 2.0 趟                 0.261    2.72x
    降到 1.5 趟  ← 目标                    0.196    3.21x
    降到 1.0 趟                           0.131    3.91x
    力免费（Amdahl 上限）                   0        6.93x

分解（全部单变量、同 binary）：

    pair pass                 0.519 f 单位   88.6%   = 1.56 CustomGB 遍历当量 = 4.49 普通遍历
    buildEnvPairs（摊销）      0.066          11.4%
    density + chain          ~0.000           0.0%   低于噪声
    每步阻塞同步（已去掉）      0.006           —

**融合建表只值 0.15x。pair pass 从 4.5 趟到 1.5 趟值 1.55x → 3.2x。**
所以除非有余力，**不要先做融合建表**。

## 2. 第一步：测，不要猜

**上 `ncu`。** 在此之前不要改 kernel —— 这个项目在性能上已经错过五次，每次都是
手边能测而去推了。

    ncu --set full --kernel-name computePairAndAdjoint \
        --launch-count 1 -o pair_profile \
        ./build-cuda/platforms/cuda/BenchmarkCudaLocalCWLD \
        ./build-cuda/platforms/cuda docs/reports/dec005_1aay_dpolar 100 100 1

要看的四个数，按重要性：

1. **Achieved Occupancy**。发射几何已经对了（272 blocks × 256 threads，`NonbondedUtilities`
   给的），但寄存器或 LOCAL 可能把实际占用率压下去。跟 OpenMM 自己的
   `computeNonbonded` kernel 比同一个数。
2. **Registers Per Thread / Local Memory Overhead**。寄存器溢出到 local memory 会
   直接吃掉几倍。算术核 + 6 个 LOCAL 数组，很可能溢出。
3. **Memory Throughput vs Compute Throughput**（roofline）。决定往哪个方向优化，
   两个方向的做法是相反的。
4. **Warp Stall Reasons**。`Stall Long Scoreboard`（等全局访存）vs
   `Stall MIO Throttle`（共享内存/特殊功能单元）vs `Stall Barrier`（`SYNC_WARPS`）。

## 3. 嫌疑清单（按优先级，一条都未验证）

### 3.1 `mixed` 能量累加 = 每对一次 fp64 加法
`energy += (mixed) pairEnergy;` 在内循环里。`Precision=mixed` 下 `mixed` 是 **double**，
而这张 2080 Ti 的 fp64 吞吐是 fp32 的 **1/32**。每步约 1.13e7 对。

⚠ OpenMM 自己的 `customGBEnergyN2.cc` 也这么写（`mixed energy = 0; energy += ...`），
所以这**不一定**是差异源 —— 但 CustomGBForce 正是那个 2.6 趟/趟的东西，而参照物是
1 趟的 `NonbondedForce`。**去查 `NonbondedForce` 的 kernel 怎么处理能量**，
以及 `includeEnergy=false` 时（动力学的每一步都是）能不能整个跳过。

**怎么测**：加一个编译期 define 把能量累加去掉，跑 benchmark。物理会错，纯计时探针。
如果 f 掉一大截，就做成 `includeEnergy` 的两个变体。

### 3.2 寄存器压力 / LOCAL 占用
我们 6 个 LOCAL 数组共 44 B/线程（256 线程 = 11.3 KB/块）：
`localPos(real4=16) localQ(4) localQbase(4) localForce(real3=12) localAdjoint(4) atomIndices(4)`。
OpenMM 的 `customGBEnergyN2` 只有 4 个共 32 B。

**怎么测**：ncu 的 occupancy + registers。**怎么改**：
* `localPos` 用 `real3`（12 B）而不是 `real4` —— `.w` 从来没用过，白占 4 B。
* `localQbase` 可以去掉：`dq2 = q2 - qb2` 里只用到 `qb2`，而 `qbase` 可以塞进
  `localPos.w`（那个位置本来就空着）。省 4 B 且少一次共享内存访问。
* `localAdjoint` 和 `localForce` 能不能合成一个 `real4`？（fx,fy,fz,adj）—— 16 B 对齐，
  少一次共享内存事务。

### 3.3 伴随量的额外原子操作
每原子每 tile 4 次原子操作（力 3 + 伴随量 1），非键只有 3 次。**+33%，不是 4.5x**，
所以单独不足以解释，但跟别的叠加。

### 3.4 `SYNC_WARPS` 在内循环里
每次迭代一次。OpenMM 也这么做，**大概率不是差异源**，但 ncu 的 `Stall Barrier` 能确认。

### 3.5 已排除
* **`atomIndices` 全局重读**（2026-09-04）：改成缓存进 LOCAL（跟 OpenMM 一致），
  **f 0.5839 → 0.5842，无收益**。那 32 个 int 是连续 128 字节、一两条 cache line，
  warp 第一次读完就全在 L1 里。改动保留（不再依赖缓存行为），但**不是瓶颈**。

## 4. 测量纪律（踩过的坑，别再踩）

1. **每个探针只改一件事，且必须能用"把它加回去"直接验证。**
   第 7 节和第 9 节两版成本模型都是因为探针不是单变量的而被推翻：
   第 7 节说 density+chain 占 0.549（实测 ~0）；第 9 节说同步占 0.099（实测 0.006，差 17 倍）。
2. **三个测量不要去解四个未知数。** 手边能直接测就直接测。
3. **跑 3 轮，不要跑 5 轮。** 这张卡热降频凶且锁不了频（无权限）：3 轮绝对速率掉 12–13%，
   5 轮掉 23%。轮转让**比值**免疫（比值散度 0.2–1.5%），但长跑只会更热不会更准。
4. **看 benchmark 的闸门。** 轮间比值散度 > 3% 会拒绝给判定。今天有两次跑被抢卡，
   都是它拦下来的；其中一次如果只看中位数会得到一个完全正常的数。
5. **判定的余量要跟散度比。** 现在 1.55x 对 1.5 线的余量约 +3%，散度 1.4%，
   只有 2 倍。要写进论文得跑 5 次取中位数。
6. **改完 kernel 必跑 `ctest`（7/7）。** 尤其 `TestCudaLocalCWLDDynamics`：
   静态验收测不到"该不该重建"这类错误，`maxTiles` 那个 bug 就是这么漏的。

## 5. 现成的工具

    # 三臂/四臂轮转 benchmark，带比值闸门
    ./build-cuda/platforms/cuda/BenchmarkCudaLocalCWLD <plugin dir> <dump prefix> \
        [warmup=1000] [measure=4000] [reps=3]

    # 计时探针（都是环境变量，都在 device 上生效）
    LOCALCWLD_PROBE_ENV_CUTOFF=1e-4   r_env 压到近零 ⇒ density+chain ≈ 0（物理错）
    LOCALCWLD_ALWAYS_REBUILD=1        每步重建（物理正确，只是更新鲜）
    LOCALCWLD_NEVER_REBUILD=1         永不重建（物理错，但直接给出建表成本）
    LOCALCWLD_PROBE_SYNC=1            把每步阻塞读加回来（单变量对照）
    LOCALCWLD_DEBUG=1                 每步阻塞检查 + 打印重建间隔。**会拖慢，别用它跑分**

    # 正确性
    cd build-cuda && ctest        # 7/7，其中 Dynamics 跑 400 步 MD

## 6. 不要做的事

* **不要合并 density + chain。** 实测合计 ~0.000 f 单位。设计文档第 2 节和第 7 节
  都建议过，两次都错。
* **不要动 padding。** `r_env + 0.1 = 0.45 nm`，降低会漏 pair（正确性），
  升高不减少重建次数（跟 OpenMM 的标志走）。peer 的参数扫描确认当前值在最优点。
* **不要引用第 7 / 9 节的外推表。** 都作废了，理由见第 10 节。
* **不要在没跑 `ncu` 之前改 kernel。**
