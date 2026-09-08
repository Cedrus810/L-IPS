# DEC-004: 工作精度 = float32（GPU 与离线后端一致）

- 状态：**冻结**
- 日期：2026-09-03
- 工单：影响 LCWLD-070 / 090 / 100 / 110 / 120，以及既有的 `python/localcwld_fast/`
- 决定人：**用户**（本次交互中明确指示）
- 记录人：Claude (assistant)

## 决定

`LocalCWLDForce` 的计算精度统一为 **float32**：

1. 原生插件的 GPU kernel（LCWLD-100/110）以 float32 实现，不做 float64 变体，
   也不做 float32 计算 / float64 累加的 mixed 模式。
2. 离线后端 `python/localcwld_fast/` 从 float64 改为 **全链路 float32**：参数、
   坐标、几何、density、Q、能量、力全部 float32，**不提供精度开关**。
3. `python/localcwld_reference/` 保持 **float64 不变**。它是 CPU 上的审计用
   oracle，不是生产路径；float64 只允许存在于这里。

## 理由（用户口径）

> 「md 是 fp32 的，你 gpu 当然也要 fp32」
> 「fp64 那是给 cpu 的东西，不是给 gpu 的」

生产轨迹本来就跑在单精度上：`run_zn_job.py` 默认
`--platform CUDA --precision mixed`，OpenMM 的 CUDA platform 在这个设置下
**用单精度算力**（double 只出现在积分器的位置/速度累加里）。也就是说今天
`CustomGBForce` 给出的 CWLD 力已经是 float32 的。一个 float64 的离线后端衡量
的是没有人在跑的 kernel，既慢又给出虚假的精度承诺。

## 后果（已实测，不是推断）

### 1. 离线后端提速有限：约 1.2×

同一台机器、同一 benchmark（`python/experiments/benchmark_fast_cpu.py`，
合成周期体系、ell=2、penalty 开、完整 energy+force、五次 median）：

| 粒子数 | float64（2026-08-31） | float32（本次） | 提速 |
|---:|---:|---:|---:|
| 32  | 0.562 ms  | 0.558 ms  | 1.01× |
| 128 | 1.799 ms  | 1.454 ms  | 1.24× |
| 512 | 14.968 ms | 12.533 ms | 1.19× |

这个量级是合理的：在这些规模上 cKDTree 邻居搜索和 Python 调用开销占大头，
不是 float64 算术带宽。**float32 的真正收益在 GPU 上，不在这个 CPU 后端上。**
把 CPU 后端改成 float32 的主要价值是它现在忠实预演 GPU kernel 的数值行为。

### 2. 与 float64 Reference 的一致性从"近逐位"降到"单精度量级"

实测最大偏差（`benchmark_fast_cpu.py` 的 `force_max_abs_error`）：

| 粒子数 | float64 时 | float32 时 |
|---:|---:|---:|
| 32  | 4.26e-13 | 3.82e-05 |
| 128 | 5.40e-13 | 8.06e-05 |
| 512 | 9.66e-13 | 1.55e-04 |

`test_fast_evaluator.py` 的容差带因此改为 `rtol=2e-4, atol=2e-3`（实测最坏
约 2e-5 相对，留约一个量级余量）。这不是"把测试放松到能过"，而是继续把
容差压到 float32 eps（1.2e-7）以下只会在测试"float32 是不是 float64"。

### 3. 有限差分梯度检验必须改在 float64 Reference 上做

`h=1e-5 nm` 的中心差分要相减两个约 200 kJ/mol 的接近相等的能量。float32 下
剩下的位全是噪声——实测 `|F - FD| ≈ 2.19 kJ/mol/nm`，而力本身约 97，即 **2%
的噪声地板**；同一算例 float64 Reference 是 `3.5e-07`。

因此 `test_fast_candidate_total_force_finite_difference` 改名为
`test_total_force_finite_difference_against_reference`，改为对 float64
Reference 做有限差分（验证力表达式本身），再断言 float32 后端复现 Reference
的力。梯度覆盖没有丢，只是靶子换成了唯一能承载它的精度。

### 4. cutoff 边界的归属在 float32 下是**真的**不确定，而且代价不小

这一条需要 LCWLD-100 的实现者知道，不是学术注脚：

- 本 closure 的 pair **能量**在 `rc` 处归零（ell=1/2/3 的系数和都是 −1，
  `closure(rc) = 1/rc − 1/rc = 0`），但 pair **力不归零**。残余力为
  `k · qbase_i·qbase_j / rc³ · Δr`，是 shifted-potential 固有的截断不连续，
  由原生 NonbondedForce 那侧对消。
- 于是一个恰好落在 `rc` 附近约 1e-7 相对范围内的 pair，在 float32 下算进算
  不进是掷硬币，而翻转一次的力差是**几十 kJ/mol/nm 量级**，不是舍入碎屑。
  实测该跳变约 29 kJ/mol/nm。
- 概率上这很稀薄（约 1e-7 的 pair 落在该带内），对 ~10⁷ pair 的体系大约每帧
  1 个 pair、影响 2 个原子，相对于典型 ~10³ kJ/mol/nm 的原子受力可以接受，
  **且今天的 CustomGBForce CUDA 路径已经在这么做**。但它不能被当作零。

原 float64 后端里那段"在 cutoff 附近极薄边界层用 scalar dot 复核、以逐位匹配
Reference"的代码已经删除：float32 下它既无意义（边界带宽到 1e-7，本就不确定）
又是性能陷阱（落入该带的 pair 数量大到那个 Python 循环会真的变慢）。

`test_fast_cutoff_roundoff_in_arbitrary_directions` 相应改为：明确可分辨的
内/外侧（偏移 16 个 float32 ULP，盖过旋转带来的约 √3 ULP 长度误差）照常断言
一致；**恰好在 cutoff 上**则断言结果必须等于"算进"或"算不进"两个合法分支之
一——把不确定性写成断言，而不是靠放宽容差掩盖掉。

### 5. numpy 的 bincount 没有 float32 版本

`np.bincount` 只有 float64 累加实现，`_sum()` 因此在内部走 float64 再窄化回
float32。这是库的限制，不是精度选择：离开该函数的每个值都是 float32。GPU 端
会用原生 float32 scatter-add，所以那边的逐粒子求和与这里可能相差 float32 舍入。

## 对 LCWLD-120（数值总门禁）的影响

计划第 9/25 节的测试阈值是在"Reference 为 double"的前提下写的。LCWLD-120 落地
时必须按本决定重写阈值表：float32 kernel 对 float64 golden 的验收带应取本文件
第 2 节实测量级（约 `rtol=2e-4`），不能沿用原表。**这一条尚未执行**，只是在此
标记，避免到时候拿着 double 阈值去卡一个单精度 kernel。

## 未做的事

- 没有重新生成任何 golden fixture。7 个 golden 仍是 float64，SHA256 不变；
  改的只是拿它们比对时的容差。
- 没有改 `localcwld_reference/reference.py` 一行。
- 没有跑任何 MD 轨迹或分析。
