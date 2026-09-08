# LocalCWLDFastEvaluator：离线 CPU 加速后端

这是显式选择的 **analytic / float32 CPU** 后端，用于静态快照、参考计算和
测试迭代。它不是 OpenMM Force，没有接入 v2.6 的生产模拟，也没有取代原来的
`localcwld_reference`。DEC-002 的 tabulated 兼容模式仍未定案。

**精度：全链路 float32，没有精度开关**（见 `docs/decisions/DEC-004-precision.md`）。
生产轨迹本来就是单精度跑的，本后端的作用之一是忠实预演 GPU kernel 的数值行为。
float64 只保留在 `localcwld_reference`，它是 CPU 上的审计 oracle。

## 使用

依赖 NumPy、SciPy；无需安装新 JIT 或 GPU 库。将 `python/` 放入 PYTHONPATH：

```python
from localcwld_fast import LocalCWLDFastEvaluator

# particles：九个数组；globals_：localcwld_reference.Globals。
# 参数在构造时复制，后续修改调用者的数组不会改变 evaluator。
evaluator = LocalCWLDFastEvaluator(particles, exclusions, globals_)
result = evaluator.compute(positions_nm, box_vectors_nm)
energy = result.energy_total
forces = result.force_total

# 不请求能量时跳过能量计算，完整链式力仍然存在。
forces_only = evaluator.compute(positions_nm, box_vectors_nm, include_energy=False)
```

支持无周期盒、正交盒及 OpenMM reduced triclinic 盒。长度 nm、电荷 e、能量
kJ/mol、力 kJ/mol/nm，返回类型和分阶段诊断与 Reference 相同（但返回数组是
float32）。与 Reference 的一致性是**单精度量级**（约 1e-5 相对），不是逐位。
没有增大时间步，也没有降低 Q 的更新频率——降的只有算术精度。

每次 `compute()` 都重新构建邻居、density、Q、lambda 和力；不冻结 Q、不缓存
旧坐标或邻居。参数变化时显式创建新的 evaluator，坐标和盒子直接传入下一次
`compute()`。返回值不会持有可修改内部参数的引用。

## 加速点与边界

- 使用 SciPy 原生邻居搜索，不在 Python 中全体系两两枚举。
- 对三斜盒，在 fractional torus 上用 `rc/sigma_min(box)` 保守搜索，然后按
  Reference 的 Cartesian 最小镜像及真实 cutoff 过滤；不是把三斜盒当正交盒。
- density/pair/chain 共用一次邻居查询和 pair 几何，使用 NumPy 批量归约。
- 固定 source/response 系数与 closure 系数只准备一次。
- density 与 chain 不再各自重复开方/单位向量计算；pair 使用相消改写。
- 不响应粒子仍参与 pair 修正；same-residue 与 explicit exclusion 保持区别。
- 微小负坐标取模恰好舍入到盒边时规范化到等价周期端点。
- **原先那段"cutoff 附近极薄边界层用 scalar dot 复核以逐位匹配 Reference"
  的代码已删除。** float32 下 cutoff 归属在约 1e-7 相对范围内本就不确定，
  该复核既无意义又是性能陷阱。注意这个 closure 的 pair 力在 rc 处**不归零**，
  所以一次归属翻转值几十 kJ/mol/nm（实测约 29），不是舍入碎屑；概率很低
  （约 1e-7 的 pair），且今天的 CustomGBForce CUDA 路径已经如此。细节见 DEC-004。

邻居搜索使用 [cKDTree.query_pairs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_pairs.html)
的 `eps=0`，没有近似漏邻居。临时内存随候选邻居数量增长，不创建 N×N 距离
矩阵；密集体系仍可能有 O(N²) 邻居，因此不能无限扩展或据此保证线性复杂度。

**不要把本后端装进 GPU 模拟的逐步 Python callback。** 那会重新引入此前
KDTree 路径的 GPU→CPU 坐标回传问题。本次没有 GPU/CustomGB 性能比较，不能
把对 Python Reference 的加速倍数当作 MD 的 ns/day 提升。

## pytest 与完整求值 benchmark

在 `openmm-localcwld/` 下，使用装有 NumPy/SciPy/pytest 的解释器：

```bash
python -m pytest python/experiments/test_fast_evaluator.py -q
python -m pytest python/experiments/benchmark_fast_cpu.py -q -s
```

benchmark 文件不以 `test_` 开头，只有显式运行才计时。先验证全部返回字段，
再预热并交替执行五次，报告 median；**邻居表重建、density、Q、pair、完整
chain 和 energy 都在 compute 计时内**。固定参数准备时间单独记录，没有
计入重复求值，但需要单次使用者计入首次调用成本。不设置容易波动的性能
断言，不与不同精度、不同物理参数或只算部分力的路径比较。

可加 `-o junit_family=legacy --junitxml=/tmp/localcwld-speed.xml` 导出每次计时、
依赖版本、最大力误差和 median。原有输入契约实验中的 10 项失败属于旧
metadata/Reference，未在本次速度优化中修改，不代表完整插件验收通过。

## 实测（2026-09-03，float32 版本）

Python 3.12.13、NumPy 2.4.3、SciPy 1.17.1；合成周期体系，ell=2、penalty 开启，
同一输入、同时计算 energy 和 force，五次完整求值的 median：

| 粒子数 | Python Reference (f64) | Fast CPU (f32) | 加速 | 参数准备（另计） |
|---:|---:|---:|---:|---:|
| 32 | 18.178 ms | 0.558 ms | 32.6× | 0.251 ms |
| 128 | 282.995 ms | 1.454 ms | 194.7× | 0.200 ms |
| 512 | 4468.250 ms | 12.533 ms | 356.5× | 0.282 ms |

⚠ 这一列加速是"Python 双循环 float64 Reference → 向量化 float32 后端"，
**同时混了算法改动和精度改动**，不是 float32-vs-float64 的对照。

**float64 → float32 本身只值约 1.2×**（同一后端、同一算例对照）：

| 粒子数 | float64（2026-08-31） | float32（本次） | 提速 |
|---:|---:|---:|---:|
| 32 | 0.562 ms | 0.558 ms | 1.01× |
| 128 | 1.799 ms | 1.454 ms | 1.24× |
| 512 | 14.968 ms | 12.533 ms | 1.19× |

这个量级合理：这些规模下 cKDTree 邻居搜索和 Python 调用开销占大头，不是
float64 算术带宽。**单精度的真正收益在 GPU 上，不在这个 CPU 后端上。**

总力最大绝对差从 float64 时的 `4.26e-13 / 5.40e-13 / 9.66e-13` 变为
`3.82e-05 / 8.06e-05 / 1.55e-04 kJ/mol/nm`。
结果是本机观测值，不是跨机器性能保证；没有比较 GPU/CustomGB，也不能外推
为真实 1CKK 生产轨迹速度。缩短的是目前 Python 双循环 Reference 的静态求值。

验证结果：共 **156 passed**（其中后端回归 65 项，全部在 float32 容差带下通过）；
显式 benchmark 另 **3 passed**。7 个 golden JSON 及 fixture README 的 SHA256
校验全部通过。旧输入契约的 10 项失败保持可见，未更改 Reference 来绕过它们。
