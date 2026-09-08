# pytest 实验：代码优化与输入契约

只运行小型 CPU 数值实验；不启动 MD，不创建 OpenMM Context，不覆盖 golden。
候选公式仅位于实验测试内，不修改生产 Reference 或 v2.6。

后续速度实现已独立放在 `python/localcwld_fast/`，仍不修改原 Reference 或
v2.6。新增端到端回归 `test_fast_evaluator.py`；显式性能实验运行
`python -m pytest python/experiments/benchmark_fast_cpu.py -q -s`。
详见 [快速 CPU 后端说明](../localcwld_fast/README.md)。

在 `openmm-localcwld/` 下运行（需要 numpy、pytest；输入契约测试另需 OpenMM）：

```bash
python -m pytest python/experiments/test_numerical_candidates.py -q -s
python -m pytest python/experiments/test_input_contracts.py -q --tb=short
python -m pytest python/tests/ -q
```

当前工作区可用解释器：
`/home/ruigengji/miniforge3/envs/openmm_dev/bin/python3.12`。

## 数值实验

- 固定种子的 8 粒子三斜盒，非对称 source/sink、same-residue、显式 exclusion。
- ZMM ell=1/2/3、Q penalty 开/关。
- 预计算参数、解析 density/chain 去开方与 Reference 的分阶段对照。
- `r_env` 内侧、边界、外侧；energy/force 四种请求组合。
- 粒子重排、逐粒子晶格平移、整体平移及总力检查。
- 总能量中心有限差分，h=1e-4/1e-5/1e-6，检查误差收敛。
- pair 相消改写的 float64 等价性；float32 原式/改式对照 70 位 Decimal。

float32 实验固定扫描 r=0.08–0.6 nm，包含固定电荷、小响应和较大响应，比较
整体 RMSE 并记录最大绝对误差。高精度对照使用相同的 float32 输入值，避免
混入输入表示误差。输出为 `(dE/dr)/C`，未乘库仑常数。

候选去开方表达式只适用于解析 density kernel，不决定 DEC-002 的
analytic/tabulated 选择。A/D 对照沿用 Reference 的 PBC、Q/dQ/lambda，验证的
是局部代数等价性，不是完整优化实现。NumPy float32 不是 GPU 模拟器，不覆盖
GPU FMA、并行归约、原子累积和性能；本实验通过不代表 GPU 或发布验收通过。

测试使用 `record_property` 保存误差，可添加 `--junitxml=/tmp/cwld-experiments.xml`
导出结果（需要属性时使用 `-o junit_family=legacy`）。未设置运行时间门槛：
Python 执行时间不能代替 GPU kernel benchmark。

## 输入契约实验

`test_input_contracts.py` 描述期望的拒绝行为，当前实现存在已知缺口，因此预计
会失败。未使用 skip/xfail，也不通过断言“接受非法输入”来制造绿灯。失败表示
生产校验尚未修复，不能视为完成。数值实验与此文件可以分别运行；运行整个
`python/experiments/` 会包含这些真实失败。

覆盖粒子数不匹配、缺失/多个 NonbondedForce、二维参数数组、非有限常量/
坐标和越界 exclusion。冻结 dataclass 的深层可变性等设计选择不混作硬性契约。

## 首轮实测（2026-08-31）

- 数值候选实验：**69 passed**。
- 输入契约：**10 failed**，均为上述生产校验缺口；缺失 NonbondedForce 时抛出
  `StopIteration` 而非约定的 `ValueError`，其余用例未拒绝非法输入。
- 原有 `python/tests/`：**22 passed**。
- 7 个 golden JSON 及 fixture README 的 SHA256 校验全部通过。
- 6 组 ell/penalty 组合的总力有限差分均随步长缩小收敛；h=1e-6 时最大绝对
  误差约 `8.41e-8`–`1.47e-7 kJ/mol/nm`。

float32 pair 导数误差 RMSE（未乘 C，取输出约 6 位有效数字）：

| ell | deltaQ_i（deltaQ_j=-2 deltaQ_i） | 原式 | 改式 |
|---|---:|---:|---:|
| 1 | 0 | 5.48828e-6 | 2.19059e-8 |
| 1 | 1e-8 | 5.99650e-6 | 2.81480e-8 |
| 1 | 1e-3 | 7.26773e-6 | 3.63488e-8 |
| 2 | 0 | 5.70023e-6 | 5.64902e-8 |
| 2 | 1e-8 | 6.16794e-6 | 6.37072e-8 |
| 2 | 1e-3 | 7.16439e-6 | 7.76790e-8 |
| 3 | 0 | 5.95633e-6 | 7.56721e-8 |
| 3 | 1e-8 | 6.80634e-6 | 9.42537e-8 |
| 3 | 1e-3 | 7.71810e-6 | 1.11573e-7 |

这里的改善约 69–251 倍，**只针对声明的扫描域内这个标量公式**，不是所有
输入的保证，也不是完整 GPU 力误差或加速倍数。尚未进行真实 GPU、tabulated
兼容模式或端到端候选实现的验证。
