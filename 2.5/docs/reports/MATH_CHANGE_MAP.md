# 大改数学逻辑前：公式到底散在哪儿

> 2026-08-31 建。CWLD 的 **同一套数学有 4 份独立实现**，另外有一批冻结 sha256 的
> fixture 和两份冻结决策文档会把旧公式钉住。只改其中一处，剩下的会静默不一致，
> 而且不一致的方式是"结果照样跑出来、数字看着正常"。

> **2026-08-31 追加**：pair closure 的改动走独立路线，设计见
> [`PLAN_PSWF_Closure_Family.md`](PLAN_PSWF_Closure_Family.md)（窗函数族 / PSWF）。
> 该路线**只动 closure `U_ell(r)`，不动密度核 `K(r)`**，因此下表 #3（分析侧镜像）
> 与 DEC-002 不受影响；需要同步的是 #1/#2、#4、#5。

## 数学链条（当前 v2.6 语义）

```
dens_i  = Σ_j  dens_sink_i · dens_source_j · source_class_weight_j · charge_mod_j
                 · K(r_ij) · step(|mol_id_i − mol_id_j| − 0.5)          ← 排除同分子
K(r)    = (1 − (r/r_env)²)²      for r < r_env, else 0                   ← Lucy-like，紧支撑
Q_i     = qbase_i + q_delta_clamp · tanh( (static_phase_i · is_polar_i · dpolar_i
                                          · tanh(k_polar·dens_i/rho0)) / q_delta_clamp )
U_pair  = ONE_4PI_EPS0 · [ Q_i Q_j · U_ell(r) − qbase_i qbase_j · (1/r − 1/rc) ]
                                     └ Sakuraba ZMM α=0 closure   └ 抵消原生 NonbondedForce 的 RF
U_self  = ½ · k_penalty · (Q_i − qbase_i)²                                ← ENABLE_Q_PENALTY 才有
```

力里还有 **链式响应项** `dU/dQ · dQ/ddens · ddens/dR`（就是 CustomGBForce 帮你自动
求的那部分）。改 `Q(dens)` 或 `K(r)` 的形式，这一项跟着变；手写实现里必须同步改
解析导数，OpenMM 那边不用。

## 4 份实现，改动必须同时落地

| # | 位置 | 角色 | 改的时候要动什么 |
| --- | --- | --- | --- |
| 1 | `test_lips_vs_pmeV2.6.py::setup_cwld_lips_system`（约 1146–1215 行） | **生产 MD**：CustomGBForce 的 Lepton 字符串（`density_expr` / `q_expr` / `pair_expr` / `self_expr`）+ `addTabulatedFunction("density_kernel")` | 表达式字符串；导数由 OpenMM 自动求，不用手写 |
| 2 | `test_lips_vs_pmeV2.6.py::zmm_uclosure` + `pair_uclosure`（约 49–110 行） | closure `U(r)`。**2026-09-01 起有两条路**：`L_IPS_CLOSURE=zmm`（默认，读 `L_IPS_ZMM_ELL` ∈1/2/3）走 `zmm_uclosure()` 的冻结字符串；`=pswfz2` 走 `closure_windows.py` 的窗层 | 改 ZMM 分支要动那三行硬编码字符串（**改了会让 fixture 的 1e-12 断言变红**）；加新窗只需在 `closure_windows.WINDOW_REGISTRY` 注册，不碰这里 |
| 3 | `test_lips_vs_pmeV2.6.py` 的**分析侧镜像**：`compute_fast_cwld_q_from_positions`(666)、`_local_dens_at`(862)、`_delta_q_from_dens`(884)、`compute_water_q_profile`(736)、`compute_water_net_delta_q`(945) | 事后从 dcd 重算 q-profile 的 numpy/KDTree 路径。注释里明写 "mirroring the exact CustomGBForce density_expr" | **最容易漏**。漏了的话 MD 用新公式、q-profile 图用旧公式，图完全是假的 |
| 4 | `openmm-localcwld/python/localcwld_reference/reference.py`：`zmm_uclosure`(34)、`zmm_duclosure_dr`(55)、`compute_local_cwld`(158) | 插件的 golden 参考实现（纯 numpy 显式循环） | 能量式**和解析导数** `zmm_duclosure_dr`、`lambda_` 链式项都要手改 |
| 5 | `openmm-localcwld/python/localcwld_fast/evaluator.py::LocalCWLDFastEvaluator`（closure 系数表在 81 行，tanh 链在 183–184 行） | 加速 CPU 后端 | 同上，另外它把系数预计算成表，改 closure 要改表 |

（#1–#3 在同一个文件里，但是三条彼此独立的代码路径。）

## closure 家族入口（2026-09-01 新增）

```
L_IPS_CLOSURE = zmm (默认) | pswfz2
  └ =zmm 时继续读 L_IPS_ZMM_ELL ∈ {1,2,3}，默认 2 —— 语义一字未变
```

**默认路径不导入窗层**，三档 Lepton 串对改动前的备份
`test_lips_vs_pmeV2.6.py.bak-before-closure-entry` 逐字节相同（已验）。

配套工具（都不跑 MD，除 NVE 那个）：

| 文件 | 干什么 |
| --- | --- |
| `closure_windows.py` | 窗层。`U(r)=(1-A(x))/r, A=∫₀ˣχ`。ZMM 三档构造时即断言与 v2.6 冻结系数逐位一致；新窗 `pswfz2` 冻结在 `FROZEN_PSWFZ2_CHI_EVEN`。`python closure_windows.py` 跑自检 |
| `closure_force_audit.py` | **静态力审计**：固定坐标/电荷/排除表，参照 PME 精确 Ewald，只换 kernel。零 `Integrator.step()`。判 closure 优劣就用它 |
| `closure_nve_drift.py` | NVE 能量漂移（力审计量不到的通道）。带 `--selftest`，**必须用 `--platform CUDA` 才能覆盖「同组 cutoff 必须一致」那条约束，Reference 上不报错** |
| `measure_szz.py` | 从已有 dcd 实测电荷结构因子 S_ZZ(k)（2026-09-03 由 `measure_szz_1ckk.py` 改名并把体系参数化；阶段 A 默认走 GPU/float32） |
| `hybrid_lowk.py`（2.5：`lips-hybrid-lowk`，2026-09-07 新增） | **路线 B 的 B0 关**：纯 numpy 单帧，local closure + low-$k$ reciprocal 修正 $w(k)=\hat c(kr_c)$ 对精确 Ewald 的力误差。自带 CHECK-1（split 恒等式）/2（参照对 α 无关）/3（$k_c\to$大 收敛）。**改 closure 数学后这三个 CHECK 必须重跑**——CHECK-1 是 $\hat L=(4\pi/k^2)(1-\hat c)$ 的直接验证，形式或系数推错就在那里炸 |
| `closure.py::pcf_window`（2026-09-07 新增） | **PCF**：$\chi_{\ell,c}=(1-x^2)^\ell\psi_0^c/\!\int$，注册名 `pcf{ell}_c{c}`。表示走偶多项式拟合（残差 4e-15）⇒ **上表 #1/#2/#4/#5 一个字节都不用改**。`pswfz2`/`FROZEN_PSWFZ2_CHI_EVEN` 等旧名一律保留为别名 |
| `stage1_closure_kill_switch.py` | 窗族的解析判据。**⚠ 已被力审计推翻**：推导假设全对求和，漏了排除对，在 2–5% 量级上会翻符号。只在 >10% 的差别上参考 |

## ⚠ 已知的历史数据问题

`2.4` 的 5ns×3seed 矩阵**closure 不一致**：`PME`/`aq0_ca_w1`/`aq0_ca_w2`（07-11）是 ell=1，
`aq0p5_ca_w2`（07-14）是 ell=2。参数选择结论因此被混杂，
详见 `README_v2.6_1CKK_CONCLUSION.md` 顶部警告框。

## 会把旧公式钉住的"冻结物"

| 东西 | 位置 | 改数学后怎么办 |
| --- | --- | --- |
| 9 个数值 fixture + sha256 冻结清单 | `openmm-localcwld/tests/fixtures/*.json` + `sha256sums.txt` | 旧 fixture 的 E/F 是按旧公式算的，新公式下 `python/tests/test_reference_fixtures.py` 会红。要用 `docs/reports/generate_lcwld_030_fixtures.py` / `generate_water_ca_cluster_fixture.py` **重新生成并更新 sha256sums.txt**，同时在 `DEC-030` 里记明为什么换 |
| `PLAN_...md` 第 5 节「必须保持一致的数学语义」、第 18 节「可直接编码的数学规格」（18.1 density / 18.2 Q / 18.3 ZMM / 18.4 pair / 18.5 链式响应力 / 18.6 Q penalty）、第 31 节逐行伪代码 | `PLAN_LocalCWLDForce_OpenMM_Plugin.md` | 这是插件 CUDA kernel 的**唯一规格来源**。不同步改，将来写 kernel 的 agent 会照旧公式实现 |
| `DEC-002`（density kernel：analytic vs 1024 点 tabulated） | `openmm-localcwld/docs/decisions/DEC-002-density-kernel.md` | 状态是"证据已收集、结论待签署"。如果 `K(r)` 的函数形式要变，这份证据（在旧 K 上做的）直接作废，得重跑 `dec002_kernel_comparison.py` |
| `README_v2.6_1CKK_CONCLUSION.md` / `v2.6_results_analysis.md` / `1ckk_v26_replicate_matrix_analysis.md` | 2.5 根目录 | 里面的结论是旧公式下的 5 ns 结果。新公式跑出来之前，别把这些数字当新方法的基线 |

## 旧结果/轨迹在哪

数据（88 个 dcd、全部 csv/png、作业日志、`0625/0706/0707/MTS/OLD/...`）都留在
`/home/ruigengji/L-IPS/2.4/`，2.5 只放代码。要对旧轨迹做复分析：

```bash
L_IPS_DATA_DIR=/home/ruigengji/L-IPS/2.4 python block_time_check_v26.py
```

（`L_IPS_DATA_DIR` 只影响**读**输入轨迹；输出还是写到脚本所在目录，不会污染 2.4。）

## 平台

`select_platform()` 现在硬要求 CUDA，不再静默退回 CPU。调试用
`L_IPS_PLATFORM=cpu`（会打印醒目警告，产出不算生产数据）。
