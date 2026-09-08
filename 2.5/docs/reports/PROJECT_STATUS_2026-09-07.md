# L-IPS / CWLD 项目总结与进度报告

> 2026-09-07。范围：`/home/ruigengji/L-IPS/2.5`（代码）+ `2.4`（历史数据）+
> `openmm-localcwld`（原生插件）。
>
> 本文是**汇总**。绝大多数数值取自既有报告并标注来源；少数在汇总过程中新做的推导或
> 新发现的代码语义问题，一律用「**（本次汇总新增）**」标出，并说明验证到什么程度。
> 未落盘的数字显式标为「待复现」。

**目录**

1. [一页纸](#1-一页纸) · 2. [符号与参数总表](#2-符号与参数总表) · 3. [数学](#3-数学) ·
4. [PSWF 能不能解决 ZMM 的长波长问题](#4-pswf-能不能解决-zmm-的长波长问题) ·
5. [逻辑功能](#5-逻辑功能) · 6. [五份实现](#6-同一套数学的五份实现) ·
7. [现状](#7-现状) · 8. [方法论教训](#8-方法论教训) · 9. [缺口与待办](#9-缺口与待办) ·
10. [引用纪律](#10-引用纪律这些数字现在不能用) · 11. [环境](#11-环境) · 12. [复现路径](#12-复现路径)

---

## 0. 接手须知（下次从这里读，30 秒）

> 2026-09-07 收盘。**本页只写状态与待办；论证在 §1 之后。**

### 0.1 🔴 全部改动未提交

`/home/ruigengji/L-IPS/.git` 上次提交是 2026-05-26，且 `/home/ruigengji` 是 **NFSv3，
拒写 mode 0444**（git 松散对象正是 0444 建的）+ 仓库属主非当前用户（dubious ownership）
⇒ **在这棵树里 git 写操作会失败**。
`git config --global --add safe.directory /home/ruigengji/L-IPS` 只解决后者。
⇒ **要提交得从别的挂载点、或以属主身份操作。今天 22 个文件的改动全在工作树里。**

### 0.2 待跑（⚠ 用户 2026-09-07 要求：本地正在跑测试，**不要占用计算**）

| 命令 | 为什么 | 代价 |
|---|---|---|
| ~~`pytest tests/ -q`~~ | ✅ **已跑：47 passed（51.96 s）** ⇒ tag 改动已验证 | — |
| `python -m lips.analysis.cross_run` | 已跑过；改了派生量之后要重跑（`--force`） | 秒 |
| ell=1 生产跑 | §9.0-A/B 已放行，见 §0.4 的命令 | 3×5 ns |

### 0.3 三个待决定（都不是测量问题）

1. **换不换 zmm1 —— 证据齐了，但推荐「不改代码默认值」，见 §9.0-C。**
   力精度 4 系综一致（好 7.4%/第一壳 15%）、NVE **不区分**（0.04–0.69σ，dt² 已验证）、
   ell=1 力验收通过（p99 4.7e-3 vs ell=2 的 4.3e-3）、tag 已带阶数、47 passed。
2. **§13 hybrid 的范围。** 它撞在 `PLAN` §2 写死的非目标「不引入 FFT/reciprocal space」上，
   是 assistant 提出的扩展 ⇒ 需用户显式裁定。且卡 **H1（净电荷 +8.3 e，已量化）**
   与 **H2（响应一致性，4 通道 + 中性约束这第 5 个）**。
2b. **PCF on GPU**（§14.7b）：路径已查清 —— kernel 的 Horner 系数表是通用的、
   `c0..c6` 是运行时参数，只差一条「4 个偶次系数」的 API 入口；PCF `deg=2` 装得进，
   代价 8e-6（比生产已接受的引擎间差异小 400 倍）。**是一次对 C++ 插件的真实改动，需点头。**
3. **1CKK 负对照**（§9.6）：「CWLD 不把 PME 本来就对的体系搞坏」**换体系后从未复核**，
   而这是审稿人必问。需要 MD。

### 0.4 ell=1 生产跑的命令（放行了，等机时）

```bash
L_IPS_ZMM_ELL=1 L_IPS_LIGAND_DPOLAR=-0.15 lips-run-zn --system 1aay
#   -> ZN_1aay_cwld_ligm0p15metal_zmm1_plugin_seed{0,1,2}_traj.dcd
```

⚠ 这是**一套新轨迹**，与现有 `_zmm_`（=ell=2）那批**不能混进同一统计**。
要比 zmm1 vs zmm2 的科学结果，两条路：两边都在新命名下各跑一套（贵一倍），
或**声明「旧的 `_zmm_` 就是 zmm2」再对照**（口径已写进 §9.0-B，推荐这条）。

### 0.5 今天动过的文件（22 个，两个会话）

**本会话**：`engine/v26.py`（桥参数 + 非 ZMM 拒绝）、`run/zn_job.py`（closure tag）、
`paths.py`（`result_new` / `run_sig`）、`systems.py`（`assert_traj_matches_system`）、
`analysis/{force_audit,nve_drift}.py`（产物名 + 前置校验 + 分块斜率判据）、
`analysis/dens_parity.py`（修 3 个 bug）、`analysis/{cross_run,charge_budget}.py`（**新增**）、
`tests/test_plugin_globals.py`（**新增**）、`tests/test_cwld_engine_switch.py`（fixture 从 dump 推 ell）、
`docs/reports/{PROJECT_STATUS_2026-09-07(新增),REPORT_1AAY_ZN,STAGE1_CLOSURE_KILLSWITCH_REPORT}.md`、
`docs/plans/PLAN_PSWF_Closure_Family.md`（§11–14）。

**peer session `2-5-9e`**：`engine/closure.py`（`pcf_window` + `pcf{ell}_c0` 别名 + 修 `chat` 混叠）、
`analysis/{hybrid_lowk(新增),kill_switch}.py`、`tests/{test_closure_pcf(新增),test_lint}.py`、
`pyproject.toml`、`docs/reports/{REPORT_HYBRID_LOWK_B0_2026-09-07(新增),MATH_CHANGE_MAP}.md`。

### 0.6 读之前先看 §10

§10 是**引用纪律表** —— 今天有 **11 个数字/说法被撤回**（4.8× 模式节省、NVE 的 0.894、
截断组的 κ≈30、「低 k 差 17 倍」、「S_ZZ 比值发散」、全类天花板 1.39×、「窗只能动 6%」、
「PCF 是退路」、「计划的目标是恢复 Coulomb 长波」、$\hat c$ 的两组手算值、
「CWLD 极化只贡献 1.6%」）。**引用本报告任何数字前先扫一眼那张表。**

---

## 1. 一页纸

**做的事**：给固定电荷力场加一层**局部环境驱动的隐式电荷响应**（CWLD =
Charge-from-Weighted-Local-Density）。用一个标量「局部密度」`dens` 调制每个原子的
电荷 `Q`，从而在**不引入任何显式极化自由度**（无 Drude 粒子、无诱导偶极自洽迭代、
无额外积分自由度）的前提下，拿到极化力场级别的局部化学行为。

**为什么这个赌注可能赢**：显式极化的代价主要来自自洽迭代（AMOEBA 每步解诱导偶极）。
CWLD 把「响应」压缩成一个**瞬时的、局域的、解析可微的标量映射** $R\to\mathrm{dens}\to Q$，
于是响应力只是一次额外的链式反传，没有迭代。代价是丢掉了各向异性（只有电荷，没有偶极/
四极的方向性）和非局域性（$r_{\rm env}=0.35$ nm 之外没有响应）。
**项目要回答的就是：这个压缩在化学上够不够用。**

**参照系两端**：PME + 固定电荷（下界）与 AMOEBA 极化力场（上界）。

**目前最强的证据**（`REPORT_1AAY_ZN.md`）：1AAY Cys₂His₂ 锌指，5 ns × 3 seed × 9 个 Zn 位点：

| 方法 | 四配位核心完整时间占比 | Zn–N (Å) |
|---|---|---|
| AMOEBA（1 ns × 1，仅作锚点） | 100.0% | 2.135 ± 0.012 |
| **CWLD + 配位原子 dpolar 覆盖** | **100.0%**（9/9 位点） | **1.912 ± 0.002** |
| CWLD（仅修 Zn 密度源识别） | 62.9% ± 13.5% | 2.871 ± 0.348 |
| PME 固定电荷 | 66.9% ± 12.2% | 3.140 ± 0.443 |
| PME + Zn 电荷转移 q=1.5 | 3.0% | 5.637 |

Mann-Whitney（样本单位 = 位点，不假设正态，适合零方差组）：
CWLD vs PME 核心完整 **p=0.0053**、Zn–N **p=0.0004**。

**工程侧**：CWLD 的生产实现已从 `CustomGBForce` 换成自写 CUDA 插件 `LocalCWLDForce`，
端到端 **3.1–3.6×**（2080 Ti）/ **2.80–3.28×**（3090），CWLD 从 PME 速度的 8.2% 提到
**42.6%**，29×5 ns 的矩阵从 48.4 GPU-h 降到 14.4。

**当前阶段（2026-09-07 收盘）**：**主结果已成立、性能已达标，
当日把"验证与可复现性"这个缺口关掉了大半。**

### 1.1 今天关掉的四件事

| # | 事项 | 结果 | 产物 |
|---|---|---|---|
| 1 | 分析侧密度实现从未做 parity（5 份实现里最后一份） | ✅ **PASS**，max rel **1.7e-16** | `logs/dens_parity_*.log`（§9.1-R） |
| 2 | 两个 Δq 数字无落盘、无法复现 | ✅ **11.5% 复现为 11.73%**（12 真配位原子 + 平衡 MD 帧） | 同上（§9.2-R） |
| 3 | closure 阶数 $\ell$ 的选择 | ✅ **证据链闭合：应换 zmm1**。力精度 4 系综一致（好 7.4%/第一壳 15%）；NVE **不区分**（0.04–0.69σ，dt² 已验证 0.27–0.28 对 0.25） | `closure_force_audit_*`、`closure_nve_drift_dt{1,0.5}fs_mixed_2k*`（§7.3-R / §9.3-R4） |
| 4 | PCF（§2 目标 1）能不能立住 | ✅ **立住，且拿到正面结构结论**（下面 theorem 与 measurement 分层） | `cross_run_pcf_c2_scaling.csv`（PLAN §14.6c） |

**关于第 4 条，必须把定理与实测分开写**（2026-09-07 用户要求）：

* **定理（结构性，不依赖任何具体判据）**：$\chi_{\ell,c}=\chi_{\ell,0}+O(c^2)$
  （因集中算子提出正因子 $c$ 后余下的 $\mathrm{sinc}(cu)$ 对 $c$ 为偶，而正标量因子不改变
  特征向量）⇒ 对**任意**对 $\chi$ 光滑的泛函 $J$，
  $J[\chi_{\ell,c}]=J[\chi_{\ell,0}]+O(c^2)$，故 $\left.dJ/dc\right|_{c=0}=0$。
  ⇒ **$c=0$（=ZMM）是每个光滑判据的驻点。**
* **实测（当前两个具体判据的曲率方向，不是由驻点推出的）**：
  在**静态力**判据上 $c=0$ 对应**局域力最佳**（沿 $c>0$ 单调变坏，$\Delta/c^2\to0.0030$）；
  而**带外残差**沿非零 $c$ 方向**先改善**（$\Delta/c^2\to-0.18$，且 $\ell$=1/2 在
  $c\approx4$/$2.5$ 有真正的**内点最优**）⇒ 形成约 **40–60:1** 的「局域力–谱整形」兑换。

⚠ **「力最优 / 谱最劣」是实测结论，不是定理推论**；驻点只保证一阶导为零，不定曲率符号。

### 1.2 今天被实测撤回的（一并记在 §10）

**4.8× 模式节省**（实测不单调、最严处 0.68×）、**NVE 的 0.894**（短窗单 seed 的产物，
正确值 0.972 落在噪声里）、**「PCF 是精度路线的退路」这个框架**（因果反了，§2 目标 1 本身就是它）、
**给计划安了一个不存在的目标**（"恢复 Coulomb 长波"）。

⚠ **当日最值得记的模式：四个闸门里有三个是先修判据才给出可用答案的** ——
lint 门对语法错误失明（只读 stdout、漏 stderr）、NVE 线性度判据被对称鼓包骗过、
产物名两次会互相覆盖。**判据本身出错的次数多于物理出错的次数**（§8 教训 #13–15）。

### 1.3 派生数字的可复现性（本报告的自我约束）

本报告与 `../plans/PLAN_PSWF_Closure_Family.md` 里的**全部派生量**
（dt 标度比、$A/dt+B\,dt^2$ 拆解、seed sem、$\sigma$ 检验、对 zmm2 归一、$O(c^2)$ 系数、兑换率）
现在都能由一条命令从盘上 CSV 重算：

```bash
python -m lips.analysis.cross_run       # -> cross_run_*.csv（5 个）
```

**为什么专门加这个**：这些数字原本是当日用一次性 `python -c` 算完直接写进 markdown 的——
与 §9.2 那两个丢掉来源的 Δq 数字（0.75% / 11.5%）**是同一种失败**。
散文里的数字不算落盘。

### 1.4 剩下的（都是决策，不是测量）

* **换不换 zmm1** —— **证据齐、tag 阻塞已关闭（§9.0-B，47 passed 已验），
  只剩「是否切默认」这个决策**（推荐不改代码默认值，见 §9.0-C）。
* **§13 hybrid 的范围** —— 它撞在 `PLAN` §2 写死的非目标「不引入 FFT/reciprocal space」上，
  是 assistant 提出的范围扩展，需用户显式裁定；且卡 **H1（净电荷 +8.3 e，已量化）**
  / **H2（响应一致性，4 通道 + 中性约束这第 5 个）**。
* **仓库不在有效版本控制下**（NFS 拒写 0444 + dubious ownership）⇒ 今天所有改动**均未提交**。

---

## 2. 符号与参数总表

| 符号 | 代码名 | 含义 | 生产值 |
|---|---|---|---|
| $r_{\rm env}$ | `r_env` | 密度核支撑半径 | 0.35 nm |
| $r_c$ | `rc` | closure / 原生 NB 共用 cutoff | 1.2 nm |
| $r_{\rm on}$ | `r_on` | switching 起点，**只作用于原生 NB** | 0.9 nm |
| $\rho_0$ | `rho0` | 密度标度（内层 tanh 的分母） | 13.5 |
| $k_{\rm polar}$ | `k_polar` | 内层 tanh 增益 | 0.8 |
| $q_{\rm clamp}$ | `Q_DELTA_CLAMP` | $\Delta q$ 软上限 | 0.20 e |
| $a_{q^2}$ | `DEFAULT_A_Q2` | charge_mod 斜率 | 0.5 |
| — | `CHARGE_MOD_{MIN,MAX}` | charge_mod 硬 clip | [0.25, 2.5] |
| $\ell$ | `L_IPS_ZMM_ELL` | closure 阶（$U$ 在 $r_c$ 的光滑阶） | 2 |
| $C$ | `ONE_4PI_EPS0` | 库仑常数 | 138.935458 |
| $k_{\rm penalty}$ | `k_penalty` | Q penalty 劲度（**默认停用**） | 180.0 |
| $S_i$ | `dens_sink` | 「我响应吗」 | 0/1 |
| $B_j$ | — | $\mathrm{dens\_source}\cdot w^{\rm class}\cdot \mathrm{cmod}$ | 见 §3.2 |
| $A_i$ | — | $\mathrm{static\_phase}\cdot\mathrm{is\_polar}\cdot\mathrm{dpolar}$ | 见 §3.4 |

**source class weight**（谁作为环境更「重」）：

| 类 | 常量 | 值 |
|---|---|---|
| 水 O | `WATER_SOURCE_WEIGHT` | 0.5 |
| 单价离子 Na⁺/Cl⁻ | `ION_SOURCE_WEIGHT` | 2.0 |
| 二价金属 Ca²⁺/Mg²⁺/Zn²⁺ | `CA_SOURCE_WEIGHT` | 2.0 |
| 蛋白带电侧链位点 | `CHARGED_SOURCE_WEIGHT` | 1.5 |
| 配体带电位点 | `LIGAND_CHARGED_SOURCE_WEIGHT` | 1.5 |
| 其它极性重原子 | `POLAR_SOURCE_WEIGHT` | 1.0 |
| 脂头基 | `LIPID_HEADGROUP_SOURCE_WEIGHT` | 0.3 |

**dpolar**（响应幅度与符号）：

| 对象 | 值 | 占 clamp | 符号含义 |
|---|---|---|---|
| 水 O | −0.15 | 75% | 高密度环境下**更负** |
| 水 H（各） | **+0.075 = −0.5 × dpolar_O** | 37.5% | 为分子电中性设计，见 §3.4.3 |
| 溶质 N / O（按元素） | +0.012 | 6% | **变正 ⇒ 削弱**对阳离子的吸引 |
| 溶质 S（按元素） | +0.015 | 7.5% | 同上 |
| **配位原子覆盖** | **−0.15** | 75% | `L_IPS_LIGAND_DPOLAR`，默认关闭 |

**MD 协议**：300 K / 1 bar / dt 2 fs / 每 10 ps 存帧；平衡 NVT 50 ps + NPT 50 ps；
生产 5 ns（矩阵）或 500 ps（快速消融）；production 段关压浴。

---

## 3. 数学

### 3.1 总链条

```
     坐标 R
        │  ① 密度核 K(r)，紧支撑于 r_env
        ▼
     dens_i  ──② 双层 tanh 响应──▶  Q_i
                                     │  ③ pair closure U_ell(r) + RF 抵消
                                     ▼
                                  U_pair(R)

  力 = −∂U/∂R│_Q  −  Σ_i (∂U/∂Q_i)(dQ_i/ddens_i)(∂dens_i/∂R)
        └ direct ┘     └────────────── ④ 链式响应力 ──────────────┘
```

这四个阶段就是 Reference（numpy）/ fast（CPU）/ CUDA kernel 三份实现的**共同算法骨架**
（`PLAN_LocalCWLDForce_OpenMM_Plugin.md` §18）：
① density accumulation → ② Q 与 $dQ/d\,\mathrm{dens}$ → ③ pair energy/direct force 与
伴随量 $\lambda$ → ④ density backprop chain force。用 `CustomGBForce` 时 ②④ 的导数由
OpenMM 自动微分给出；手写实现必须自己写解析导数。

### 3.2 ① 局部密度

$$
\mathrm{dens}_i \;=\; S_i \sum_{j} B_j \, K(r_{ij}) \,
\mathbb{I}\bigl[\mathrm{res}(i)\neq \mathrm{res}(j)\bigr]\,
\mathbb{I}\bigl[(i,j)\ \text{未被排除}\bigr]
$$

$$
S_i=\mathrm{dens\_sink}_i,\qquad
B_j=\mathrm{dens\_source}_j\cdot w^{\mathrm{class}}_j\cdot \mathrm{cmod}_j,\qquad
K(r)=\Bigl(1-\tfrac{r^2}{r_{\rm env}^2}\Bigr)^{2}\ \ (r<r_{\rm env})
$$

**$K$ 的选择理由**（Lucy 型）：
$K(r_{\rm env})=K'(r_{\rm env})=0$，所以密度**和链式力**在支撑边界上都连续
（链式力用 $K'$，只有 $K$ 连续是不够的——这是不用 $(1-x^2)$ 一次幂的原因）；
$K$ 是 $r^2$ 的多项式，**不需要开方**，GPU 上省一个 `sqrt`；紧支撑让邻居表可以只建到
$r_{\rm env}=0.35$ nm，而不是 $r_c=1.2$ nm（体积差 40 倍，这是插件性能的前提）。

$$
K'(r)=-\frac{4r}{r_{\rm env}^2}\Bigl(1-\frac{r^2}{r_{\rm env}^2}\Bigr)
$$

**四条容易踩的语义**：

1. **方向性是硬约束**：$j$ 的 source 参数贡献给 $i$ 的 sink。不得对称化成 $B_iB_j$，
   不得用 target 的 $w^{\rm class}_i$。「并行方便」不是理由。
   物理上这是「$i$ 感受到的环境」，本来就不对称：水感受金属 ≠ 金属感受水。
2. **排除判据是 residue id，不是 molecule id**。代码里变量叫 `mol_id`，实际赋的是
   `residue.index`。v2.6 的精确语义是「**同 residue 不计入密度**」。
   对水/离子两者等价；对蛋白**完全不等价**（相邻残基互相贡献密度）。
   改成真 molecule id 属于**独立的物理模型变更**，不能混进性能移植。
3. **`FAST_ACTIVE_DENSITY=True` 是物理近似，不是纯优化**。它让溶质里
   **只有 O/N/S 和带电侧链位点成为密度源**，碳和氢 `dens_source=0`。
   名字叫 "fast"，但它改变的是模型本身（碳骨架不再产生环境密度）。
4. **排除对（bonded exclusions）不计入密度**，与静电排除表共用一张表。

**（本次汇总新增）语义不一致**：`enable_solute_polarization=False` 时，
`build_phase_cwld_metadata` 的整个溶质分支被跳过，于是蛋白原子**根本不是密度源**
（`dens_source=0`）；而 `setup_cwld_lips_system` 在同一配置下打印的是
「溶质仍是 density source，但不响应」。**代码与日志矛盾。**
影响面：消融矩阵里 `Water_only` 这一臂（溶质极化关、水响应开）的语义实际是
「水只对水和离子的密度响应」，而不是「水对全部环境响应、溶质不响应」。
✅ **不影响 §7.4 的判别实验**：那里用的 `LIPS_fixedQ_baseline` 两种响应全关，
密度通道整体不参与，结论不受影响。
⚠ 未验证 1CKK 时代基于 `Water_only` 的任何结论是否受影响。

### 3.3 charge_mod：相对本体溶剂的电荷强度

$$
\mathrm{cmod}_j=\mathrm{clip}\Bigl(1+a_{q^2}\bigl(q_{{\rm base},j}^2-q_{\rm ref}^2\bigr),\;0.25,\;2.5\Bigr)
$$

$q_{\rm ref}^2$ **动态锚定**：取当前体系中「真正作为密度源的水」的 $q^2$ 均值。
对 TIP3P 就是水 O 的 $q^2$，**不是 O+H+H 的平均**——否则普通水被人为上调，
整个密度标度跟着漂。

意义：$\mathrm{cmod}$ 让「一个 $\pm 1$ 的离子」比「一个水 O」在密度里更重，且这个比较是
**无量纲、相对本体溶剂**的。硬 clip 到 $[0.25, 2.5]$ 防止高电荷离子（$q^2=4$）把密度
拉爆——这是唯一一处用硬 clip 的地方，因为它作用在**逐原子常数**上，不进力的求导路径。

### 3.4 ② 电荷响应：为什么是两层 tanh

$$
t_i=\tanh\!\Bigl(k_{\rm polar}\frac{\mathrm{dens}_i}{\rho_0}\Bigr),\qquad
y_i=\frac{A_i t_i}{q_{\rm clamp}},\qquad
\boxed{\,Q_i=q_{{\rm base},i}+q_{\rm clamp}\tanh y_i\,}
$$

$$
A_i=\mathrm{static\_phase}_i\cdot \mathrm{is\_polar}_i\cdot \mathrm{dpolar}_i
$$

**内层**把无上界的密度压成占据度 $t\in[0,1)$。$\rho_0$ 是「一个正常溶剂化壳」的密度标度，
$k_{\rm polar}/\rho_0 = 0.8/13.5 = 0.0593$ 是线性区斜率。

**外层**做限幅，且**保持一阶恒等**：小信号下
$q_{\rm clamp}\tanh(A t/q_{\rm clamp}) = A t - \tfrac{(A t)^3}{3q_{\rm clamp}^2}+O(A^5)$，
所以在 $|At|\ll q_{\rm clamp}$ 时**不引入额外增益**，$\Delta q\approx A_it_i$；
而 $|At|\to\infty$ 时 $\Delta q\to \pm q_{\rm clamp}$。全程 $C^\infty$。
**硬 clip 会在力里造出不连续**（$dQ/d\,\mathrm{dens}$ 阶跃到 0），所以规格里明写
「不得用硬 clip 替换嵌套 tanh」。

**3.4.1 饱和上界与「clamp 对溶质从未生效」**

$\tanh y \le 1$ 且 $t\le 1$ ⇒ $|\Delta q|\le \min(|A_i|,\;q_{\rm clamp})$。
对按元素给 dpolar 的溶质，$|A|=0.012$，而 $q_{\rm clamp}=0.20$
⇒ **限幅器从来没有起过作用，真正的天花板是 dpolar 本身（clamp 的 6%）**。
这正是 §7.1 那个逐原子覆盖修法的动机之一。

**3.4.2 导数**

$$
\frac{dQ_i}{d\,\mathrm{dens}_i}=\underbrace{A_i\frac{k_{\rm polar}}{\rho_0}}_{\text{线性区斜率}}
\bigl(1-\tanh^2 y_i\bigr)\bigl(1-t_i^2\bigr)
$$

两个 $(1-\cdot^2)$ 因子分别是外层和内层的饱和衰减。
两个极限**都必须自然为零、不需要分支**：$S_i=0$ ⇒ $\mathrm{dens}_i=0$ ⇒ $t=0$ ⇒
$dQ/d\mathrm{dens}=A_ik/\rho_0$ 但 $\lambda$ 侧的密度对没有 ⇒ 无贡献；
$A_i=0$ ⇒ 导数恒为 0 ⇒ $Q_i=q_{\rm base}$。

**这条链在低密度端已被正面验证**：HIE149:ND1 的 dpolar 确实被覆盖成 −0.15，
但它距最近 Zn 13.13 Å ⇒ dens≈0 ⇒ $t\approx0$ ⇒ $\Delta q\approx0$ ⇒
实测力扰动 0.033 kJ/mol/nm，**低于 fp32 噪声地板 0.122**，比最弱的真配位原子小 3257 倍。
（验的是生产 CustomGBForce，不是 §6 的分析侧镜像。）

**3.4.3 水分子的电中性：设计意图与实际行为**

水 H 拿 $\mathrm{dpolar}_H=-\tfrac12\mathrm{dpolar}_O$，意图显然是让
$\Delta q_O + 2\Delta q_H = 0$（分子电中性）。**实际不严格成立**，两个独立原因：

* **一阶就破**：生产 `CustomGBForce` 里 O 和两个 H **各自按自己的位置算 dens**，
  三者的 $t$ 不同，所以一阶项不抵消。
  （分析侧的 fast 路径有 `q_driver`，让三个原子共享 O 的 dens——**只有那条路径**
  才有精确的一阶抵消。这是实现 #1 与 #3 的一处真实语义分叉。）
* **三阶还会再破一次（本次汇总新增的推导）**：即使**假设**三个原子共享同一个 $t$，记
  $a=\mathrm{dpolar}_O t/q_{\rm clamp}<0$，

  $$\frac{\Delta q_O+2\Delta q_H}{q_{\rm clamp}}=\tanh a-2\tanh\tfrac a2=-\frac{a^3}{4}+O(a^5)>0$$

  即**外层 tanh 的曲率**会让净电荷偏**正**。

  **实测量级如下**（`2.4/1ckk_v26_long_convergence_5ns_per_seed_metrics_QFIX.csv`，
  该 CSV 的 `water_net_delta_Q_*` 列，3 seed 互相吻合到 1%）：

  | 量 | 实测 |
  |---|---|
  | `water_net_delta_Q_mean` | **+8.0e-4 e**（符号为正 ✓） |
  | `water_net_delta_Q_abs_mean` | 9.1e-4 e（≈ mean ⇒ **系统性同号**，不是噪声） |
  | `water_net_delta_Q_max` | 6.1e-3 … 1.1e-2 e |
  | `near_Ca_water_net_delta_Q_mean` | −1.3e-3（`ca_w2`）… −6.6e-3（`aq0p5_ca_w2`） |

  ⚠ **归因未定，不要把这组数说成「验证了三阶机制」**（2026-09-07 收紧）。
  三阶项在 $t\sim0.3$ 时给 ~1e-3 e，符号和量级确实都对得上，
  **但生产引擎里一阶中性本来就先破了**（上一条），而一阶残差

  $$\Delta Q_{\rm linear}=d_Ot_O+d_H t_{H_1}+d_H t_{H_2}$$

  完全可能贡献同量级、甚至主导。**近 Ca 的水反号恰恰提示一阶 dens mismatch 很重要**
  ——如果是三阶曲率主导，符号应当不随环境翻转。

  **干净的判别只需一次现有轨迹分析**（不跑 MD）：逐水分解成

  $$\Delta Q_{\rm linear}\quad\text{与}\quad
  \Delta Q_{\rm nonlinear}=\Delta Q_{\rm exact}-\Delta Q_{\rm linear}$$

  就能知道 $+8\times10^{-4}$ e 到底出自哪一项。列为待办 §9.5-1。

`compute_water_net_delta_q()` 的 docstring 已明确写着这是
"a genuine measurement of per-molecule charge non-conservation in the exact engine,
not a bug in this diagnostic"。⇒ **模型的已知性质，不是 bug。**

⚠ **（本次汇总新增的待办）**：每水 +8e-4 e，乘上 10⁴ 量级的水分子，
体系总净电荷可能到 **e 量级**。这个乘法**本报告没有实测核对过**
（现有诊断是 500 三联体/帧的抽样，不能用来推总量），
也没在任何既有报告里结论化。列为待办 §9.5-1。
✅ 但它**不是** §7.4 那个长波长屏蔽不足的原因：判别实验里固定电荷臂
（$\Delta q\equiv0$ ⇒ 净电荷严格守恒）与 CWLD 臂在最低模式上的 $S_{ZZ}$
只差 11%（1.29e-3 vs 1.15e-3），**两者都比 PME 大 22–25 倍**。

### 3.5 ③ Pair closure：窗函数正规形式

生产用 Sakuraba 零多极（ZMM，$\alpha=0$）closure。**关键的结构性认识**
（`PLAN_PSWF_Closure_Family.md` §1）：它字面上已经是窗函数形式。令 $x=r/r_c$，

$$
U(r)=\frac{1-A(x)}{r},\qquad A(x)=\int_0^x\chi(t)\,dt
$$

于是三条性质**直接从 $\chi$ 读出**，不必逐项验算多项式系数：

| 性质 | 等价条件 |
|---|---|
| $U(r_c)=0$（能量在 cutoff 归零） | $A(1)=1\iff\int_0^1\chi=1$ |
| $U'(r_c)=-\chi(1)/r_c^2$，力连续 | $\chi(1)=0$ |
| $U^{(m)}(r_c)=0,\ m=0..\ell$ | $\chi$ 在 $x=1$ 处有 $\ell$ 阶零点 |

**⚠ 一处早前写错的推理（2026-09-07 修）**：光靠上面这组端点条件 + 归一化，
**推不出** $(1-x^2)^\ell$ 是最低次多项式——$(1-x)^\ell$ 次数只有 $\ell$，
同样满足 $\chi(1)=\dots=\chi^{(\ell-1)}(1)=0$ 和归一化，次数还低一半。
（`PLAN_PSWF_Closure_Family.md` 里「可行空间 $\chi=(1-x^2)^\ell P(x)$」这句话
同样比端点条件本身强，需要同一个补充假设。）

**还差的那个条件是 $\chi$ 为偶**（只含 $x$ 的偶次幂）。它有三个独立理由：

* **① 数值结构**：$U$ 因此只含 $r$ 的偶次幂，$U=1/r+\sum_m b_m r^{2m}/r_c^{2m+1}$，
  **GPU 上不需要开方**（`closure.py::poly_coeffs` 在 $A(x)$ 出现偶次项时直接抛异常，
  就是在守这条）。
* **② Fourier 结构**：偶性消掉分部积分在**原点**的低阶边界项（$\chi'(0)=\chi'''(0)=0$），
  **配合 $x=1$ 处的 $\ell$ 阶零点**（那边的边界项也逐阶消掉），使 ZMM-$\ell$ 达到
  $\hat c_\ell(q)=O(q^{-(\ell+1)})$ 的衰减。
  ⚠ **不要写成「偶性 ⇒ $q^{-3}$」**（2026-09-07 修）：$q^{-3}$ 是 $\ell=2$ 这一档的值，
  由闭式立刻可读出 $\ell=1,2,3$ 分别是 $q^{-2},q^{-3},q^{-4}$（§3.6）。
  衰减阶主要由**端点零点阶 $\ell$** 定，偶性只负责不让原点那一侧先掉一阶
  （`closure.py` 里 `pswfz2` 取「$x^2$ 的 Legendre」而非移位 Legendre 的注释说的是
  $\ell=2$ 的具体情形：奇次基会在原点留折点，把 $q^{-3}$ 退化成 $q^{-2}$——
  那句话对该档成立，但不是偶性的普遍结论）。
* **③ 物理结构**：ZMM 的零多极构造本身给出的就是 $x^2$ 的多项式。

**补上偶性之后结论就是干净的**：偶多项式在 $x=1$ 有 $\ell$ 阶零点 ⇒ 由偶性它在 $x=-1$
也有 $\ell$ 阶零点 ⇒ 可被 $(1-x^2)^\ell$ 整除 ⇒ 最低次数恰为 $2\ell$，
且商必为常数 ⇒ $\chi=c(1-x^2)^\ell$，$c$ 由归一化定死。
**ZMM 公式本身没错，错的是早前给它的「由端点条件唯一推出」这个解释。**

⇒ 在**偶多项式**这个类里，ZMM 是满足这组条件的**最低次**成员：

$$
\chi_\ell(x)=c_\ell(1-x^2)^\ell,\qquad
c_\ell=\frac{1}{\int_0^1(1-t^2)^\ell dt}=\frac{(2\ell+1)!!}{(2\ell)!!}
$$

**而 $c_\ell$ 恰好就是代码里那个常数项的分子**（1.5、15/8、35/16）——
三档逐项验算全部对上，无一例外。这不是巧合：常数项就是 $-c_\ell/r_c$。

| $\ell$ | $\chi_\ell(x)$ | $U_\ell(r)$ | 物理别名 |
|---|---|---|---|
| 1 | $\tfrac32(1-x^2)$ | $\tfrac1r-\tfrac{3}{2r_c}+\tfrac{r^2}{2r_c^3}$ | ZDipole / RF-like |
| **2（默认）** | $\tfrac{15}{8}(1-x^2)^2$ | $\tfrac1r-\tfrac{15}{8r_c}+\tfrac{5r^2}{4r_c^3}-\tfrac{3r^4}{8r_c^5}$ | ZQuadrupole |
| 3 | $\tfrac{35}{16}(1-x^2)^3$ | $\tfrac1r-\tfrac{35}{16r_c}+\tfrac{35r^2}{16r_c^3}-\tfrac{21r^4}{16r_c^5}+\tfrac{5r^6}{16r_c^7}$ | ZOctupole |

$$
\frac{dU}{dr}=-\frac{1-A(x)}{r^2}-\frac{\chi(x)}{r_c\,r}
$$

> **副产品**：CWLD 的密度核 $K(r)=(1-(r/r_{\rm env})^2)^2$ 就是 $\chi_2$ 去掉归一化。
> 同一个窗在代码里出现了两次，但**这两处不要合并**——物理约束不同：
> 密度核只需紧支撑，closure 还要背静电求和规则。

**窗层的六项自检**（`engine/closure.py::self_check`，新窗必须全过）：
① 归一化 $\int_0^1\chi=1$（否则 $U(r_c)\neq0$）；② 端点条件 $\chi(1)=0$（否则力在 cutoff
跳变，NVE 会漂）；③ 矩 $\int\chi x^{2m}$（备查不判定）；④ `dUdr` vs 中心差分——
判据必须是**混合绝对/相对**的，因为 $U'(r_c)\to0$ 时纯相对误差没有意义；
⑤ $r\to0$ 前导奇异性：$rU-1 = -A(x)$ 本来就是 $O(x)$，所以判据必须**随 $x$ 缩放**
（$(rU-1)/x\to-\chi(0)$），否则是假失败；⑥ 直接验 $U(r_c)=0$。

**冻结纪律**：`ZMMWindow` 构造出来后**立刻**与 v2.6 的冻结系数逐位比对，不一致直接抛异常；
`lepton()` 返回的是**冻结字符串本身**，不是由系数重新生成的（重新生成的串只用于等价性测试）。
默认路径**根本不导入窗层**，改动前的备份逐字节相同（已验）。

### 3.6 Fourier 侧：截断误差的正确写法

把 $1/r$ 拆成保留项与丢弃项：$1/r = L + S$，$L=U(r)=(1-A(x))/r$。对 $\hat L$ 分部积分：

$$
\boxed{\;\hat S(k)=\frac{4\pi}{k^2}\,\hat c(q),\qquad
\hat c(q)=\int_0^1\chi(x)\cos(qx)\,dx,\qquad q=k r_c\;}
$$

即 **Fourier 空间的 split 就是 $4\pi/k^2$ 乘窗的余弦变换**。三条独立验证全过：
中心恒等式在 18 个 (窗,k) 点上相对偏差 **9.0e-14**；$\hat c(0)=1$ 到 3e-15；
$E_F$ 两条独立路线（k 空间积分 vs 实空间闭式）相对偏差 2.2e-6。

> 判据陷阱记录：恒等式验证里有 2 个点相对偏差 1.8e-7，**不是**恒等式不成立——
> $\hat S_{\rm direct}$ 在大 k 要用 $4\pi/k^2$ 减 $\hat L$，**相消 6 个量级**；
> 按相消前量级衡量是 9e-14。

**闭式（本次汇总新增，取代早前的两阶展开）**：ZMM 窗的 $\hat c$ 有精确解。
由 Poisson 的球 Bessel 积分表示 $j_n(z)=\dfrac{z^n}{2^{n+1}n!}\displaystyle\int_{-1}^{1}(1-t^2)^n\cos(zt)\,dt$ 得

$$
\boxed{\;\hat c_\ell(q)=(2\ell+1)!!\;\frac{j_\ell(q)}{q^{\ell}}
=\sum_{m\ge0}\frac{(-1)^m (q^2/2)^m}{m!\prod_{n=1}^{m}(2\ell+1+2n)}\;}
$$

（$q\to0$ 给 $\hat c=1$，与 $\int\chi=1$ 自洽；一阶项 $-q^2/(2(2\ell+3))$ 即
$-q^2m_2/2$，$m_2=1/(2\ell+3)$ = 1/5, 1/7, 1/9。）

**大 $q$ 渐近 —— 注意这里有两个不同 scope 的陈述**（2026-09-07 按 peer 意见分开）：

**(i) 一般结论（不依赖上面那个闭式）**：对**任何**偶的、在 $x=1$ 有 $\ell$ 阶零点的 $\chi$，
反复分部积分时两端的边界项逐阶消掉（原点侧靠偶性、$x=1$ 侧靠零点阶），
第一个存活项 $\propto\chi^{(\ell)}(1)/q^{\ell+1}$，于是

$$
\hat c(q)=O\bigl(q^{-(\ell+1)}\bigr)\quad\Rightarrow\quad
\ell=1:\;q^{-2},\qquad \ell=2:\;q^{-3},\qquad \ell=3:\;q^{-4}
$$

即**谱衰减指数只由端点零点阶 $\ell$ 定**，与 $\chi$ 在内部长什么样无关
⇒ **同 $\ell$ 内的任何优化（换 $c$、解约束最优）只能动前因子，动不了指数。**
这是「高 $\ell$ 更光滑」在 Fourier 侧的准确说法，见 §3.5 ② 的限定。

**(ii) ZMM 专属**：上面那个闭式 $\hat c_\ell=(2\ell+1)!!\,j_\ell(q)/q^\ell$
**只对 ZMM 的 $\chi_\ell=c_\ell(1-x^2)^\ell$ 成立**，
它给出 (i) 的一个特例（$j_\ell(q)\sim\sin(q-\ell\pi/2)/q$），
**不要让 (i) 继承 (ii) 的适用范围**——`pswfz2` 与 PCF 都不在这个闭式里。

**(i) 已被实测确认**（peer session `2-5-9e`，`REPORT_HYBRID_LOWK_B0_2026-09-07.md` §5）：
包络斜率实测 **−2.000 / −2.952 / −4.000**（zmm1 / ℓ=2 组 / zmm3），
且对 PCF 在 $c\in[10^{-6},2\pi]$ 全程稳定 ⇒ 指数确实只跟 $\ell$ 走。

⚠ **算它必须用级数，不要用 $j_\ell$ 的三角闭式**：后者在 $q\lesssim2$ 是两个
$O(1)$ 数相减得到 $O(10^{-2})$，相消两三个量级——本报告第一版就是在这里把
$\hat c_2(1.1088)$ 算成了 0.868 / 0.920（真值 0.91512），随 sin/cos 的精度乱跳。
这与 §3.6 开头那条「相消 6 个量级的地方相对误差没有意义」是同一个陷阱。

**这条规则比原先写的更宽**（2026-09-07，peer 从另一个方向撞到）：
$I_n=\int_0^1x^n\cos(qx)dx$ 的**闭式递推**（大 $q$ 稳定，因为递推在除 $q$）
在小 $q$ 同样炸——从 $q=10^{-6}$ 积起，zmm3 的 $E_F$ 被积函数出到 **1e26**、
pswfz2 出到 **1e110**。⇒ **凡是「闭式/递推」形式，在小 $q$ 都要当不可用处理**；
正确分工是**自适应求积管全程，递推只当大 $q$ 的 oracle**
（两者在 $q\in[200,400]$ 上吻合 5 位，互为验证）。

均方力误差里 $k^4$ 与 $\hat S^2$ 的 $k^{-4}$ 精确抵消：

$$
E_F[\chi]\;\propto\;\int_0^\infty \hat c(k r_c)^2\,S_{ZZ}(k)\,dk
$$

**这正是 PSWF（长椭球波函数）的能量集中问题**，只是权重从锐截断换成物理的
电荷结构因子 $S_{ZZ}$。取 Debye 型 $S_{ZZ}=k^2/(k^2+\kappa^2)$、$s\equiv\kappa r_c$ 有闭式：

$$
\tilde E_F[\chi;s]=\int_0^1\chi^2-\frac s2\iint\chi(x)\chi(y)\bigl[e^{-s|x-y|}+e^{-s(x+y)}\bigr]dxdy
$$

即「恒等算子减平滑核」，与 PSWF 同构。**⚠ 但 $E_F$ 作为判据已被静态力审计推翻**，见 §7.3。

### 3.7 ③' Pair 能量与 reaction-field 抵消

$$
E_{ij}=C\Bigl[\,Q_iQ_j\,U_\ell(r_{ij})\;-\;q_{{\rm base},i}q_{{\rm base},j}\bigl(\tfrac1{r_{ij}}-\tfrac1{r_c}\bigr)\Bigr],\qquad C=138.935458
$$

$$
\frac{dE_{ij}}{dr}=C\Bigl[\,Q_iQ_j\,\frac{dU_\ell}{dr}+\frac{q_{{\rm base},i}q_{{\rm base},j}}{r^2}\Bigr]
$$

**第二项是干什么的**：体系里**原生 `NonbondedForce` 仍在跑**（它带着 LJ、排除表、
exception，这些不能重写），静电被设成 `CutoffPeriodic` + $\varepsilon_{\rm RF}=1$，
其静电部分恰是 $q_iq_j(1/r-1/r_c)$。CWLD 力里减掉它 ⇒
**净静电 = $Q_iQ_j U_\ell(r)$**，既不重复计算也不漏算基线。

$\varepsilon_{\rm RF}=1$ 的选择让这一项**与 closure 阶数无关**——换 $\ell$ 不必改抵消项，
它永远是 $1/r-1/r_c$。这是「$\ell$ 可以自由扫」的前提。

`r_on` switching 只属于原生 `NonbondedForce`（作用于 LJ），
**CWLD pair 本身不套 switching**。

### 3.8 ④ 链式响应力

**只让能量对、忽略 $dQ/dR$ 的实现不合格**——即使短轨迹表面稳定也不能进生产。
先算每个粒子的伴随量（adjoint）：

$$
\lambda_i=\frac{\partial E}{\partial Q_i}=C\sum_j Q_j\,U_\ell(r_{ij})
\;\;\bigl[+\,k_{\rm penalty}(Q_i-q_{{\rm base},i})\ \text{若启用}\bigr]
$$

$$
g_i=\lambda_i\frac{dQ_i}{d\,\mathrm{dens}_i}
$$

每个密度邻居对的附加径向贡献：

$$
\boxed{\;\mathrm{chain}\frac{dE}{dr}(i,j)=\bigl[g_iS_iB_j+g_jS_jB_i\bigr]K'(r_{ij})\;}
$$

**非对称是本质的**：$g_iS_iB_j$ 是「$j$ 作为源改变了 $i$ 的电荷」这条通路，
$g_jS_jB_i$ 是反向通路，两者的参数完全不同，**不能合并成一个对称因子**。
仅当 residue 不同、pair 未排除、$r<r_{\rm env}$ 时存在。

注意 $\lambda_i$ 的求和跑遍 $r_{ij}<r_c$ 的**所有**对（1.2 nm），而链式力只在
$r<r_{\rm env}$（0.35 nm）的对上落地。**两个不同的邻居半径**，插件因此维护两张表：
共享的 $r_c$ 表（OpenMM 的）+ 自建的 env pair 表。

用 `CustomGBForce` 时这一整段由 OpenMM 自动微分给出；
手写实现（reference / fast / CUDA kernel）**必须手动同步解析导数**——
这是 §6 那张同步表存在的根本原因。

### 3.9 ⑤ Q penalty（默认关闭）

$$
E^{\rm penalty}_i=\tfrac12 k_{\rm penalty}(Q_i-q_{{\rm base},i})^2
$$

它不仅加能量，还**通过 $\lambda_i$ 进入链式力**。只加能量、漏掉响应力是错误实现。
当前 `ENABLE_Q_PENALTY=False`：外层 tanh 的软 clamp 已经在做限幅，
penalty 属于冗余的第二道，留着是为了将来需要「弹性回复到 qbase」这种语义时能开。

### 3.10 判据数学：$E_F$ 为什么在细粒度上不可靠

$E_F$ 的 Poisson 求和推导假设**全对求和**，但真实截断误差只涉及**非排除对**
（1CKK 体系有 43338 个排除对）。而实测 $S_{ZZ}$ 在 $k\approx31$ nm⁻¹ 的峰
**主要来自水分子内 O–H 关联**——这部分进了 $S_{ZZ}$ 却不参与求和。扣掉分子内项后
zmm1 的排序符号翻正，与直接审计一致：

| kernel | 全对 $S_{ZZ}$（原做法） | 扣分子内 | 审计实测 |
|---|---|---|---|
| zmm1 | 1.015 ❌ | 0.851 | **0.980** |
| pswfz2 | 1.079 | 1.077 | 1.137 |
| zmm3 | 1.197 | 1.218 | 1.263 |

但**不能说「扣掉就修好了」**：$S_{\rm inter}$ 在多数 k 上为负（反关联），上表只取正部，
是 hack；且修正后 0.851 vs 实测 0.980 仍差 15%。
**根因是 RMS 力误差需要四点关联，不是 $S_{ZZ}$ 这个二点量能定的**——$E_F$ 全程是平均场近似。

⇒ **$E_F$ 在 >10% 的差别上大致可用（pswfz2、zmm3 两种算法都判对了），
在 2–5% 量级上不可靠且可能翻符号。直接静态力审计取代它作为判据。**

---

## 4. PSWF 能不能解决 ZMM 的长波长问题

**核心结论（2026-09-07 收紧，替代早前一版更强的表述）**：

> **任何紧支撑、有限 cutoff、可积的 pair kernel，其 Fourier 变换在 $k=0$ 处有限且解析；
> 而 Coulomb 是 $4\pi/k^2$。因此换窗（PSWF 也算）不能恢复 Coulomb 的严格长波长极点，
> 也不能恢复由长程 Coulomb 强制出来的 Stillinger–Lovett 渐近系数。
> 它仍然可以改变有限 $k$ 的响应；但当前实现的 `pswfz2` 既优化错了频段，
> 直接力审计也没有测到收益。**

这条比早前那条「$\hat c(0)=1$ + 窗只能动 6% + 全类天花板 1.39×」的证明链**窄，但硬得多**，
而且不依赖后来已被自己推翻的 $E_F$ 判据。下面逐条说明**改窄的四个理由**，
因为那四处是真实的逻辑越界，不是措辞问题。

### 4.1 稳的那部分：kernel 层的极点

$U(r)$ 在 $r>r_c$ 恒为零、在 $r\to0$ 只有 $1/r$ 的可积奇异性 ⇒ $\hat U(k)$ 在 $k=0$
有限解析。Coulomb 的 $4\pi/k^2$ 有二阶极点 ⇒ **被丢掉的部分 $\hat S=4\pi/k^2-\hat U$
必然保留这个极点**，与窗的形状无关。

写成窗的语言就是 $\hat S(k)=\dfrac{4\pi}{k^2}\hat c(kr_c)$ 且 $\hat c(0)=\int_0^1\chi=1$
（而这个归一化**恰恰就是 $U(r_c)=0$ 这条约束本身**，见 §3.5）。

⚠ **但要注意极点的存活比 $\hat c(0)=1$ 更一般**：即使放弃 $U(r_c)=0$（允许能量在 cutoff
跳变），$\hat U(0)$ 仍然有限，极点照样留着。所以正确的依据是**紧支撑**，
不是归一化条件——早前那版把结论挂在 $\hat c(0)=1$ 上，反而把它讲窄了。

### 4.2 越界之一：kernel 极点 ⇏ 平衡态 $S_{ZZ}$ 比值发散

早前那张表最后一格写「实测 CWLD/PME → 发散」，**这是越界的**。
$\hat S$ 是**被丢掉的相互作用核**；$S_{ZZ}$ 是**跑完一个不同 Hamiltonian 之后的平衡态
结构因子**。前者发散不能推出后者的比值发散。

而且早前文档内部就不自洽：§7.4 一边用 $S_{ZZ}\simeq k^2/(k^2+\kappa^2)$ 拟出
$\kappa_{\rm PME}\approx126$ / $\kappa_{\rm cut}\approx30$，若两者**都**服从该形式，则

$$
\frac{S^{\rm cut}_{ZZ}}{S^{\rm PME}_{ZZ}}\xrightarrow{k\to0}
\Bigl(\frac{\kappa_{\rm PME}}{\kappa_{\rm cut}}\Bigr)^2\approx 17.6
$$

是**有限常数**；一边又报 $k=0.924$ 实测 **44×**。两个数从来没对上过。

**（本次汇总查明，见 §7.4.1）真因是：截断轨迹根本不服从那个形式**，
所以「$\kappa$ 差 4.2 倍 ⇒ 低 k 差 17 倍」这条推断本身无效，
而 44× 是最低模式上的**直接测量**。两者不需要一致——其中一个建立在不成立的形式上。

**因此正确的表述**：有限 cutoff kernel 无法恢复 Coulomb 的 $k\to0$ 极点与 SL 渐近系数
（$S_{ZZ}\propto k^2$ 本来就是 Coulomb perfect screening 的长波长约束）；
**但实际截断体系的 $S_{ZZ}(k\to0)$ 究竟是趋常数、仍然 $\propto k^2$、
还是在有限盒下压根没进入渐近区，必须由数据单独判定**——而现有数据判不了（§7.4.1）。

### 4.3 越界之二：「6% vs 44×」是直觉图，不是 bound

$\hat c$ 的窗间落差是**kernel 层**的差；44× 是**平衡态**的差。
响应可以非线性放大，两者不能做量纲式比较。而且「6%」也**不是全部允许窗的自由度上限**，
只是 zmm1/2/3 三兄弟在 $q\approx1.11$ 附近**恰好挨得很近**。

顺带把数值修正掉——早前那两行是两阶展开，误差和误导都不小。
用 §3.6 的闭式 $\hat c_\ell(q)=(2\ell+1)!!\,j_\ell(q)/q^\ell$ 精确算：

| $k$ (nm⁻¹) | $q=kr_c$ | $\hat c$(zmm1) | $\hat c$(zmm2) | $\hat c$(zmm3) | 窗间落差 |
|---|---|---|---|---|---|
| 0.924 | 1.1088 | 0.8823 | 0.9151 | 0.9336 | 5.8% |
| 2.17 | 2.604 | **0.4670** | **0.5981** | **0.6766** | **45%** |
| → 0 | → 0 | 1 | 1 | 1 | 0 |

（早前写的 $q=2.604$ 三个值 0.32/0.52/0.62 是错的，即使标了「只看量级」也会误导，已换成闭式值。）

**这张表其实同时给出两个读数**，两个都要说：

1. $k\to0$ 端**自由度严格为零**（这是 §4.1 的定理，不是经验）。
2. 但到 $q\approx2.6$ 窗间落差已经有 **45%** ⇒ **在有限 $k$ 上窗是有真自由度的**。
   所以「换窗对有限 $k$ 也没用」是**没有根据的**，早前那版把它一起否掉了。

正确的定位：**ZMM 1–3 在 $q\lesssim1.1$ 的实际可调幅度只有约 6%，远小于观察到的结构因子
差异，作为经验佐证**——不是解析封死的一部分。

### 4.4 越界之三：$c=2\pi$ 是这个实现的选择，不是 PSWF 的性质

PSWF 的经典定义是：**给定一个人为选定的 bandwidth**，求能量最集中的函数
（Slepian–Pollak 那一支的原始设定）。$2\pi$ 不是 PSWF 的宇宙常数——
它在本项目里之所以是 $2\pi$，是因为实现时按「$q=kr_c$，$k=2\pi/r_c\Rightarrow q=2\pi$」
把带宽绑到了 cutoff 上（`closure.py` 的注释写的是「由 cutoff 定死，不是自由参数」，
那句话对**这个实现**成立，对 PSWF 这个工具不成立）。

**准确说法**：

> 目前实现的 `pswfz2` 采用 $c=2\pi$ 的带外集中目标，因此主要优化 $q>2\pi$；
> 而实际异常出现在 $q<2\pi$。故**这一具体 PSWF objective 与目标错位**。

这样才不会和 §3.6 自己的结论打架：那里已经正确指出，加了 $S_{ZZ}(k)$ 权重之后，
$E_F$ **仍然是一个 PSWF-like 的能量集中问题**。
⇒ **「PSWF 这个数学工具没用」不成立；「当前的 pswfz2 窗没解决这个问题」成立。**
即使重做一个 low-$k$ 加权的 PSWF，它**仍然补不回 Coulomb 极点**（§4.1），
但**可能改善有限 $k$**（§4.3 的 45% 就是可动空间）。

### 4.5 越界之四：「全类天花板 1.39×」必须降级

这是早前那版最明显的内部矛盾：旧 §4.4 把 0.519 / 1.39× 叫作「任何窗都逃不掉」的
**全类物理天花板**，而 §3.10 已经证明 **$E_F$ 不是可靠的真实 RMS 力泛函**——
它把分子内已排除的 O–H 关联塞进 $S_{ZZ}$，而真实 RMS 力误差需要四点关联，
连 zmm1 的方向都会判错。**用一个被推翻的 surrogate 去宣布全类天花板，是循环的。**

**只能这么叫**：

> 0.519 / 1.39× 是**在当前 $E_F$ surrogate、给定解析 $S_{ZZ}$ 模型、给定约束类
> （$\chi$ 偶、$x=1$ 处 $\ell$ 阶零点、归一化）下的数学下界**。

**不能推出**：所有 finite-cutoff pair kernel 的**真实力精度**最多只能改善 1.39×。

那张表里仍然站得住的是**定性结构**（来自不确定性原理：$\chi$ 紧支撑于 $[0,1]$ 且
$\hat c(0)=1$ ⇒ $\hat c$ 的宽度 $\gtrsim1$ ⇒ $\int\hat c^2S_{ZZ}$ 有下界；
以及「端点连续性几乎不要钱」这个 3% 的观察）。
**真正可信的证据是 §7.3 的静态力审计**：固定坐标、固定 qbase、只换 pair kernel，
`pswfz2` 所有通道确实差于 zmm2，而 zmm1 反而更好。**这个证据比 $E_F$ 强得多**，
早前那版反而把它写得比 $E_F$ 的天花板还轻。

保留的推论仍然有效（它不依赖 $E_F$）：**任何在 $r_c$ 处归零的 pair kernel 都等价于
某个窗**（令 $A(x)=1-rU(r)$、$\chi=A'$），所以设想中的
$L_{\rm PSWF}(r)+C_{\rm closure}(r)$ 分解**没有额外自由度**——$C_{\rm closure}$ 就是换窗。

### 4.6 pswfz2 实测怎么样

已跑完：**在力精度上每个通道都比 zmm2 差 3–15%**（§7.3 的表），
13.7 倍的带外能量削减**没有换来任何力精度**。
唯一未测的通道是 NVE 能量漂移（脚本已就绪待跑，§9.3）。

### 4.7 所以要修长波长，得往哪走

前提：**必须先量清楚这个效应对目标可观测量到底有多大影响**（§9.5），
否则容易为一个不影响结论的伪影付高价。而且第一步应该是 §7.4.1 那件事——
**先把「截断体系的 $S_{ZZ}$ 在低 k 到底什么形状」测清楚**，现在连这个都没定。

| 方向 | 说明 | 代价 |
|---|---|---|
| 有限尺寸/背景电荷修正 | 事后修正自由能，不改动力学 | 最低，但只修可观测量 |
| ABFE 的 decharging 段改用全 Ewald | 只在最敏感的环节换静电 | 中；要处理两套静电的一致性 |
| 给 CWLD 配倒空间补项 | 真正补上 $k<2\pi/r_c$ | 最高；等于放弃「无 FFT」这个卖点 |
| **紧支撑约束下的有限-$k$ 谱整形**（low-$k$ 加权的窗，重做 PSWF objective） | **不要叫「修复长波长 Coulomb」**——$\hat U(0)<\infty$ 而 $\widehat{1/r}=4\pi/k^2$，极点永远补不回。它的正当定位是：**实际盒子压根看不到 $k=0$**，最低可访问模式从 $k_{\min}=0.924$ nm⁻¹ 起，于是专门优化 $k\in[k_{\min},\,2\pi/r_c]$ 这一段，问能不能改变**有限盒实际能访问的长波模式**（§4.3 那 45% 就是可动空间） | 低（纯 numpy + 静态力审计即可判） |

**两条路线不冲突，分工是干净的**：

| | 管什么 |
|---|---|
| 有限-$k$ 谱整形（换窗） | **有限盒里有限 $k$ 的响应** |
| 倒空间 / Ewald 补项 | **恢复真正非局域的 Coulomb 极限（热力学 $k\to0$）** |

这正是这轮 PSWF 试验留下的理论结果：问题被劈成**拓扑性的不可达部分（pole）**
与**优化性的可调部分（finite-$k$ spectrum）**，两者不再互相偷换。

**⇒ 完整的路线重定级已写进 `../plans/PLAN_PSWF_Closure_Family.md` §11–§14**（2026-09-07）：

| 计划节 | 路线 | 状态 |
|---|---|---|
| §11 | **按计划原始三条目标计分**（2026-09-07 更正：我早前把「恢复 Coulomb 长波物理」当成该计划的目标并宣布它判死，**那个目标从来不在计划的 §2 里**）：目标 1「closure 抽象成窗层、ZMM 成为其中一个实例」与目标 3「保留 ZMM 入口」**已完成**；目标 2「做出新 closure」形式上完成但**精度上无收益** | — |
| §12 | **路线 A**：有限 $k$ 的 closure 优化（判据换成盒子实际存在的离散模式；$E_F$ 只当搜索代理，验收用力审计 + 实测 $S_{ZZ}$） | 便宜，先做 |
| §13 | **路线 B**（⚠ **撞在计划 §2 写死的非目标「不引入 FFT/reciprocal space」上，是 assistant 提出的范围扩展，需显式决策**）：local closure + 少量 low-$k$ reciprocal modes。**关键结果：修正权重不是自由参数，$w(k)=\hat c(kr_c)$** ⇒ 带内变精确 Coulomb 且与窗无关 ⇒ 窗只剩「局域力精度」+「带外残差」两个作用。375 个独立模式、**interaction-count 与一趟 pair traversal 同量级**（不是计算成本同量级），
**是 MTS 的自然候选，需实测 reciprocal force / $\lambda$ 的时间谱后再决定 outer step**
（CWLD 的 $Q_i=Q_i[\mathrm{dens}(\mathbf R)]$ 会让局部壳重排给 $\rho_{\mathbf k}$ 注入快时间尺度，
低 $k$ 的 phase 慢**不代表** $Q_i(t)$ 慢） | **最值得做**；**B0 已 PASS**，见下 |
| §14 | **路线 C**：Prolate Closure Family $\chi_{\ell,c}=(1-x^2)^\ell\psi_0^c$ —— **ZMM 变成 $c\to0$ 的退化极限**；$\ell$ 定衰减指数、$c$ 定带内形状。定位是「更一般、更漂亮、可调参」，**不宣称最优** | 活；**C0–C2 已 PASS** |

**（2026-09-07 晚，peer session `2-5-9e` 实测，见 `REPORT_HYBRID_LOWK_B0_2026-09-07.md`）**

* **B0 PASS**：CHECK-1 把 $w(k)=\hat c(kr_c)$ 这个形式与系数验到 **1.6e-12**（$q\in[0.24,60]$，四个窗）
  ⇒ §13.1 的推导是对的。合成体系上给 375 个模式，逐原子力误差
  **14–18% → 0.19–1.06%**（13.6×–90.5×）。
* **C0–C2 PASS**：PCF 的 $c\to0$ 逐位退化到 ZMM 冻结系数；偶多项式拟合残差 4e-15
  ⇒ §14.7 的选项 B1 成立，**下游（Lepton 串 / 插件系数表 / 解析导数）一个字节不用改**。
* ⚠ **我那个「pswfz2 只需 zmm2 的 1/4.8 模式数」的量级估算实测不成立**（我当时标了
  「未实测，引用前必须测」——现在测了）。节省倍率**不单调**，在最严目标上翻成劣势
  （1.00× / 1.00× / 3.34× / **0.68×**）。能站住的只有**残差比**：在标称带宽
  $k_c=2\pi/r_c$ 上 pswfz2 的残差比 zmm2 低 **3.8×**（1.89e-3 vs 7.12e-3）。
* ⚠ **新约束：窗的优劣与 $k_c$ 耦合**。实测 pswfz2 的 $\hat c$ 包络在 $q\approx8$ 被 zmm3、
  $q\approx13$ 被 zmm2 反超——**压带边换来一条更胖的尾巴**。
  所以 $k_c\approx2\pi/r_c$ 时 pswfz2 赢，$k_c\gtrsim10$ 时**提高 $\ell$ 更划算**（zmm3 反超全部候选）。
  ✅ **这与 §3.6 的渐近式一致**：$\hat c_\ell=O(q^{-(\ell+1)})$，指数只由 $\ell$ 定
  ⇒ 任何固定 $\ell$ 的优化（pswfz2 就是 $\ell=2$）只能在有界的 $q$ 窗口里赢一个前因子，
  走远了必然输给更高的 $\ell$。⇒ **设计规则：先由 $k_c$ 定 $\ell$，再在该 $\ell$ 内优化。**
  这正好是 §14 那个 $(\ell,c)$ 两参数 family 的用法。
**§13 的两个阻塞项（从 PLAN §13.4b 镜像过来 —— 你收不到 PLAN 那份，而这两条是
「数学对、Hamiltonian 实现错」最容易发生的地方）**：

> **H2 — response-consistent reciprocal correction（比 H1 更基础）**
>
> $$\lambda_i=\frac{\partial U_{\rm total}}{\partial Q_i}\quad\text{，}U_{\rm total}\ \text{是完整能量，含每一个「修正项」}$$
>
> **枚举方式必须是「照着能量表达式逐项求导」，不是「把项分成物理项和修正项」** ——
> 两次遗漏都出自后者：在代码里它看起来像个 correction，于是没人对它求导。
> 失效模式：**能量看起来补对了（带内精确恢复 $4\pi/k^2$），力却不是该 Hamiltonian 的导数**；
> 不会炸轨迹，只慢慢漂 —— 而 §9.3-R/R2 已证明那个通道在 fp32 与短窗下测不出来。
>
> | # | 通道 | $\partial/\partial Q_i$ | 为什么容易漏 |
> |---|---|---|---|
> | 1 | reciprocal pair | $2\mathrm{Re}[e^{-i\mathbf k\cdot\mathbf r_i}\rho^*_{\mathbf k}]$ | 唯一"显然"的 |
> | 2 | reciprocal **self** | $\propto2Q_i$ | 固定电荷下是常数、对力零贡献 ⇒ **PME 的习惯在 CWLD 下直接错** |
> | 3 | **背景/中性化**约定 | $\propto2\sum_iQ_i$ | 同上；且恰在 H1 被违反时非零 ⇒ 背景项就是 H1↔H2 的耦合处 |
> | 4 | **排除对扣除**（实空间） | $-\sum_{j\in\rm excl(i)}Q_jS_{\rm band}(r_{ij})$ | peer 补、我已独立验导数；1AAY 有 **43338** 个 bonded exclusion |
>
> ⚠ **一个反例，防"顺手一起修"**：RF 抵消项 $-q_{{\rm base},i}q_{{\rm base},j}(1/r-1/r_c)$
> 用的是 **qbase 不是 $Q$** ⇒ $\partial/\partial Q_i=0$，**它确实不进 $\lambda_i$**。
> 它长得最像该进的那一项，却是唯一不进的。（已核：`v26.py:1396` customgb / `:1476` fast 两处一致。）
>
> **H1 — 净电荷 $\sum_iQ_i\neq0$**：倒空间丢 $\mathbf k=0$ 等价于假设中性背景，
> 而 CWLD 的 $\Delta q$ 实测系统性同号（每水 $+8\times10^{-4}$ e）⇒ 先做 §9.5-1 的
> $\sum_i\Delta q_i$ 直接测量。
>
> ⚠ **B0 PASS 覆盖不到这两项**：固定电荷杀掉通道 1–4，严格中性格点独立杀掉通道 3，
> 合成格点没有 bonded exclusion ⇒ 通道 4 双重缺席。**B0 只验了 $w(k)=\hat c(kr_c)$ 的形式与系数。**

* 另：$k_c=0$（纯 closure）那一列构成一次**两体系交叉验证**，
  但**只有排序可引用，量级不可引用**（2026-09-07 peer 更正了我先前「量级也大致对上」
  的说法，那句话过于宽松）：

  | window | B0（合成，对 zmm2 归一） | 生产 1CKK 力审计 | 相对差 |
  |---|---|---|---|
  | zmm1 | 0.869 | 0.980 | −11.4% |
  | pswfz2 | 1.032 | 1.137 | −9.2% |
  | zmm3 | 1.098 | 1.263 | −13.1% |

  **排序完全一致**（zmm1 < zmm2 < pswfz2 < zmm3），这一条是真结论；
  但三个非基线窗在合成体系里**系统性地更靠近 zmm2**，一致偏 9–13%、**单方向**
  ⇒ 不是随机散度，是体系差异（合成 $\rho=3.18$/nm³ vs 生产 95.6 原子/nm³
  ⇒ $S_{ZZ}(k)$ 形状不同）。**引用时只说排序。**
| §14 | **路线 C**：Prolate Closure Family $\chi_{\ell,c}=(1-x^2)^\ell\psi_0^c$ —— **ZMM 变成 $c\to0$ 的退化极限**；$\ell$ 定衰减指数、$c$ 定带内形状。定位是「更一般、更漂亮、可调参」，**不宣称最优** | 活（叙事/定义层面） |

> 参考：Stillinger–Lovett 的 $S_{ZZ}\propto k^2$ 是 Coulomb perfect-screening 的长波长
> 约束（用户提供的参考：PMID 22181896）；PSWF 的 bandwidth-集中定义见
> Slepian–Pollak, Bell Syst. Tech. J. 40 (1961)，doi:10.1002/j.1538-7305.1961.tb03976.x。

---

## 5. 逻辑功能

### 5.1 包结构与职责

```
src/lips/
  paths.py       PROJECT_ROOT / DATA_DIR / RESULTS_DIR / INBOX_DIR 的单一来源
  systems.py   ★ 体系加载的唯一入口
  engine/
    v26.py       CWLD 引擎：closure 数学 + metadata + 力场装配 + MD 驱动（~2900 行）
    closure.py   窗函数族（zmm1/2/3、pswfz2），纯 numpy，不依赖 OpenMM
  build/         zinc_finger(1AAY) / zn_water 体系构建
  analysis/      szz / zn_coordination / force_audit / nve_drift / deltaq_probe /
                 dens_parity / block_time / kill_switch / dcd_integrity / force_cost
  run/zn_job.py  作业驱动（唯一真正跑 MD 的入口）
tests/           回归 + 静态检查闸门（含 lint 门：未定义的名字直接判失败）
openmm-localcwld/  LocalCWLDForce 原生 CUDA 插件（独立 CMake 工程）
docs/{reports,plans}/
data/            轨迹与产物，**平的一层**；inbox/ 放新上传的东西
logs/            作业日志
```

`data/` 保持平的一层是**故意的**：`szz` 这类「读自己写的 csv、同时扫 dcd」的工具需要
单一根，分子目录得给它两个根，收益不抵成本。`inbox/` 是唯一例外——分析的 glob 不扫它。

### 5.2 九个逐原子参数（CWLD 的全部输入）

`build_phase_cwld_metadata()` 的产物，也是插件 `addParticle()` 的签名：

| # | 参数 | 语义 | 谁拿到非零值 |
|---|---|---|---|
| 1 | `qbase` | 基础电荷，从 `NonbondedForce` 提取 | 全部 |
| 2 | `charge_mod` | $\mathrm{clip}(1+a_{q^2}(q^2-q_{\rm ref}^2))$ | 全部 |
| 3 | `dpolar` | 响应幅度**与符号** | 水 O/H、溶质 O/N/S、被覆盖的配位原子 |
| 4 | `is_polar` | 「我是响应粒子吗」的开关 | 同上 |
| 5 | `dens_source` | 「我产生环境密度吗」 | 水 O、离子、溶质重原子（受 `FAST_ACTIVE_DENSITY` 与 `enable_solute_polarization` 影响） |
| 6 | `dens_sink` | 「我感受环境密度吗」 | `is_polar>0` 的原子 |
| 7 | `source_class_weight` | 作为源的权重（§2 的表） | `dens_source>0` 的原子 |
| 8 | `static_phase` | 相位/界面开关，可整体关掉某类响应 | 响应粒子 |
| 9 | `mol_id` | **实为 `residue.index`**，用于同 residue 排除 | 全部 |

**source 与 sink 是两套独立的掩码**，这是模型的核心不对称性：
水 O 永远是源（不论水响不响应），而它只在 `enable_water_response` 时才是 sink；
金属离子是**强源但不是 sink**（$2.0\times$ 权重，自身不响应）。

**为什么金属必须是源**：整个 1AAY 结论都挂在这上面——
`normalize_ion_resnames()` 漏调一次，Zn 的 `dens_source` 静默变 0，
配位原子处的密度 **2.123 → 0.033（差 65 倍）**，CWLD 机制整个关掉且不报错。
现已加硬断言：任何 Ca/Mg/Zn 的 `dens_source=0` 直接抛异常。

### 5.3 一次生产跑的数据流

```
1AAY.pdb ──lips.build.zinc_finger──▶ system.xml + solvated.pdb + meta.json
                                          │（meta.json 存 hid_resids / hie_resids）
                                          ▼
                              lips.systems.load_system()
                                 ├ normalize_ion_resnames()   ← 漏了它 CWLD 静默关闭
                                 ├ 按 meta.json 还原 HID/HIE  ← PDB 往返会塌回 HIS
                                 └ 断言二价金属 dens_source>0
                                          ▼
                     build_phase_cwld_metadata()  → 9 个逐原子参数
                                          ▼
              setup_cwld_lips_system(engine="plugin"|"customgb")
                   ├ 原生 NonbondedForce: CutoffPeriodic, rc=1.2, eps_RF=1, switch@0.9
                   └ CWLD 力: dens → Q → pair(含 RF 抵消) → 链式力
                                          ▼
                     lips.run.zn_job  →  *_traj.dcd + *_top.pdb + *_state.csv
                                          ▼
       lips.analysis.{zn_coordination, szz, deltaq_probe, dcd_integrity}
```

**两个体系构建期的陷阱**（都在 §7.1）：
`hydrogens.xml` 的 CYS 条目**没有 CYM 变体** ⇒ Cys 必须「先加氢 → 删 HG → 改名 CYM」；
`Modeller.addHydrogens(variants=...)` **只按变体加氢、不改残基名**，
而 `app.PDBFile` **读盘时用替换表把 HID/HIE 塌回 HIS**（写是保留的，丢在读这一侧）。

### 5.4 CWLD 引擎双轨

| `engine` | 实现 | 用在哪 |
|---|---|---|
| `"plugin"`（默认） | `LocalCWLDForce` 原生 CUDA 插件 | **新跑的一切** |
| `"customgb"` | `mm.CustomGBForce` | 只用于补跑 2026-09-05 之前的既有矩阵 |

⚠ **两者数值等价但不是逐比特相同**（逐原子力 p99 3.3e-3，DEC-005 阈值 1.0），
**同一个统计量里不能混**——2.4 的 5 ns 矩阵就是被引擎不一致混掉过一次。
产物带 `cwld_engine` 字段（`cwld_engine_of()`），因为**光看一个 dcd 分辨不出它是哪个
引擎跑的**。`force_cost.py` / `block_time.py` 已显式钉死 `customgb`。

插件从 repo 内 `openmm-localcwld/build-cuda/` 加载，**不装进共享 conda env**——
那个 env 有别的项目在用，往它的 plugin 目录放东西会让机器上**所有** OpenMM 进程加载这个力。
⚠ 一棵构建树只能属于一台机器（`CMakeCache.txt` 存编译器绝对路径；仓库在 NFS 上被多台
机器挂着，共用 build 目录会让后一台拿着前一台的配置去建、失败、并删掉前一台的产物——
2026-09-05 实际发生过）。别的机器用 `LOCALCWLD_BUILD_DIR=build-cuda-<机器名>`。

### 5.5 分析工具（每个是一个 console script）

| 命令 | 干什么 | 跑 MD 吗 |
|---|---|---|
| `lips-run-zn` | Zn 判据作业 | **是** |
| `lips-nve-drift` | NVE 能量漂移（力审计量不到的通道） | **是**（NVE） |
| `lips-force-audit` | 静态力审计：固定坐标/电荷/排除表，参照精确 Ewald，**只换 pair kernel**，零 `Integrator.step()` | 否 |
| `lips-szz` | 从已有 dcd 实测电荷结构因子 $S_{ZZ}(k)$；阶段 A 默认走 GPU/float32 | 否 |
| `lips-zn-coord` | Zn 配位数/几何的逐帧分析 | 否 |
| `lips-deltaq` | Δq 定点探针（显式原子表，不抽样） | 否 |
| `lips-block-time` | 分块收敛检查 | 否 |
| `lips-dcd-check` | 轨迹完整性校验 | 否 |
| `lips-build-1aay` / `lips-build-znwat` | 建体系 | 否 |
| （无入口）`lips.analysis.dens_parity` | dens 实现间 parity + 耗时交叉点 | 否；**2026-09-07 修掉三个 bug 后首次跑通，PASS**（§9.1-R） |
| （无入口）`lips.analysis.kill_switch` | 窗族解析判据，**已被力审计推翻**，只在 >10% 差别上参考 | 否 |

体系是参数：`--system 1aay`（默认）/ `1aay-amoeba` / `1ckk`，
或 `--topology X.pdb --system-xml Y.xml` 指任意预建体系。

### 5.6 环境变量总表

| 变量 | 取值 | 作用 | 改了要注意 |
|---|---|---|---|
| `L_IPS_CLOSURE` | `zmm`(默认) / `pswfz2` | closure 家族入口 | 默认路径不导入窗层 |
| `L_IPS_ZMM_ELL` | 1/2/**2**/3 | ZMM 阶数 | 改前必须先做 NVE 对照（§9.3） |
| `L_IPS_LIGAND_DPOLAR` | 浮点，默认**未设** | 配位原子 dpolar 覆盖幅度 | **影响物理，必须进 tag** |
| `L_IPS_LIGAND_SCOPE` | `metal`(默认) / `all` | 覆盖范围 | 同上 |
| `L_IPS_RUN_MODE` | 7 种矩阵模式 | 实验矩阵调度 | — |
| `L_IPS_ONLY_CONFIGS` | 逗号分隔 | 只跑指定臂 | — |
| `L_IPS_RUN_TAG` | 字符串 | 产物文件名后缀 | **两次不同的跑不能同名**（§5.7-5） |
| `L_IPS_DATA_DIR` / `L_IPS_RESULTS_DIR` | 路径 | 覆盖数据根 | 复算 1CKK 旧数据时指到 `../2.4` |
| `L_IPS_PLATFORM` | `cpu` | 调试用，会打印醒目警告 | 产出**不算生产数据** |
| `LOCALCWLD_BUILD_DIR` | 路径 | 插件构建树 | **每台机器一棵** |
| `LOCALCWLD_PROBE_*` | — | 插件性能探针（单变量 A/B） | 只用于归因，不进生产 |

### 5.7 被制度化的五条约束

这几条都是**事故换来的**，不是风格偏好：

1. **体系加载只能走 `lips.systems`。** 打包前它在三个脚本里各写一份、各自演化，
   `normalize_ion_resnames()` 只在其中一份被调到 ⇒ 金属 `dens_source=0` ⇒
   **CWLD 机制整个关掉且不报错**，实测废掉过一整批轨迹。
   `tests/test_single_entry_point.py` 会在有人写第二份时失败。
2. **1CKK 相关默认值是复现锚点，不要「清理」**：`L_IPS_CLOSURE=zmm`、
   `L_IPS_LIGAND_DPOLAR=None`、`CHARGED_SIDECHAIN_POLAR_ATOMS` 里的原 `"HIS"` 条目——
   它们存在的唯一理由就是保 1CKK 逐位复现。
3. **路径只能来自 `lips.paths`**；挪数据之前**先确认没人在跑、留清单**。
   （原文是「数据不由代码移动」，起因是一次挪动正撞上另一个会话在跑分析。
   2026-09-05 又挪了一次 118 个文件 / 17 GB，这次先确认了无进程在写、两小时内无改动、
   所有读取都走 `DATA_DIR`，并留了清单。约束的本意是**先查、再挪、留清单**，
   不是「永远不能整理」。）
4. **分类按元素判，不按残基名判。** `force_audit.species_of()` 原本写死 `el=="Ca"`
   （1CKK 钙调蛋白时代），换到 1AAY 之后**锌被静默归进 `protein_heavy`**，
   金属那一列和整个分壳 breakdown 消失且不报错。
   新增元素改 `DIVALENT_BY_ELEMENT` / `lips.systems.DIVALENT_METAL_SPECIES`。
5. **产物文件名必须带上「是什么跑出来的」**（引擎、精度、closure、全部影响物理的开关）。
   **加新维度时先问「两次不同的跑会不会同名」。**

### 5.8 插件的架构与验收设计

标准 OpenMM 插件分层：公共 `Force` API → `ForceImpl` → 抽象 kernel → 各 Platform 的
kernel 实现。目录：`openmmapi/` `platforms/cuda/` `serialization/` `python/` `tests/`。

**Python 通路有两条**（先后做的，都保留）：
`serialization/` 的 `LocalCWLDForceProxy` 走 XML（Python 不构造对象，
`XmlSerializer.deserialize` 在 C++ 侧建），注册靠 `__attribute__((constructor))`，
加载 .so 即完成注册；以及 LCWLD-080 的 SWIG 类型化绑定（`localcwld.LocalCWLDForce`）。

**验收设计（DEC-005，在 kernel 一行代码都还没写的时候落盘）**——
理由写得很直白：**先写 kernel 再定阈值，阈值一定会被拟合到跑出来的偏差上。**

* **Oracle 是生产 `CustomGBForce`，真实 1AAY 体系，GPU 对 GPU**，不是 CPU Reference。
  要验的是「新 kernel 复现生产里真实在跑的那个东西，**包括它的 fp32 舍入**」。
  `localcwld_reference`（float64）**只作为定位 oracle**，不进论证链——
  因为对不上时故障可能在 host 层 / device kernel / metadata 映射三处，
  只有一个对照点无法把三者分开。
* **必须逐粒子比，不许比总量**：32794 个原子里 13 个偏 3%，看总能量根本看不出来。
  对照集：A=13 个覆盖命中原子；**B=`HIE149:ND1` 单独一行**（距 Zn 13.13 Å，
  是唯一「覆盖生效但环境是体相」的原子，**天然的假阳性探针**：
  若它的 dens 跟配位原子相当，说明 dens 算错了）；C=3 个 Zn²⁺；
  D=随机 200 水 O + 200 水 H；E=全体原子的 p50/p99/max。
* **分级报偏差**：`dens` → `Q` → `energy` → `force`。偏差在哪一级出现直接指向哪段代码。

固定物：9 个数值 fixture（`two_particle_directional` / `three_particle_chain` /
`same_residue` / `excluded_pair` / `pbc_cross_boundary` / `water_ca_cluster` /
`zmm_orders` …）+ **sha256 冻结清单**。改数学 ⇒ 这些 fixture 会红，
必须用生成脚本重新生成并更新 sha256，同时在 DEC-030 里记明原因。

### 5.9 精度政策

平台固定 CUDA，生产/GPU 路径一律 **float32**（DEC-004）；float64 只保留在
**CPU 审计 oracle** 里。这不是妥协：生产轨迹本来就跑单精度，
把审计和生产分开才能让审计的差值有意义——例如 §7.1 那个
「fp32 噪声地板 abs_p99 = 0.122 kJ/mol/nm」正是用双精度作差测出来的，
它把「HIE149 的 0.033」判成了**低于噪声**，而不是「小但存在」。

---

## 6. 同一套数学的五份实现

CWLD 的**同一套数学在仓库里有 5 份独立实现**（`MATH_CHANGE_MAP.md`）。
只改其中一处，剩下的会静默不一致，**且不一致的方式是「结果照样跑出来、数字看着正常」**。

| # | 位置 | 角色 | 改数学时要动什么 | 验过吗 |
|---|---|---|---|---|
| 1 | `v26.py::setup_cwld_lips_system` | 生产 MD（CustomGB 轨） | Lepton 字符串；导数 OpenMM 自动求 | 是（是 oracle） |
| 2 | `engine/closure.py` + `pair_uclosure` | closure $U(r)$ | ZMM 三档是**冻结字符串**，改了 fixture 的 1e-12 断言变红 | 是（逐位断言） |
| 3 | `v26.py` 的**分析侧镜像**：`_local_dens_at` / `_delta_q_from_dens` / `compute_fast_cwld_q_from_positions` / `compute_water_q_profile` / `compute_water_net_delta_q` | 事后从 dcd 重算 q-profile 与 Δq | **最容易漏**。漏了 ⇒ MD 用新公式、图用旧公式，图完全是假的 | ✅ **2026-09-07 已验**：对独立向量化实现 `max rel = 1.7e-16`（§9.1） |
| 4 | `localcwld_reference/reference.py` | 插件 golden 参考（纯 numpy 显式循环） | 能量式**和解析导数** `zmm_duclosure_dr`、$\lambda$ 链式项都要手改 | 是（fixture + LCWLD-030） |
| 5 | `localcwld_fast/evaluator.py` | 加速 CPU 后端 | 同上，另外 closure 系数被预计算成表 | 是 |

（#1–#3 在同一个文件里，但是**三条彼此独立的代码路径**。）

**#3 与 #1 已知的一处真实语义分叉**：fast 路径有 `q_driver`（水的三个原子共享 O 的 dens），
**生产路径没有**（各自按自己位置算）。见 §3.4.3。

外加会把旧公式钉住的「冻结物」：9 个 fixture + sha256；
`PLAN_LocalCWLDForce_OpenMM_Plugin.md` §5/§18/§31（插件 kernel 的**唯一规格来源**，
不同步改，将来写 kernel 的人会照旧公式实现）；`DEC-002`（density kernel 的
tabulated vs analytic 证据，状态是「证据已收集、结论待签署」；若 $K(r)$ 形式要变，
那份证据直接作废）。

---

## 7. 现状

### 7.1 主线：1AAY 锌指

**为什么换掉 1CKK**：1CKK 的 Ca²⁺ 被 EF-hand 螯合死——实测
`ca_protein_o_coord`=6.60 vs `ca_water_coord`=**0.92**，7.5 个配位氧里只有 0.9 个是水。后果：
(a) `ca_water_peak_g`/residence τ 实际在**测一个水分子**，这才是 40–45% 运行间噪声的来源；
(b) 局部密度恒定 ⇒ Δq 恒定且极小（螯合羧基氧只有 clamp 的 0.56%）；
(c) **最致命：PME 在 1CKK 上本来就对**（Ca 总配位 7.66，与实验相符）⇒ **headroom 为零**。

**判据因此换成**：不看「第一壳有几个水」，看**固定电荷基线是否明确失败、且已知正确答案**。
1AAY 满足：固定电荷 Amber = 已知失败 / AMOEBA·Drude = 已知成功 / CWLD = 待测。
**Zn 必须用标准 12-6 `charge="2.0"`，不要 12-6-4**——那等于先用一个 $r^{-4}$ 项手工补掉
失败，CWLD 就没东西可证明。

**修的三个 bug（全部会静默污染结论）**：

| bug | 机制 | 后果 | 防线 |
|---|---|---|---|
| Zn 没被认成密度源 | 判的是 `resname in ION_ALIASES.values()`（`Zn2+`），而 PDB 里是 **key**（`ZN`） | 配位处 dens **0.033 → 2.123（65 倍）** | 二价金属 `dens_source=0` 直接抛异常 |
| 残基名在 PDB 往返中丢失 | `addHydrogens` 不改名；`PDBFile` 读盘塌回 HIS | 无法判断哪个氮去质子（HID→NE2 / HIE→ND1） | 从 `meta.json` 还原 |
| dpolar 按元素给 | 152 个 N 里只有 6 个配位（**3.9%**），全拿同一个 dpolar，**且符号为正** | 幅度天花板 6%、符号还反了 | `LIGAND_DPOLAR_ATOMS` 逐原子覆盖，判据是**化学的** |

第三个 bug 值得多说一句：同一个 HID 的 NE2（去质子、配位 Zn）与 ND1（带 H、朝外）
拿到**同一个** dpolar，**全靠局部密度碰巧把它们分开**（NE2 dens=2.12 / ND1 dens=0.00）——
那是几何巧合，不是化学识别。补 `CHARGED_SIDECHAIN_POLAR_ATOMS` 的 HID/HIE/CYM 条目
同样重要：不补，最该被极化的 12 个配位原子会被当成普通极性原子，
**作为密度源少 33% 的权重**（1.0 vs 1.5），等于在要看的位点上自我削弱。

**结果与归因**：

| 对照 | 核心完整 p | Zn–N p | 读法 |
|---|---|---|---|
| 配位极化 vs PME | **0.0053** | **0.0004** | 主结果 |
| 配位极化 vs 只修 Zn | **0.0140** | **0.0004** | 效果不来自密度源修复 |
| 只修 Zn vs PME | 1.0000 | 0.79 | 只修 Zn 与 PME **完全无差别** |
| metal(13) vs all(31) | 1.0000 | 0.93 | 多带 18 个非配位羧基氧**结果完全不变** |

⇒ **效果可归因到 12 个真配位原子**（6×CYM:SG + 6×HID:NE2）。
覆盖判据命中 13 个，第 13 个是 **HIE149:ND1，距最近 Zn 13.13 Å，根本不配位**——
它被命中是因为判据按设计是**化学的**（去质子咪唑氮 = 孤对给体）而非几何的。
该对照已用**力学证据**关闭（不必再跑一档 MD）：

| | ΔF = \|F(dpolar=−0.15) − F(off)\| (kJ/mol/nm) |
|---|---|
| 12 个真配位原子 | 最小 106.4 / 中位 197.4 / 最大 254.0 |
| **HIE149:ND1** | **0.033** |
| fp32 噪声地板 (abs_p99) | 0.122 |

**dpolar 到底做了什么**：只看**全程完好**的位点，dpolar 让 Zn–N 只缩短 0.030 Å；
主表里 2.871 vs 1.912 的巨大差距**几乎全部来自解离的位点**。
⇒ **dpolar 的作用是防解离，不是缩键长。**

| | 完好位点数 | Zn–N |
|---|---|---|
| PME | 4/9 | 1.987 |
| CWLD ligoff | 4/9 | 1.942 |
| CWLD −0.15 | **9/9** | 1.912 |

**幅度扫描**：−0.15 → 100%（0/9 漏）；**−0.08 → 96.0%（1/9 漏）**；+0.012（关）→ 62.9%（5/9 漏）。
转折点在 0.08 附近，`−0.15` 买的是最后那 4%。
⚠ 不要把 −0.08 的 Zn–N 1.995 读成「键长更准」——它落在晶体区间内是**被那个 2.58 Å 的
漏点抬上去的**；八个完好位点是 1.91–1.93，与 −0.15 一样。**幅度改不了键长偏短。**

**保留意见（原报告 §5，务必随结论一起引用）**：
① Zn–S 在**所有**方法里都比晶体短 0.2–0.45 Å，AMOEBA 也是 ⇒ 力场层面的事，与 dpolar 无关；
② Zn–N 比晶体下限短 0.028 Å，且 9 个位点散度只有 ±0.002 Å，比晶体的位点间差异
（跨度 0.14 Å）小两个量级 ⇒ −0.15 可能强于必要；
③ **AMOEBA 只有 1 ns × 1 seed，不能声称 arm 间排序**；
④ **原子数口径不一致**：PME/pme_q/AMOEBA 是 32818，三个 CWLD arm 是 32794
（中间重建过体系）⇒ **CWLD arm 之间对照干净，CWLD vs PME 等价于换了水的随机种子**；
⑤ 平衡阶段就会破坏部分位点（PME 有位点在 production 第 0 帧已被拉开，最远 9.60 Å），
而平衡段没存轨迹。

**1CKK 保留的三个角色**（所以 2.4 的 88 个 dcd 与全部 1CKK 产物**不得删除**）：
① **回归基线**——引擎里所有新开关默认值的唯一存在理由就是让 1CKK 逐位复现；
② **负对照（有真实科学价值）**——1AAY 证明 CWLD 能救回 PME 救不了的位点，
审稿人接着会问「那它是不是把**所有**东西都抓紧了？」，
1CKK 是「PME 本来就对」的情形，CWLD 在那里不把 Ca 配位搞坏 ⇒
增益来自局部响应而非全局过度收紧。**这一条尚未在换体系后重新核对**；
③ **方法学工作的载体**——closure kill-switch、静态力审计、$S_{ZZ}$ 测量都在 1CKK/纯水上做，
结论关于 closure 本身，与体系无关。

### 7.2 插件：LocalCWLDForce

**已完成，ctest 7/7，Python 35 passed，无待办。**

性能的关键一跳（LCWLD-161）**没有碰 pair kernel**：真因是
`LocalCWLDForceInfo::areParticlesIdentical` 比较了 `residueId`，于是任意两个不同残基的
原子都不可互换，**OpenMM 的原子重排（只在「相同分子」集合内部生效）等于不做**，
共享邻居表失去局部性，**所有走这张表的 kernel 慢 2–5 倍，包括我们自己的 pair kernel**。

修法：`areParticlesIdentical` 不再比 `residueId`。**为什么安全**——kernel 只对它做
相等判断、从不读值，物理只依赖残基**分区**；而 OpenMM 搬的是整个分子，
排除表的传递闭包就是成键连通性，所以分区在重排后依然成立。
前提由 `residuesFitInMolecules()` 并查集**实测**，**不成立时退回严格比较，不抛错**。
`LOCALCWLD_PROBE_STRICT_FORCEINFO=1` 恢复旧行为（同 binary A/B）。

| 量 | 2080 Ti | 3090 |
|---|---|---|
| 修复前端到端 | 1.548x | 1.454x ← **低于 1.5x 线** |
| 修复后端到端 | **3.1–3.6x** | 2.80–3.28x |
| CWLD 力的 rc 遍历当量 | 4.83 → 1.51 | 5.18 → 1.72 |

⚠ **修复前的实现在 3090 上不达标**——LCWLD-150 的那个 PASS 是**硬件相关且勉强**的。
这次修复把一个会随硬件翻车的结论变成了稳固的。

相对 PME（3090 同机同 harness）：PME 497.2 / CWLD 插件 212.0 / CWLD 旧引擎 41 ns/day
⇒ **8.2% → 42.6%**。29×5 ns 矩阵 48.4 → 14.4 GPU-h。

**性能的会计**（全部单变量实测，前两版靠代数推的模型都被推翻过）：

```
一趟普通 rc 遍历 = 0.1305 个 CustomGBForce；CustomGBForce = 7.7 趟
pair pass 占我们成本的 88.6%，buildEnvPairs 11.4%，density+chain ≈ 0（低于噪声）
```

⇒ **剩下的性能全部在 pair kernel 里**：它当时 4.49 趟，物理只要 1–1.5 趟。
但 **LCWLD-160（pair kernel 重写）已决定关闭，从未开工**；
闸门诊断**已修**（能区分「各臂同步漂」与「各臂漂速不同」，
判据边界由 `check_gate_diagnosis.py` 用三组真实数据钉住——
⚠ 它是 C++ 判据的副本，**改阈值要改两处**）；
**fp64 能量累加不追**（约 27 µs/步 ≈ 8%）。

⚠ **引用 3.1–3.6x，不要引用 3.391x**：闸门给的是 NO VERDICT。
原因不是抢卡（卡干净，47→73 °C 纯热降频），而是 **new 臂快了一倍后发热骤减、
不再跟对照臂同步漂**，于是轮转不再让比值免疫。**加轮数解决不了。**
保守下界 3.075x（两次跑的最冷轮）。

⚠ 插件 kernel 读的是 OpenMM **内部**邻居表布局（`exclusionTiles` / `interactingAtoms` /
`interactionCount`，**不在安装的头文件里**），照 `36a30cb` 的源码转写。
换 OpenMM 版本**不会编译报错、也不会加载失败，只会静默算错力**——
`ensure_localcwld_plugin()` 里有一道 `RuntimeWarning` 比对 `git_revision`。

### 7.3 closure 路线（PSWF）：已判定无收益，路线未关

* **Stage 0 完成**：窗层落地并接入生产，ZMM 三档构造时即断言与冻结系数逐位一致。
* ⚠ **「closure 与隐式极化是正交的两个变量」这句话不对**（2026-09-07 修；
  原文在 `PLAN_PSWF_Closure_Family.md` §（Stage 顺序）那一节）。固定 $Q$ 时两者可以
  **分阶段隔离测试**（静态力审计就是这么做的），但在完整 CWLD 里
  $\lambda_i=C\sum_j Q_jU_\ell(r_{ij})$、$g_i=\lambda_i\,dQ_i/d\,\mathrm{dens}_i$，
  **换 $U_\ell$ 会直接改变链式响应力** ⇒ 数学上不正交。
  正确说法是「**可以分阶段隔离测试**」，不是「正交」。
* **Stage 1 判据（$E_F$）被静态力审计推翻**。审计设置：固定坐标、固定电荷 qbase、
  同一排除表，**唯一变量是 pair kernel**；参照 = NonbondedForce + PME（Ewald 容差 1e-6），
  被测 = CustomNonbondedForce + $U(r)$；20 帧 × 4 kernel，CUDA 双精度，零 `Integrator.step()`。

  相对 RMS 力误差（对 zmm2 归一，<1 更好），PME 系综构型：

  | kernel | 总体 | Ca²⁺ | 第一壳 | 第二壳 | 水O | 蛋白重原子 |
  |---|---|---|---|---|---|---|
  | **zmm1** | **0.980** | **0.860** | **0.844** | **0.788** | 0.996 | **0.789** |
  | zmm2（生产默认） | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
  | pswfz2 | 1.137 | 1.071 | 1.146 | 1.233 | 1.129 | 1.221 |
  | zmm3 | 1.263 | 1.201 | 1.266 | 1.364 | 1.253 | 1.369 |

  CWLD 系综构型给出**同样的排序**，但绝对误差高 2.7 倍（5.6e-2 vs 2.0e-2），
  与 §7.4 的长波长屏蔽不足一致。

#### 7.3-R 在 1AAY（当前研究体系）上的复核 + PCF 首测（2026-09-07 实测）

产物 `closure_force_audit_ZN_1aay_cwld_ligm0p15metal_zmm_seed0_double_9k7bd0b0.csv`
（20 帧 × 9 kernel，CUDA/double，零 `Integrator.step()`）。对 zmm2 归一，<1 更好：

| kernel | rel_rms | 第一壳 | Zn²⁺ | 蛋白重原子 |
|---|---|---|---|---|
| **zmm1** | **0.926** | **0.849** | **0.867** | **0.829** |
| zmm2（生产默认） | 1.000 | 1.000 | 1.000 | 1.000 |
| zmm3 | 1.089 | 1.121 | 1.096 | 1.154 |
| pswfz2 | 1.037 | 1.051 | 1.030 | 1.072 |
| **pcf1_c2** | **0.947** | 0.911 | 0.921 | 0.900 |
| pcf1_c4 | 1.025 | 1.037 | 1.025 | 1.051 |
| pcf1_c6.28 | 1.125 | 1.168 | 1.121 | 1.221 |
| pcf2_c4 | 1.122 | 1.163 | 1.121 | 1.213 |
| pcf2_c6.28 | 1.221 | 1.280 | 1.194 | 1.375 |

**三个读数**：

1. **zmm1 在当前研究体系上也赢，这是第四个独立系综。** 而且第一壳的数字
   **0.849 vs 1CKK 的 0.844** —— 跨体系吻合到 0.6%，这不像巧合。
   四个系综：1CKK/PME 0.980、1CKK/CWLD 0.938、合成格点 0.869、**1AAY/CWLD 0.926**。
   ⇒ 「zmm1 局域力更准」已经不是单体系观察。
   ~~换生产默认的唯一缺口仍是 NVE~~ ⇒ **该缺口已于当日 20:02 关闭，见 §9.3-R4**：
   NVE **不区分**（0.04–0.69σ，dt² 标度已验证 0.27–0.28 对预期 0.25）
   ⇒ **证据链闭合，应换 zmm1。**
2. **PCF 的 $c$ 旋钮在纯 closure 判据下单调变坏**（$\ell=1$：$c\to0$ 即 zmm1 0.926
   < c=2 0.947 < c=4 1.025 < c=2π 1.125），$\ell$ 同样（c=4 时 $\ell$=1 的 1.025 <
   $\ell$=2 的 1.122）。**没有任何 PCF 成员打得过 zmm1。**
   ⇒ **§14.6 那个可被打脸的预测在「静态力审计」这个判据下实测成立。**

   ⚠ **但必须点明是哪个判据 —— 换判据结论就变**（2026-09-07，与 peer 结果对齐）：
   peer 在**带外能量**（PSWF 的谱判据）上测到 $\ell=1$/$\ell=2$ 有**真的内点最优**
   （$c\approx4$ / $c\approx2.5$，带外能量到 zmm2 的 0.163 / 0.234），只有 $\ell=3$ 才单调。
   两个结果**不冲突，正是 §13.2 那个分解**：窗同时影响 (a) 局域力精度、(b) 带外残差，
   而 $c$ 在这两个判据上的最优方向**相反**。
   ⇒ 正确表述：**$c$ 在纯 closure（力精度）判据下是净损失；它的价值只在带外残差
   成为判据的场合，也就是 §13 的 hybrid。** 说「c 这个旋钮没用」是错的，
   说「c 在力精度上没用」才对。
   （peer 另测到 hybrid 主表里 pcf1_c4 仍不如 zmm3 ⇒ **hybrid 下的胜负手是 $\ell$ 不是 $c$**，
   与 §3.6 的渐近式一致：指数由 $\ell$ 定，$c$ 只动前因子。）
3. **pswfz2 1.037** —— 与 peer 合成体系的 **1.032** 几乎相同，而 1CKK 上是 1.137。
   ⇒ 「pswfz2 差于 zmm2」在三个系综上一致，**幅度是体系相关的**（3–14%），
   与 §4.7 那条「排序可引用、量级不可引用」一致。

绝对值 rel_rms 4.8–6.3e-2 与 §7.3 的 CWLD 系综（5.6e-2）同档，内部自洽。

  ⇒ **pswfz2 每个通道都差 3–15%，13.7 倍的带外能量削减没换来任何力精度。**
* **意外收获**：`zmm1` 在两个独立系综上都比生产默认 `zmm2` 更准，
  且在第一壳（0.844）/第二壳（0.788）/蛋白重原子（0.789）上差距远大于总体的 2%。
  这与 Sakuraba 论文偏好 $\ell=2/3$ 冲突——合理解释是**论文偏好高 $\ell$ 不是为了力的
  RMS 精度，而是 cutoff 处更高阶光滑**（能量守恒/积分器稳定性），
  那是静态审计**量不到**的通道。⇒ **动 `L_IPS_ZMM_ELL` 之前必须先做 NVE 对照。**
* **NVE 首轮数据作废**：`drift_of()` 在 NVT 平衡前无条件 minimize
  （那行是为修合成体系 NaN 加的，对**已平衡的真实帧**是错的：它把体系推离平衡，
  之后的弛豫让 NVE 的积分误差随时间变化，线性拟合的斜率因此没有物理意义）。
  被自带的 dt 标度对照判死：真实积分器误差 $\propto dt^2$，dt 减半应降到 0.25 倍且
  **符号不变**，实测 **5 个 kernel 全部翻号、量级还略增**。
  脚本已修（minimize 默认关、加线性度检验、存 $E(t)$ 原始轨迹、平衡 50→200 ps、
  汇总固定打印 dt 标度判据）。
  ⇒ ~~待重跑~~ **已重跑并关闭，见 §9.3-R4**。本条与 §9.3-R/R2/R3 一起
  **降级为「历史失败尝试」的记录**：它们的价值在于记下三次失败各自的原因
  （minimize bug → fp32 地板 → 窗太短 + 判据被对称鼓包骗过），**结论一律以 §9.3-R4 为准**。
  ⚠ 一并修掉一个错误论断：原脚本打印「PME 参照 = 漂移地板，低于它都可疑」——
  **这是错的**：PME 有倒空间网格插值误差，而一个在 $r_c$ 处光滑归零的截断 kernel
  **本身就是严格保守的 Hamiltonian**，能量守恒优于 PME 是正常的。

### 7.4 物理侧发现：截断静电的长波长屏蔽不足

12 条轨迹沿方法**完美劈成两组、零重叠、组内 seed 散度极小**：

| | ~~$\kappa$~~ (nm⁻¹) | $k<k_{\min}$ 段对 surrogate 积分的贡献占比 |
|---|---|---|
| PME ×3 | 125.7 – 129.8 | 7.0% |
| CWLD ×9（三种配置全部） | ~~29.6 – 31.8~~ | 17.6% |

⚠ **这张表的两列都要改口径**（2026-09-07，见 §7.4.1）：

* $\kappa$ 对**截断组**只能叫「**供代码外推用的 fitted summary parameter**」，
  **不是 screening constant**；对 PME 组仍是有意义的屏蔽长度。
* 「低 k 外推占比」这个名字会让人以为「我们知道真实低 k 区有 17.6%」。
  实际上 $0<k<k_{\min}$ 这一段对截断组**整段都是 imposed continuation
  （强加的 $k^2$ 延拓），不是 measured long-wave behavior**。
  准确的名字是「**$k<k_{\min}$ 的强加 $k^2$ 延拓对 surrogate 积分的贡献占比**」。

⚠ **早前这里写「$\kappa$ 差 4.2 倍 ⇒ 低 k 的 $S_{ZZ}$ 差 ~17 倍」——该推断已作废**
（2026-09-07，见 §7.4.1）：它假设两组都服从 $S_{ZZ}=k^2/(k^2+\kappa^2)$，
而截断组**不服从**。逐 k 直接比对才是数据：

| k (nm⁻¹) | 0.924 | 2.17 | 3.85 | **5.24 = 2π/rc** | 5.3–40 |
|---|---|---|---|---|---|
| CWLD/PME | **44.0** | 2.66 | 1.23 | — | **0.97–1.02** |

**实测 crossover 位于 cutoff 波数 $2\pi/r_c$ 附近**（⚠ 不能写"精确落在"：
紧支撑 kernel 的 Fourier 变换是解析的，**没有 hard switch**）；
在当前采样的 $k>2\pi/r_c$ modes 上吻合到 1–3%
（峰位 30.3 vs 31.3 nm⁻¹、峰高 1.22 vs 1.20 都一致）。
这**基本排除了电荷定义伪影**（那会影响所有 k，不会精确地在 $2\pi/r_c$ 处开关）。

判别实验（`2.4/0706/` 同目录同协议，500 ps / 60 帧）把两个混在一起的变量分开了：

| 轨迹 | 静电 | 电荷 | $S_{ZZ}(k_{\min}{=}0.922)$ | 对 PME 的倍数 | ~~$\kappa$~~ |
|---|---|---|---|---|---|
| `PME_Baseline` | 全 Ewald | 固定 | 5.22e-5 | 1 | ~~131.8~~ |
| `LIPS_fixedQ_baseline` | ZMM 截断 | **固定** | **1.292e-3** | **24.7×** | ~~41.8~~ |
| `Both_on_aq0p5` | ZMM 截断 | CWLD 极化 | **1.154e-3** | **22.1×** | ~~42.5~~ |

（$S_{ZZ}(k_{\min})$ 三列是 2026-09-07 从 `data/szz_0706__*.csv` 直接读的，
用来**取代 $\kappa$** 作为判别量——理由见 §7.4.1。原 $\kappa$ 列划掉但保留，便于对照旧文。）

⇒ **归因（2026-09-07 按新口径重写；旧版「1.6%、几乎严格为零」作废）**：

> **固定电荷截断单独已经产生约 $25\times$ 的最低模式异常；加入 CWLD 极化后仍约 $22\times$。
> 因此异常的主因明确是截断静电。CWLD 对最低模式存在至多约 10% 量级的二级影响，
> 其统计意义需由同协议的 seed 方差判定。**

旧版依据 $\kappa$ 的 41.8 vs 42.5 说「只差 1.6%、极化贡献几乎为零」——
$\kappa$ 那个量对截断组无意义（§7.4.1），而按直接可测的 $S_{ZZ}(k_{\min})$
两臂差的是 **10.5%**，**不再支持「几乎严格为零」**。主归因完全不受影响。

**（本次汇总查了 seed 方差）这 10.5% 目前判不了显著**：
0706 那三条是每臂 $n=1$、500 ps；查 5 ns 矩阵同一 config 的 3 个 seed 在
$k_{\min}$ 上的散度（`data/szz_LONG5ns_*_seed{0,1,2}.csv` 首行）：

| config | $S_{ZZ}(k_{\min})$ 三个 seed | 变异系数 |
|---|---|---|
| PME | 5.82 / 6.20 / 5.23 (e-5) | **8.5%** |
| CWLD `aq0_ca_w2` | 2.529 / 2.512 / 2.556 (e-3) | 0.7% |
| CWLD `aq0p5_ca_w2` | 2.645 / 2.253 / 2.789 (e-3) | **10.8%** |

⇒ **run-to-run 散度（0.7–10.8%）与那 10.5% 同量级**，且跨了不同协议（500 ps vs 5 ns）
⇒ **现有数据不能判定 CWLD 的二级影响是否真实存在。** 要判就得在同协议下给两臂各跑 3 seed。
这是两种可能里**较轻**的一种：属于截断静电的已知类别问题，不是 CWLD 特有缺陷。

补充观察：截断类轨迹的 $\kappa$ **随模拟长度下降**（500 ps 约 42 → 5 ns 约 30），
而 PME 不随长度变 ⇒ 该假象**随模拟时长增长**，
与「长波长电荷涨落缓慢发展、截断无法抑制」的图像一致。

**为什么值得追**：ABFE 的 decharging 自由能对长波长/有限尺寸静电**极其敏感**，
而 ABFE 兼容性正是这个项目的目标之一。修的方向见 §4.7（**不是换窗**）。

#### 7.4.1（本次汇总新增）$\kappa\approx30$ 不是屏蔽长度，17× 与 44× 的矛盾由此解释

用户指出 §4 与 §7.4 互不自洽（$(\kappa_{\rm PME}/\kappa_{\rm cut})^2\approx17.6$
是有限常数，而实测 44×）。查 `data/szz_*.csv` 的原始低 k 表与
`kill_switch.py::make_szz_from_table` 的拟合代码，**矛盾的来源查清了**。

**(a) $\kappa$ 是怎么来的**：代码**强行**把 $S_{ZZ}=k^2/\kappa^2$ 套在**最低 4 个**
实测点上，取 $\langle S_{ZZ}/k^2\rangle$ 再开方取倒。它本来只是为了给 $k<k_{\min}$
（盒子给不出的区域）提供一个 SL 外推，**不是**独立测出来的屏蔽长度。

**(b) 对 PME 这个形式成立，对截断组不成立**（`szz_LONG5ns_*_seed0.csv` 前 7 行）：

| $k$ (nm⁻¹) | 0.924 | 1.307 | 1.707 | 2.166 | 2.798 | 3.318 | 3.847 |
|---|---|---|---|---|---|---|---|
| PME $S_{ZZ}$ | 5.82e-5 | 1.20e-4 | 1.69e-4 | 2.39e-4 | 3.75e-4 | 5.21e-4 | 7.43e-4 |
| CWLD $S_{ZZ}$ | **2.65e-3** | 1.49e-3 | 8.57e-4 | 6.16e-4 | 6.86e-4 | 7.47e-4 | 8.96e-4 |
| 比值 | **45.5** | 11.4 | 5.1 | 2.6 | 1.8 | 1.4 | 1.2 |

* **PME 单调上升**，且 $S_{ZZ}/k^2$ 在前几点一致（6.81e-5 / 7.01e-5 …）
  ⇒ 与 $S_{ZZ}\propto k^2$（Stillinger–Lovett）相符，$\kappa\approx120\text{–}126$ **有意义**。
* **CWLD 单调下降到 $k\approx2.2$ 的一个极小，再回升去与 PME 汇合**
  ⇒ 这是**非单调**的，而 $k^2/(k^2+\kappa^2)$ 单调上升，**根本表示不出这个形状**。
  逐点反解出的「$\kappa$」是 **18.0 / 33.9 / 58.3 / 87.2 nm⁻¹**——四个值互相矛盾，
  跨了 4.8 倍。报告里那个 30.2 只是它们均值的开方倒数
  （$\langle S_{ZZ}/k^2\rangle=1.097\times10^{-3}\Rightarrow\kappa=30.2$，与报的 29.6–31.8 一致）。

⇒ **截断组的 $\kappa$ 是「把一个模型形式强加在不服从它的数据上」得到的数，不是物理量。**
17× 与 44× 从来不需要一致：**44× 是最低模式上的直接测量，17× 建立在不成立的形式上。**

**(c) 而 $k=0.924$ 恰好就是盒子的最低非零模式**：$2\pi/0.9234=6.805$ nm，
且该点 `n_samples=1500` = 250 帧 × **6** 个 k 矢量（即 $(\pm1,0,0)$ 那一壳）。
⇒ 44× 是**单个最低模式**上的测量，有限尺寸效应在这里最重；
而 $k<k_{\min}$ 的那一段 `szz.py` 自己就标着「**纯外推**」。

**(d) 因此现在能说和不能说的**：

| 能说 | 不能说 |
|---|---|
| 在 $k_{\min}=0.92$ nm⁻¹ 这个最低模式上，截断轨迹的 $S_{ZZ}$ 比 PME 大 **22–45 倍**（500 ps 24.7×，5 ns 45.5×） | 「$S_{ZZ}$ 比值发散」 |
| 截断轨迹的低 k $S_{ZZ}$ **非单调**、在 $k\to k_{\min}$ 时**上升**，与 SL 的 $\propto k^2$ 定性不符 | 「截断体系的 $\kappa\approx30$」 |
| **实测 crossover 位于 $2\pi/r_c$ 附近**；在当前采样的 $k>2\pi/r_c$ modes 上与 PME 吻合到 1–3% | 「分界**精确**落在 $2\pi/r_c$」——紧支撑 kernel 在 Fourier 空间**没有 hard switch**，数据只支持"附近" |
| 归因（截断而非 CWLD 极化）仍成立——它来自三方判别实验，不依赖 $\kappa$ | 从 $\kappa$ 推任何东西 |

**待做（不跑 MD，读现有 csv 即可）**：① 直接报 $S_{ZZ}(k_{\min})$ 与低 k 形状，
不再报截断组的 $\kappa$；② 把 $S_{ZZ}/k^2$ 随 k 的走势画出来，
看截断组有没有进入渐近区（现有数据大概率没有）；
③ `make_szz_from_table` 应在低 k 不服从 $k^2$ 时**报警而不是静默拟合**
（这正是 §8 第 10 条那类静默失败）。

### 7.5 数据完整性：三类事故与已建的防线

| 事故 | 机制 | 现象 | 防线 |
|---|---|---|---|
| 并发写同名文件 | `tag` 没带 `L_IPS_LIGAND_DPOLAR` | **帧交错**；头里仍写 2500、实际 3379/4258 | tag 带全部影响物理的开关；产出已存在**拒跑**（`--force` 才覆盖） |
| 对正在被写的文件 `mv` | 同文件系统只改目录项、**inode 不变，写句柄跟着走** | 2500 帧被换成 156 帧，**头和大小自洽** | 每条轨迹产出时**自带拓扑快照**；分析侧硬校验 |
| 插件成默认后同名 | 0.5 ns 计时跑与 5 ns 生产跑同名 | `--force` 把 984 MB 覆盖成 61 MB，**名字仍正常** | tag 带引擎（`_plugin`） |

`dcd_integrity.py` 两道校验**缺一不可**：
**自洽性**（头声称帧数 vs 文件大小推算）抓交错；
**期望帧数**抓截断——**截断文件是自洽的，自洽性抓不到**。
⚠ 两种情况 **mdtraj 都只打 warning、`md.load()` 照常返回数据**。
`flag_frame_count_outliers()` 是同批比较的启发式，**只报告不拦截**。

另有一次拓扑丢失：重建体系把 `1AAY/amber/solvated.pdb` 从 32818 覆盖成 32794，
8 条老轨迹当场无法分析，靠 `1AAY/amoeba/solvated.pdb` 恰好是那份旧溶剂化的副本才救回。
现已改为**按 dcd 头里的原子数自动选拓扑** + 产出自带快照。

已隔离的坏数据在 `results_cwld_corrupt/` 和 `salvage/`，**不要让它们进任何分析**。

---

## 8. 方法论教训

这个项目里**被判据坑掉的次数多于被 bug 坑掉的次数**。以下每条都对应一次真实返工，
值得当作后续工作的检查表。

| # | 教训 | 出处 |
|---|---|---|
| 1 | **标准误小 ≠ 结论可信。** 直线拟合一条**弯曲**的 $E(t)$ 照样给出 1.2% 的标准误。 | NVE 首轮 |
| 2 | **每个探针必须只改一件事**，且能用「把它加回去」直接验证。用代数从多变量探针里解答案 ⇒ 两版性能模型全被推翻。 | LCWLD-161 |
| 3 | **平均场判据在细粒度上会翻符号。** $E_F$ 用二点量 $S_{ZZ}$ 去预测需要四点关联的 RMS 力误差。 | Stage 1 §11.3 |
| 4 | **通过的证伪测试可能是假阳性。** 「4/4 PASS」之所以 PASS，正是因为判据里那个 bug。 | Stage 1 §8.1 |
| 5 | **撤回也可能是错的。** 曾以「与已发表方法冲突」为由撤回 zmm1 更准的观察，直接测量证明观察成立、冲突另有解释。 | Stage 1 §11.4 |
| 6 | **先定阈值再写 kernel。** 否则阈值一定会被拟合到跑出来的偏差上。 | DEC-005 |
| 7 | **相消 6 个量级的地方，相对误差不是有意义的判据。** | Stage 1 §1 |
| 8 | **看代码推机制，四次都错。** 性能归因必须实测。 | LCWLD 全程 |
| 9 | **静默失败是最贵的失败。** 本项目至今所有重大返工（Zn 不是密度源、残基名塌回、锌被归进 protein_heavy、引擎混用、轨迹截断）**无一例外**都是「不报错、数字看着正常」。⇒ 加断言优先于加功能。 | 全局 |
| 10 | **把模型形式强加在不服从它的数据上，得到的参数不是物理量。** 截断组的 $S_{ZZ}$ 在低 k 非单调，而 $k^2/(k^2+\kappa^2)$ 单调；硬拟出来的 $\kappa\approx30$ 被当成屏蔽长度用了，逐点反解其实跨 4.8 倍。**拟合前先看形状。** | §7.4.1 |
| 11 | **kernel 层的结论不能直接搬到平衡态可观测量。** 「被丢掉的核在 $k\to0$ 发散」是定理；「实测 $S_{ZZ}$ 比值发散」是越界——中间隔着一整个非线性响应。同理「窗只能动 6%」与「实测差 44×」不可直接比。 | §4.2 / §4.3 |
| 12 | **一个具体实现的选择不要当成方法的性质。** `pswfz2` 的带宽 $c=2\pi$ 是本项目按 cutoff 绑的，不是 PSWF 的定义；把它写成「PSWF 优化错频段」等于替一整类方法下结论。 | §4.4 |
| 15 | **「查非线性」的判据本身可能对某类非线性失明。** `nve_drift` 用前后半斜率差查 E(t) 弯不弯，而**对称的鼓包（中间高两头低）前后半均值相同** ⇒ 判据给 0.01–0.14「通过」，分块斜率离散度却是 2.45–3.85。三次 NVE 尝试全被 dt 标度判死，而首轮的根因（`minimize` bug）**是不完整的归因** —— 修掉它之后同样的符号翻转照样复现。**判据要用「它应该抓到的失败」去验证**（同 #14 第二条）；比较两个量之前先确认它们算在**同一个窗口**上（脚本丢前 20%，traces 存全程，我第一次就比错了）。 | 2026-09-07 |
| 14 | **subprocess 闸门必须消费被包裹工具写的每一条流，不只是承载预期输出的那条。** `tests/test_lint.py::_pyflakes()` 只读 `proc.stdout`，而 **pyflakes 把语法错误写到 stderr** ⇒ 一个**根本不能 parse 的文件与干净文件完全不可区分**，`dens_parity.py`（P0 脚本）因此从写下来那天起就是坏的、没人知道。<br>**⚠ 我第一版把机制诊断错了**：我说「语法错误那行进了过滤器但两个 kind 都不匹配」，并据此建议只加一个 kind 白名单测试。peer **注入一个探针文件**实测：按我建议实现的 `test_no_unrecognised_kinds` 在探针上**PASS**，只有 `ast.parse` 那条抓到了。⇒ **照我的诊断修，闸门会被报告成「已修」而其实仍然瞎。**<br>第二条可迁移的教训：**验证闸门的唯一办法是注入一个它应该抓到的失败**，不是读它的代码（与 #2/#8 同源）。第三条：被包裹的工具缺席时 `_pyflakes()` 会 skip 整个模块 ⇒ 必须另有一条**不依赖该工具**的解析检查。 | peer 2026-09-07 |
| 13 | **振荡积分的混叠陷阱在本仓库已经出现两次，两次都「看起来收敛」。** ① `kill_switch` 早期用 600 点 GL 算到 $q=2\times10^4$ ⇒ 尾段占 99%、$E_F$ 几乎不随 $s$ 变；② `ClosureWindow.chat` 写死 400 点 ⇒ $q\gtrsim250$ 起纯混叠，$\int_{2\pi}^{2000}\hat c^2$ 出到 2.376（真值 2.864e-3，**830× 高**）**且对外层求积网格看起来收敛**。判据：$n$ 点 GL 只撑到 $q\sim2n$，**节点数必须随 $\max q$ 走**。<br>**真正的病灶形状（peer 的更锐版本）不是「节点太少」，而是「节点数写成一个字面量、紧挨着它要覆盖的 $q$ 区间」**——两次事故都是这个形状，所以修法是让节点数**从 $q$ 区间导出**，而不是把字面量调大。<br>⚠ **反向别过度套用**：该不变量只管**振荡**被积函数。`closure.py::self_check` 的 `leggauss(600)` 积的是 $\chi$ 与 $\chi x^{2m}$，无振荡 ⇒ 字面量是对的，peer 已在那里留注释说明「不要把这个数抄进振荡积分」。 | peer 2026-09-07 |

---

## 9. 缺口与待办

### 9.0 ✅ **已修（2026-09-07 21:5x）：插件路径曾静默忽略 closure 与全部引擎参数**

> **修复**：`_swap_to_localcwld()` 现在把真实参数（`zmmOrder`/`environmentCutoff`/
> `cutoffDistance`/`rho0`/`kPolar`/`chargeDeltaClamp`/`useQPenalty`/`qPenaltyStrength`）
> **显式传给桥**，并对**插件表达不了的 closure 直接抛错**（非 ZMM 窗需要 §14.7b 的
> 系数入口）。参数设计成**必填关键字** —— 故意的：这样任何调用点都不可能"忘记"传，
> 改签名会立刻暴露全部调用处（实测确实立刻暴露了 `test_cwld_engine_switch.py` 那一处）。
>
> **实测三档全对上**：
> ```
> L_IPS_ZMM_ELL=1 -> 报 zmm1  插件实际 ZMMOrder=1  OK
> L_IPS_ZMM_ELL=2 -> 报 zmm2  插件实际 ZMMOrder=2  OK
> L_IPS_ZMM_ELL=3 -> 报 zmm3  插件实际 ZMMOrder=3  OK
> ```
>
> **回归 `tests/test_plugin_globals.py`（6 个），并做了变异测试**：把修复临时撤掉后
> **ell=1 与 ell=3 立刻变红，ell=2 照样绿** —— 因为桥硬编码的默认值恰好是 2，
> **这正是这个 bug 当初隐形的原因**，也说明只在默认参数下验收（DEC-005 就是）
> 结构上抓不到它。全套 **45 passed**。
>
> ⚠ **仍未做**：ell=1 在插件上的**力**验收。参数到位 ≠ 力算对 ——
> DEC-005 那套 oracle（plugin vs CustomGB 逐原子力）只在 ell=2 上跑过。
> 换生产默认前应补一次 ell=1 的对照（命令见 §9.0-A）。
>
> ⚠ 既有全部生产数据不受影响：它们本来就是默认参数（ell=2）。

#### 9.0-A ✅ ell=1 的力验收已通过（2026-09-07）

`tests/test_plugin_globals.py::test_plugin_matches_customgb_forces_at_each_ell`
（两侧**都现建在同一个 ell 上**，CUDA/mixed，零 MD 步）：

| ell | 参照 CWLD 力 RMS | p99 \|dF\| | 相对 p99/RMS | 判定 |
|---|---|---|---|---|
| **1** | 179.9 | **4.74e-03** | 2.63e-05 | ✅ |
| 2 | 195.5 | 4.30e-03 | 2.20e-05 | ✅ |

⇒ **ell=1 的引擎间一致性与 ell=2 同档**（DEC-005 阈值 1.0，历史 ell=2 报 3.3e-3）
⇒ **ell=1 力验收通过，插件的 ell=1 路径可用于生产。**
（两档的 `max` 都是 1.678e+01 —— 那是 cutoff 边界翻转给约 120 个原子的地板，
与 closure 无关，所以两档相同，符合预期。）

**⚠ 做这一关时我自己踩了三个坑，都值得记：**

1. **我给的第一条命令是错的**：`L_IPS_ZMM_ELL=1 pytest tests/test_cwld_engine_switch.py`。
   那个测试读**盘上冻结的 dump**（closure 烧在 XML 的 Lepton 串里，实测是 `15.0/(8.0*rc)`
   即 ell=2），环境变量改不了它 ⇒ 变成「CustomGB ell=2 对插件 ell=1」，报 p99 |dF| ≈ **105**。
   **看着像插件坏了，其实是对照设错了。**
2. **我改那个 fixture 时把 ell 接到了环境变量上**（`zmm_ell=v26.ZMM_CLOSURE_ELL`），
   正是上面那个错的成因。已改成**从 dump 的 Lepton 串自己推断 ell**，
   于是该测试不再受环境变量影响。
3. **新验收测试的第一版是零信号假通过**：正常路径下 `assign_cwld_mts_force_groups`
   **只在 MTS 时被调用**，所有力都留在 group 0，于是 `groups={1}` 两边都取到**全零**，
   差值精确为 0、测试"通过"。已加**显式分组** + 一条**「参照力 RMS 必须 > 1」的信号断言**
   —— 零信号时它先红，而不是让 p99=0 假装成功。
   （顺带发现：`assign_cwld_mts_force_groups` 只认 `CustomGBForce`
   ⇒ **MTS + 插件** 会把 LocalCWLDForce 放进 BASE_GROUP。MTS 当前默认关闭，记为待办。）

#### 9.0-C 建议：**把 ell=1 定成「新生产的推荐设置」，而不是改代码默认值**

证据支持换 zmm1（§7.3-R / §9.3-R4 / §9.0-A）。但**「换生产默认」不必等于「改代码默认值」**，
而后者有一个具体代价：

> **现有 1AAY 全部生产数据（§7.1 主表整张）都是 ell=2 跑的，
> 而 `ZMM_CLOSURE_ELL` 的默认值 2 就是「重跑能复现它们」的保证。**
> README 约束 #2（「1CKK 相关的默认值是复现锚点，不要清理」）虽未逐字列出 `L_IPS_ZMM_ELL`，
> 但保护的是同一件事。改默认之后，任何不显式设环境变量的复跑都会**换掉物理**。

**推荐方案（代价为零，且不牺牲可复现性）**：

| | 做法 |
|---|---|
| 代码默认值 | **保持 `L_IPS_ZMM_ELL=2`** —— 复现锚点不动 |
| 新生产 | **显式写 `L_IPS_ZMM_ELL=1`**，并在 README/本报告记为推荐设置 |
| 为什么安全 | **§9.0-B 的 tag 修复把「静默」变成了「有标签」** —— 忘记设也不会混淆，产物名会写 `_zmm2_` |

⇒ **正是 tag 那个修复，让「不改默认值」成为安全选项。** 若 tag 里不记阶数，
就只能靠改默认值来保证一致，那才不得不牺牲复现锚点。

⚠ 若最终仍决定改默认值，必须同时：在 README 约束 #2 里加上 `L_IPS_ZMM_ELL`
并注明「历史数据是 ell=2，复跑旧配置须显式设 2」，否则那条约束就变成半真。

#### 9.0-B ✅ closure 阶数已进产物 tag（2026-09-07，**代码已改、未跑验证**）

`run/zn_job.py` 的 tag 构造：原来只写族名 `zmm`，ell=1 与 ell=2 的轨迹**同名** ——
正是 §7.5 那次覆盖事故的形状，而且更隐蔽（两档 closure 的力差 7%，
混进同一个统计不会报错）。

**口径（用户决定「只用于新跑」）**：

| | 约定 |
|---|---|
| 新跑 | 一律写全：`_zmm1_` / `_zmm2_` / `_zmm3_` |
| **既有产物** | **保持 `_zmm_`，不重命名** —— 重命名会打断已发布报告里几十处文件名引用 |
| 历史 `_zmm_` 的含义 | **一律指 ell=2**（当时的冻结默认），本条即该约定的记录 |
| 非 ZMM 窗 | 窗名自带参数（`pswfz2` / `pcf1_c0.25`），只把 `.` 换成 `p` |

已核：**没有任何分析代码按 `_zmm_` 字面匹配**（`szz` 的自动发现是
`{prefix}*{suffix}` 的 glob，新名字照样命中）；`nve_drift`/`force_audit` 的
`--traj` 默认值指向依然存在的旧文件，继续可用。
⚠ 但**新的 zmm2 跑会产出 `_zmm2_`，不再匹配那两个默认值** —— 那时要显式给 `--traj`。

⚠⚠ **本条改动没有跑过验证**（用户 2026-09-07 要求：本地正在跑测试，不要占用计算）。
待跑：

```bash
pytest tests/ -q                      # 期望 47 passed（改的是 tag 构造，不动物理）
# 空跑一次看文件名（不落盘，只看它打算写什么）：
L_IPS_ZMM_ELL=1 lips-run-zn --system 1aay --help    # 确认 CLI 仍可用
```

真正的确认是下一次生产跑的产物名里出现 `_zmm1_`。

---

#### 9.0-H 原始记录（bug 的现象与根因，保留）

**实测**（`L_IPS_ZMM_ELL=1`，真实 1AAY，`engine="plugin"`）：

```
L_IPS_ZMM_ELL 设的是      : 1
日志/CSV 报的 closure     : zmm1(ZDipole/RF-like, alpha=0)
插件里实际的 ZMMOrder     : 2        ← 静默不一致
```

**根因**：`v26._swap_to_localcwld()` 调
`swap_customgb_for_localcwld(xml)` **不传 `globals_`** ⇒ 桥用
`DEFAULT_GLOBALS`，其中 **`zmmOrder: 2` 是硬编码的**；而桥**从不读 CustomGBForce 的
pair energy 表达式**去反推 closure。它校验逐粒子参数、做回环自检，
**但没有任何东西校验 pair 表达式与 `zmmOrder` 一致**。

**影响面比 closure 更宽** —— `DEFAULT_GLOBALS` 把**全部**引擎参数都钉死了：

| 参数 | 桥里硬编码 | 后果 |
|---|---|---|
| `zmmOrder` | 2 | `L_IPS_ZMM_ELL=1/3` 在插件路径**无效**；`L_IPS_CLOSURE=pswfz2` 同样被丢掉 |
| `environmentCutoff` / `cutoffDistance` | 0.35 / 1.2 | `setup_cwld_lips_system(r_env=…, rc=…)` 在插件路径**无效** |
| `rho0` / `kPolar` / `chargeDeltaClamp` | 13.5 / 0.8 / 0.2 | 同上 |
| `useQPenalty` / `qPenaltyStrength` | False / 180 | `ENABLE_Q_PENALTY=True` 在插件路径**无效** |

⇒ **插件路径上任何参数扫描都被静默钉在默认值上。**

**为什么至今没炸**：所有生产跑都恰好用的是这套默认值，而 DEC-005 的验收
（plugin vs CustomGB 逐原子力）也**是在默认参数下做的**，所以它结构上抓不到这个。

**这一条直接卡住三件事**：

1. **§9.3-R4 那个「应换 zmm1」的结论无法执行** —— 设 `L_IPS_ZMM_ELL=1` 会得到
   一批**标签写 zmm1、物理是 zmm2** 的轨迹。**换默认前必须先修这个**。
2. `L_IPS_CLOSURE=pswfz2` 在插件路径上**从来没真正跑过**（若有人跑过，跑的是 ZMM ell=2）。
3. PCF on GPU（§14.7b）无论如何都要先修它。

**修法（两处，需要与 peer 协调 —— `xml_bridge.py` 在插件工程里）**：
* `_swap_to_localcwld()` 把实际参数（closure 系数或 order、r_env、rc、rho0、k_polar、
  clamp、penalty）**显式传给桥**；
* 桥**对非 ZMM 的 pair 表达式必须抛错，不许回落默认**（PCF 走 §14.7b 的系数入口）。
* **必须补一条回归**：断言「桥出来的 `ZMMOrder` == 引擎实际用的 ell」。
  这条回归就是本 bug 的探针 —— 没有它，同类错误还会再来一次（同 §8 教训 #14 第二条）。

⚠ **在修好并补上回归之前，插件路径只能用默认参数跑**（也就是 ell=2）。
既有全部生产数据不受影响：它们本来就是默认参数。

### 9.1 P0 — `_local_dens_at` 的 parity 测试（脚本已就绪，**尚未跑**）

`src/lips/analysis/dens_parity.py`：独立向量化实现（走 `query_ball_tree` 而非
逐原子 `query_ball_point`，**代码路径不同**）、跨配置断言「dens 与 dpolar 无关」
（dpolar 只进 Δq，两档下 dens 应逐位相同；若不同说明覆盖改到了不该改的东西）、
以及耗时随目标原子数 K 的交叉点测量。

⚠ **口径更正（2026-09-07）：本报告前几版说它「已写好 / 脚本已就绪，只是没跑」——那是错的。
它当时根本 import 不了。** 两个 bug，第二个被第一个盖住：

| bug | 症状 |
|---|---|
| `from lips import systems as ls` 写在 `build_meta()` 函数体里、**列 0** | `IndentationError`，模块整个 import 失败 ⇒ **从写下来那天起就没跑过** |
| `from lips import paths` 在 `build_meta()` 里 import、却在 `main()`（129/135 行）里用 | 一跑就 `NameError`；**被上面那个语法错误盖住**，修完第一个 pyflakes 立刻报出来 |

两个都已修（`paths` 提到模块级，并留注释说明为什么另外三个 import 必须留在函数内——
env 要在 `import v26` 之前设好）。`pyflakes` 干净，全套测试通过。
⇒ **现在它才是真的「已就绪、未跑」**：`python -m lips.analysis.dens_parity`（CPU 即可）。
仍**没有 console 入口**（`pyproject.toml` 归 peer session 管）。

**为什么没人发现**：lint 门看不见它，见 §8 教训 #14。

为什么排首位：这曾是 5 份实现里**唯一没验过的那份**，而 §7.1 的三级 Δq 台阶
（0.1% / 0.75% / **11.5%**）出自它；1CKK 时代所有基于 q-profile 的结论也走同一条路径。

### ✅ 9.1-R 结果（2026-09-07，已跑，PASS）

产物：`logs/dens_parity_solvated_175430.log`、`logs/dens_parity_mdframe250_seed0_175430.log`。

| 判据 | solvated 单帧 | 真实 MD 帧（seed0, frame 250 ≈ 2.5 ns） |
|---|---|---|
| `dens`：`_local_dens_at` vs 独立向量化实现 | max abs 4.4e-16，**max rel 2.1e-16** | max abs 4.4e-16，**max rel 1.7e-16** |
| `Δq` 两条路径 | max abs 3.5e-18 | max abs 6.9e-18 |
| 跨配置：dens 是否依赖 dpolar | **0.000e+00**（严格相等）✓ | **0.000e+00** ✓ |

⇒ **分析侧镜像（实现 #3）与一份走不同代码路径的独立实现在机器精度上一致**
（逐原子 `query_ball_point` vs 一次 `query_ball_tree`）。§7.1 的 Δq 数字**不是**镜像写错造成的。

⚠ **这验的是「镜像内部自洽 + 与另一份 numpy 实现一致」，不是「对生产 CustomGBForce 逐位一致」**。
后者仍受 §9.1 那两条传递性边界限制（1024 点 tabulated vs 解析式；只在 10 粒子合成体系上验过）。
**待补的一小步**：真实 1AAY 上对这 12 个配位原子跑 `_local_dens_at` vs `localcwld_fast`。

**顺带三个副产品**：
1. **HIE149:ND1 从密度侧再次被判为零贡献**：MD 帧上 dens=0.0602 vs 真配位原子 2.59–2.83
   ⇒ Δq=−0.000535 e（clamp 的 0.27%），比最弱的真配位原子小 **44 倍**。
   这与 §7.1 那条力学证据（ΔF 小 3257 倍）**走完全不同的代码路径**，互为独立确认。
2. **耗时交叉点测出来了**：K=13 逐原子快 5 倍（vec 0.19x）、K≈200 打平（1.2x）、
   K=2000 向量化快 4.4x ⇒ **交叉点在 K≈150–200**。定点探针（K=13）用逐原子是对的，
   水的 q-profile（K≈6 万）该用向量化。
3. **外层 tanh 已经不是恒等**：MD 帧上 `|raw|/clamp ∈ [0.003, 0.125]`，
   而脚本自带的判据是「<0.05 时 clamp 才近似恒等」⇒ 顶端那 0.125 处
   tanh 非线性已带来约 0.5% 的相对压缩。报 Δq 时若要与 `dpolar×t` 对比须扣掉它。

✅ **主结果不受影响**——§7.1 的 p 值直接从轨迹几何算（Zn 与配体的距离/角度），
不经过密度实现。

**传递性的两条边界（不要当成无缝）**：
* 生产 CustomGBForce 走 **1024 点 tabulated** density kernel、reference 走解析式，
  $K(r)$ 吻合到约 1e-6，但 $dK/dr$ 在 $r_{\rm env}$ 边界有**真实的 spline 伪影**
  （曾把一个测试点的力相对误差顶到 2.33e-7）。对本用途够用
  （`dens` 只用 $K$ 不用 $dK/dr$，要抓的是**百分级**结构性偏差），
  但**不能表述成「已精确验证」**。
* 第一段**只在 10 粒子合成体系上验过**。`water_ca_cluster` fixture 确实把分支跑全了
  （charge_mod 含 clamp 上界 2.5、source weight 含水 0.5 与 Ca 2.0、4 种 residue_id 触发
  同 residue 排除、dpolar 含 −0.15/0/+0.075），但**邻居表/cutoff 边界这类只在真实尺度
  出现的问题它盖不到**。⇒ 需补：真实 1AAY 上对那 12 个配位原子跑
  `_local_dens_at` vs `localcwld_fast`，同一份 metadata、同一批坐标，纯 numpy 对 numpy，秒级。

**分级报偏差**：先 `dens`，再 tanh 前的自变量，再 clamp 前后——以定位是密度层还是
tanh/clamp 层。必须在**开启** dpolar 覆盖的配置下也验一遍（11.5% 那个数是覆盖开着算的）。

### 9.2 P0 — 两个 Δq 数字无落盘产物

| 数字 | 产物 |
|---|---|
| 0.1%（Zn 修复前） | ✅ `logs/zn_deltaq_165022.log` + `zn_deltaq_profile.csv`（16126 行，09-02） |
| **0.75%**（修好 Zn 识别） | ~~❌ 磁盘上没有任何产物~~ → **2026-09-07 已复现并落盘** |
| **11.5%**（再给配位原子 dpolar=−0.15） | ~~❌ 磁盘上没有任何产物~~ → **2026-09-07 已复现并落盘** |

### ✅ 9.2-R 复现结果（2026-09-07）

在**真实 MD 帧**（`ZN_1aay_cwld_ligm0p15metal_zmm_seed0`，frame 250 ≈ 2.5 ns，32794 原子）上：

| 报告原值 | 复现值 | 判定 |
|---|---|---|
| 0.75%（覆盖关，仅修 Zn） | **0.98%**（+0.00196 e，13 原子均值） | 同量级，差 30%；单帧 vs 原先大概是多帧/多 seed |
| **11.5%（2.3e-2 e）** | **11.73%（−0.02347 e，12 个真配位原子均值）** | **基本逐位对上** ✓ |

⚠ **两个口径细节，缺一个就对不上**：
1. **必须排除 HIE149:ND1**。含它的 13 原子均值是 10.85%，只有排除后的 12 原子均值才是 11.73%。
   ⇒ 报告原值 11.5% 的口径是「12 个真配位原子」，与 §0 的口径更正一致。
2. **必须用平衡后的 MD 帧**。同一脚本在 `1AAY/amber/solvated.pdb`（未平衡的初始溶剂化结构）
   上只给 **8.01%**（12 原子 8.68%），因为配位壳还没收紧（dens 1.47–2.44 vs MD 帧 2.59–2.83）。
   **早前把这个数当"复现失败"是错的读法** —— 它是帧选错了。

⇒ **11.5% 不再是「待复现」。** 0.75% 那个仍建议按多帧/多 seed 平均复核一次再定稿。

grep 过 `logs/` 与全部 csv/json 均无——它们是当场算完直接报的。
⇒ **目前无法复现来源，引用时须标注「待复现」**；parity 测试是唯一的重建途径。

另注：探针跑的体系是 **32818** 原子，生产轨迹是 **32794**（HIS→HID/HIE 还原后），
**原子编号不可直接互换**。好消息是 `source_class_weight` 在两边一致
（09-02 的 log 第 4 行已记录 `配位原子 is_polar=[1.], dpolar=[0.012 0.015], src_w=[1.5]`）。

### 9.3-R ✅ 已跑（2026-09-07）：**fp32 判不了，且已查明原因**

产物：`closure_nve_drift_dt{2,1}fs_single_3k6f48b1.csv`（+ traces）。
3 kernel × 1 seed × (200 ps NVT → 200 ps NVE)，CUDA/single。

| kernel | dt=2 fs | dt=1 fs | 对 zmm2 归一（2 fs） |
|---|---|---|---|
| pme | +1.4454e-02 | +2.4933e-02 | 0.997 |
| zmm1 | +1.4651e-02 | +2.4946e-02 | 1.010 |
| zmm2 | +1.4501e-02 | +2.4796e-02 | 1.000 |

（kT/ns/dof。`nonlin` 0.01–0.05，线性度检验**通过**。）

**dt 标度判据：失败，但这次是物理限制不是 bug。** dt 减半后漂移**×1.71**，
而真实积分误差应 ×0.25。×1.71 接近 **×2.00**，那正是「**每步**累积」的标度
（步数 ∝ 1/dt）——**单精度舍入的指纹**。按 $A/dt + B\,dt^2$ 拆：

$$A=2.42\times10^{-2},\quad B=6.0\times10^{-4}
\;\Rightarrow\; dt{=}2\,\text{fs 时舍入项占 }83\%,\ \text{真积分误差仅 }17\%$$

**三条互相独立的证据说明测到的不是 closure**：
1. **PME 坐在同一个地板上**（0.997）——它的静电与 ZMM 截断完全不同，
   若差异来自 closure 的哈密顿量守恒性，PME 不可能和它们一样。
2. **三个 kernel 的散度只有 1.4%**，而静态力审计上的差距是 **7–17%**（§7.3-R）。
3. **绝对漂移 +2400 kJ/mol/ns** 大得离谱（良好 NVE 应低几个量级）⇒ 地板主导。

⚠ **线性度通过不构成验证**：舍入累积在时间上同样是线性的，
所以 `nonlin=0.01` 只说明"斜率拟合得好"，不说明"斜率有物理意义"（同 §8 教训 #1）。

**两个可以下的结论**：
* **作为判据**：fp32 NVE **不能**回答 zmm1 vs zmm2，必须换 `--precision mixed`
  （力单精度、累加双精度，能量守恒研究的标准做法）。**这一步要用户批准**，
  因为它擦到「GPU 一律 fp32」那条线；但现在有硬证据：fp32 的地板是 1.2e-2，
  把 7–17% 的信号整个埋掉。
* **作为对生产路径的陈述（这条本身就有决策价值）**：在 fp32 下三个 kernel 的能量守恒
  **同样差、差在 1.4% 以内**，而生产轨迹本来跑 Langevin 恒温器、漂移会被吸收
  ⇒ **「为了能量守恒要保住高 $\ell$」这个论点在我们的 fp32 生产路径上不成立。**
  ⚠ 但这不等于"zmm1 已被证明安全"——它只说明**那条反对意见在生产精度下不可测**。

### 9.3-R2 mixed 精度：地板降了 23 倍，但**窗口太短**，仍未判定（2026-09-07）

`--precision mixed` 把地板从 1.45e-2 压到 6.2e-4（**23 倍**）⇒ 换 mixed 这个决定是对的。
但 dt 标度**又一次失败，且这次是符号翻转**：

| kernel | dt=2fs | dt=1fs | 比值 |
|---|---|---|---|
| pme | +1.180e-03 | −6.469e-04 | −0.55 |
| zmm1 | +5.172e-04 | −5.842e-04 | −1.13 |
| zmm2 | +6.227e-04 | −6.533e-04 | −1.05 |

**这与首轮（`STAGE1...§12`）的模式一模一样** —— 而首轮的根因被归给了 `minimize` bug，
那个 bug 早就修了。⇒ **首轮的归因是不完整的。**

**查 traces 查出了真因，顺带查出判据本身的缺陷。**
在**脚本真正拟合的窗口**（丢掉前 20%）上把 E(t) 切四段看斜率：

| run | 脚本判据（前后半） | **分块斜率离散度** | 四段斜率 |
|---|---|---|---|
| zmm1 dt2fs | 0.02 ✓「通过」 | **2.45** | +1.2e-1 +2.3e-1 +2.1e-1 +2.4e-2 |
| zmm2 dt2fs | 0.14 ✓「通过」 | **3.85** | +1.2e-1 +2.0e-1 +4.1e-1 +1.3e-2 |
| pme dt1fs | 0.01 ✓「通过」 | **2.65** | −1.8e-1 −8.3e-2 **+4.7e-2** −2.4e-1 |
| zmm1 dt1fs | 0.09 | 0.61 | −1.1e-1 −1.1e-1 −1.1e-1 −1.6e-1 |
| zmm2 dt1fs | 0.03 | 0.40 | −9.1e-2 −9.5e-2 −1.3e-1 −8.3e-2 |

⇒ **两个结论**：

1. **`nonlinearity` 这个判据会被「对称的非单调性」骗过。** 一个中间高两头低的鼓包，
   前半均值 ≈ 后半均值，于是 $|s_2-s_1|$ 天然接近 0 —— 三个 run 报 0.01–0.14「通过」，
   而分块离散度是 2.45–3.85。**它正是为了抓「弯的 E(t)」而加的，却抓不到对称的弯。**
   **已改**：判据换成**分块斜率离散度**（4 段，极差/|整体斜率|），旧判据保留为
   `nonlinearity_half` 列以便对比，并存 `block_slopes_*` 明细。上面三例全会被 >0.5 拦下。
2. **dt=2fs 那一档整个不可用**（离散 2.45 / 3.85）⇒ **符号翻转不需要用物理解释，
   是因为 dt=2fs 点本身是垃圾。** dt=1fs 的 zmm1/zmm2（0.61 / 0.40）才接近真漂移。

**目前唯一站得住的数**：dt=1fs、mixed、zmm1 −9.67e-2 vs zmm2 −1.08e-1 kJ/mol/ps
⇒ **zmm1/zmm2 = 0.894**（与力审计的 0.926 同向）。但**三条都还缺**：
dt² 标度（2fs 点报废，得改用 1fs/0.5fs 配对）、seed 方差（1 seed 无 sem）、
可用的 PME 参照臂（它在 1fs 上离散 2.65 且跨零）。

**下一步不是再加精度，是加窗长。** 瓶颈已从舍入地板变成"非单调成分"：
漂移信号 ∝ T 而游走 ∝ √T ⇒ 加长窗口按 √T 改善信噪。
**我把 `--prod-ps` 从默认 1000 砍到 200 是砍错了地方**，那 200 ps 就是现在的瓶颈。

### 9.3-R3 ✅ 加长窗 + 补 seed：**判据首次通过，结论是「NVE 不区分」**（2026-09-07）

`--kernels zmm1,zmm2 --seeds 3 --equil-ps 200 --prod-ps 1000 --precision mixed --dt 0.001`
（6 条 run，每条 272–280 s）

| kernel | mean (kT/ns/dof) | std | sem | 对 zmm2 归一 |
|---|---|---|---|---|
| zmm1 | −6.3525e-04 | 2.87e-05 | 1.65e-05 | **0.972 ± 0.025** |
| zmm2 | −6.5340e-04 | 3.50e-05 | 2.02e-05 | 1.000 ± 0.031 |

**闸门 1（线性度）首次通过**：6 条 run 的**分块斜率离散度**全在 0.05–0.14，远低于 0.5 阈值。
⇒ **把 `--prod-ps` 从 200 加到 1000 修掉了非单调性**（200 ps 那批离散度是 2.45 / 3.85）。
这是这项测量三轮以来第一次真正过掉自己的线性度关，而且过的是**换严之后**的判据。

**闸门 3（区分度）：不区分。**
|漂移| 之差 1.815e-05，合并 sem 2.613e-05 ⇒ **0.69 σ**；
按脚本自带的判读纪律，差距 1.8e-05 < 两者 sem 之和 3.7e-05 ⇒ **噪声**。

⚠ **早前那个 0.894 作废**（200 ps × 1 seed）。加长窗 + 补 seed 之后是 **0.972**——
那 11% 的"优势"是短窗与单 seed 的产物。**这正是闸门存在的理由。**

⚠ **又一个「标准误小 ≠ 可信」的实例**（同 §8 教训 #1）：逐条 run 打印的拟合
`± 1.9e-06` 看着是 0.3% 的精度，而**真实不确定度是 seed 间 sem 1.65e-05，大 8.7 倍**。
读这张表只能看 seed sem，不能看拟合 se。

**闸门 2（dt 标度）仍未做** —— `--dt 0.0005` 那一档在跑。它决定这些数字**是不是**积分误差。
但注意：**「不区分」这个结论对闸门 2 的结果不敏感** —— 若 dt 标度通过，两者在真漂移上
相差 0.69σ；若不通过，测到的根本不是漂移，同样不构成区分。

**对决策的意义（这是本档的要点）**：

> **NVE 这个通道不支持「为了能量守恒必须保住高 $\ell$」。**

⚠ 措辞必须精确：这是**零结果**，**不是**"zmm1 在 NVE 上更好"。
它的作用是**移除那条反对意见**——而当初拦住 zmm1 的唯一理由正是它。
配合静态力审计（zmm1 在 4 个独立系综上好 7.4%，第一壳 15%），
~~换生产默认 zmm2 → zmm1 的证据链现在只差闸门 2。~~
⇒ **闸门 2 已于 20:02 通过，见 §9.3-R4：证据链闭合。** 本节保留为过程记录。

### 9.3-R4 ✅✅ **闸门 2 通过，NVE 这条线收口**（2026-09-07 20:02）

`--kernels zmm1,zmm2 --seeds 3 --equil-ps 200 --prod-ps 1000 --precision mixed --dt 0.0005`
产物 `closure_nve_drift_dt0.5fs_mixed_2kf4c906.csv`（配对：`..._dt1fs_mixed_2kf4c906.csv`）。

| kernel | dt=1fs | dt=0.5fs | 比值 | dt² 预期 |
|---|---|---|---|---|
| zmm1 | −6.3525e-04 | −1.7757e-04 | **0.28** | 0.25 |
| zmm2 | −6.5340e-04 | −1.7728e-04 | **0.27** | 0.25 |

**三关全过，这是四轮尝试以来第一次：**

1. **闸门 1（线性度）**：分块斜率离散度 max 0.29 / 0.45，均 < 0.5
   （⚠ zmm2 的 0.45 已接近阈值，若要再加长窗这一项值得盯）。
2. **闸门 2（dt 标度）**：比值 0.27–0.28 对 dt² 的 0.25，**且符号不变** ⇒ 测到的**是积分误差**。
   按 $A/dt+B\,dt^2$ 拆：$A=1.1\times10^{-5}$、$B=6.2\times10^{-4}$
   ⇒ **dt=1fs 上舍入只占 1.7%、积分误差占 98%**（single 精度那档舍入占 83%）。
   ⇒ 「mixed + 1000 ps」这两个改动各修掉了一半问题：精度降地板、窗长去非单调。
3. **闸门 3（区分度）**：两个 dt 都说**不区分** ——
   dt=1fs 给 0.972 ± 0.025（0.69σ），dt=0.5fs 给 **1.002**（差 −3.0e-07、合并 sem 8.1e-06、**0.04σ**）。

⇒ **结论从「不可测」升级为「已测且为零」**（这比 §9.3-R3 写的强）：

> **zmm1 与 zmm2 的 NVE 能量守恒在已验证的测量下无法区分（0.04–0.69σ）。**
> 高 $\ell$ 的存在理由（cutoff 处更高阶光滑 ⇒ 能量守恒）**在本体系、本精度下没有兑现**。

**⇒ 换生产默认 zmm2 → zmm1 的证据链闭合：**

| 通道 | 结论 |
|---|---|
| 局域力精度 | zmm1 好 **7.4%**（总体）/ **15%**（第一壳）；**4 个独立系综一致** |
| NVE 能量守恒 | **不区分**（已验证 dt² 标度，0.04–0.69σ）⇒ 反对意见不成立 |
| 复现性 | ZMM 三档冻结串未动；`L_IPS_ZMM_ELL=1` 即可切，无需改代码 |

**剩下的是决策，不是测量。** ~~若要换，按约束 #6 必须让产物名带上 $\ell$~~
⇒ **该前提已于当日关闭（§9.0-B）**：`run/zn_job.py` 的 tag 现在写 `_zmm{ell}_`，
新跑会产出 `_zmm1_` / `_zmm2_`；既有产物保持 `_zmm_`（历史上一律指 ell=2）。
换默认的**推荐做法**见 §9.0-C（保留代码默认值 2，新生产显式写 1）。

### 9.3 P1 — NVE 漂移重跑（原文，已被 9.3-R / R2 / R3 / R4 取代）

脚本已修（§7.3）。这是**决定要不要把生产默认从 zmm2 换成 zmm1 的唯一缺口**：
静态审计说 zmm1 更准 2–21%，但它**量不到能量守恒通道**，而那正是高 $\ell$ 的理由。
必须带 `--dt 0.001` 对照，且**必须 `--platform CUDA`**——
「同组 cutoff 必须一致」那条约束在 Reference 平台上不报错。

### 9.4 P1 — 可复现性基础设施

* **仓库当前不在有效版本控制下**：`/home/ruigengji/L-IPS/.git` 最后一次提交是 2026-05-26、
  只跟踪 11 个 demo 阶段文件；且 `/home/ruigengji` 是 **NFSv3，该导出拒绝写 mode 0444 的
  文件**，而 git 的松散对象正是 0444 建的 ⇒ **这棵树里任何 git 写操作都会失败**；
  叠加仓库目录属主不是当前用户（dubious ownership），git 在此基本不可用。
  `archive/` 里那些 `.bak-*` 正是这件事的代偿。
* **轨迹缺 provenance sidecar**：`cwld_engine` 补上了引擎这一维，
  但「某批数据跑在哪个 commit 的代码上」仍然只能靠翻 `logs/`。

### 9.5 P1 —（本次汇总新增）三件未结论化的事

### ✅ 9.5-R 结果（2026-09-07，`python -m lips.analysis.charge_budget`）

产物 `charge_budget_{total,water_split}_ZN_1aay_cwld_ligm0p15metal_zmm_seed0_f5_lig-0.15.csv`
（**5 个平衡 MD 帧**跨全轨迹，32794 原子、31588 个 sink、10436 个水，**不抽样**）。

**Q1 总净电荷：$\sum_i\Delta q_i=+8.26\ldots+8.31$ e**（5 帧，散度 0.6%）。
（solvated 未平衡帧给 +7.88 e；与「每水 +8.1e-4 e × 10436 水」自洽。）
⇒ **H1 是真的阻塞项，不是理论顾虑。** 倒空间求和丢 $\mathbf k=0$ 等价于中性背景，
而体系实际带 **+7.9 e**。路线 B 开工前必须处理（或在 $Q$ 上加分子级中性约束
—— 注意约束本身也进 $\lambda_i$，即 H2 的又一个通道）。

**Q2 归因定了，而且推翻了我的推导：一阶主导，三阶可忽略。**

| 帧 | n | exact | linear（一阶） | nonlinear（三阶+） | linear_share |
|---|---|---|---|---|---|
| 0 / 624 / 1249 / 1874 / 2499 | 10436 | 8.13–8.18e-04 | 同左 | **5.7–6.3e-08** | **0.9999** |

⇒ **一阶 dens mismatch 解释了 99.99%**（5 帧一致）；我推的「外层 tanh 三阶曲率
$-a^3/4$」只占 **~7e-5** 的份额 ⇒ **是配角，不是机制**。
用户当时坚持把它从「验证了三阶机制」降级为「候选机制、尚未归因」是对的 ——
若照我原来的写法，报告里会留下一个**符号和量级都对、但归因错**的解释。
（这条进 §8：**符号与量级同时对上，仍然不构成归因** —— 竞争机制可能给出同样的符号与量级。）

真因因此是明确的：**生产引擎里 O 与两个 H 各自按自己位置算 dens**，
所以 $d_Ot_O+d_H(t_{H_1}+t_{H_2})\neq0$，设计意图（$d_H=-d_O/2$）在一阶就破了。
**近金属那一组在 1AAY 上不存在，而且这条本身是主结果的独立确认。**
实测最近的水到 Zn 是 **0.404–0.449 nm**（5 帧），`n_water_near_metal` 全为 **0**
⇒ Zn 第一壳**整条轨迹都没有水**，正是 Cys₂His₂ 四配位的另一面 ——
**从水的一侧测，而不是从配体距离/角度一侧测**，与 §7.1 的 100% 核心完整互相独立。
⇒ 1CKK 时代那个「近 Ca 水的净 Δq 反号」（−1.3e-3…−6.6e-3）**在 1AAY 上无法复核**，
要查得跑 1CKK 轨迹（`L_IPS_DATA_DIR=/home/ruigengji/L-IPS/2.4`）。属低优先级。

⚠ **写这个脚本时我又犯了今天刚加的那条规则**：第一版把产物存在性检查放在**末尾**、
产物名也**不带区分维度**，于是换轨迹/帧数第二次跑时在**算完之后**才被守卫拦下。
已改成开跑前查 + 名字带 `{轨迹}_f{帧数}_lig{dpolar}`。
⇒ **规则写进文档不等于自己会遵守**（同 §8 教训 #14 第二条：闸门要用它该抓的失败去验证）。

---

**1. 水的净电荷：~~先分解，再谈机制，最后才谈总量~~ —— 已完成，见 §9.5-R。**
`water_net_delta_Q_mean` = +8.0e-4 e 是实测且系统性同号的（§3.4.3），
但**归因未定**：一阶 dens mismatch 与外层 tanh 的三阶曲率都可能是主因，
而近 Ca 反号更像一阶主导。三步，全部只读现有轨迹、不跑 MD：

* **① 分解**（本项的关键）：逐水算
  $\Delta Q_{\rm linear}=d_Ot_O+d_Ht_{H_1}+d_Ht_{H_2}$ 与
  $\Delta Q_{\rm nonlinear}=\Delta Q_{\rm exact}-\Delta Q_{\rm linear}$，
  按「体相 / 近金属」分组报。一次就能定 +8e-4 e 出自哪一项。
* **② 总量**：直接对生产轨迹求 $\sum_i\Delta q_i$（**不抽样**，现有诊断是 500 三联体/帧
  的抽样，不能用来推总量），确认是否到 e 量级、是否随时间漂。
* **③** 若确实到 e 量级，再讨论要不要在 $Q$ 上加分子级中性约束
  （注意那会改动力：约束进 $\lambda_i$）。
* ✅ 已知它**不是** §7.4 的原因（固定电荷臂 $\Delta q\equiv0$，屏蔽不足照样存在）。

**2. `enable_solute_polarization=False` 时溶质退出密度源：先定语义，不要先改日志。**
（2026-09-07 修正了本报告第一版的处置建议。）第一版写的是「先改日志（一行）」，
**这个处置很可能是反的**：模型明确把 `dens_source` 与 `dens_sink` 定义成两套独立概念
（§5.2；水甚至可以「不响应但仍是 source」），而这个 flag 名叫 *polarization*、
日志也说「溶质仍是 density source，只是不响应」。
⇒ **从设计语义看，更像是代码把 source 一起错误关掉了，而不是日志写错。**

正确顺序：**先决定 `Water_only` 的 intended semantics**，再动代码或日志。

| 若本意是 | 则改 |
|---|---|
| 「只有水响应，但水仍感受蛋白环境」 | **改 metadata 代码**：把 `dens_source`/`source_class_weight` 的赋值移出 `if enable_solute_polarization` |
| 「蛋白完全退出 CWLD 环境」 | 改日志，**并且把 flag 改名**（`enable_solute_polarization` 名不副实） |

定完再核对 1CKK 时代基于 `Water_only` 臂的结论是否受影响。

**并且钉一个很便宜的回归不变量**（防止「flag 名 / 日志 / 代码三者各讲一种语义」再发生）。
若最终认定「`enable_solute_polarization` 只控制 response、不控制 source」是设计语义，
测试直接断言：对所有溶质原子

$$
\texttt{dens\_source}^{\rm on}=\texttt{dens\_source}^{\rm off},\qquad
\texttt{source\_class\_weight}^{\rm on}=\texttt{source\_class\_weight}^{\rm off}
$$

而 `dens_sink` / `static_phase` / `is_polar` **必须**按预期改变（on 时非零、off 时归零）。
两个 metadata 调用 + 几行断言，不跑 MD。

**3. 截断组的 $\kappa$ 与低 k 形状**：见 §7.4.1(d) 的三条待做。
其中「`make_szz_from_table` 在低 k 不服从 $k^2$ 时应报警而不是静默拟合」是代码改动，
其余两条只是换个报法。

### 9.6 P2 — 科学侧补强

| 项 | 说明 |
|---|---|
| ~~1CKK 负对照重新核对~~ | ✅ **已核，零机时（2026-09-07）—— 答案一直躺在 2.4 的 csv 里，见 §9.6-R** |
| AMOEBA 补 seed | 1 ns × 3（约 12 h）。只在需要声称 arm 间排序时才做 |
| 平衡协议 | 保存平衡段轨迹，查清哪一步在拽开位点 |
| 幅度扫 −0.11 | 卡准 0.08 附近的转折点，优先级低 |
| 水盒子 CWLD 重跑 | 现有 `ZN_water_cwld_*` 三条**名不副实**（跑在 Zn 修复前，CWLD 极化实际关着，约等于「PME + ZMM 截断」）；结论（三方都给 CN=6 八面体、无区分度）不变但标签错。很便宜（PME 1752 ns/day） |
| 长波长效应对 ABFE 的实际影响 | §4.6 的前提 |
| 插件 metadata 同步 | `openmm-localcwld/metadata.py` 是 v2.6 metadata 的独立实现，per-atom dpolar 规则改过需同步（**函数形式未变**，Lepton 串与解析导数不受影响） |

### ✅ 9.6-R 1CKK 负对照：CWLD 没有把 PME 本来就对的体系搞坏（2026-09-07，**零 MD**）

**这是审稿人必问的那一条**（「CWLD 是不是把**所有**东西都抓紧了？」），
而它此前被记成「换体系后从未复核」的待办。实际上答案一直在
`2.4/1ckk_v26_long_convergence_5ns_per_seed_metrics_QFIX.csv` 里 —— 只需要读，不需要跑。

Ca 总配位氧数（0.30 nm 判据，5 ns × 3 seed，逐 seed）：

| config | mean | sem | Δ vs PME | MWU p |
|---|---|---|---|---|
| **PME**（已知与实验相符，§1 引 7.66） | **7.659** | 0.046 | — | — |
| CWLD `aq0_ca_w1` | 7.584 | 0.063 | −0.075 | 0.700 |
| CWLD `aq0_ca_w2` | 7.486 | 0.028 | −0.172 | 0.100 |
| CWLD `aq0p5_ca_w2` | 7.519 | 0.036 | −0.139 | 0.100 |
| **全部 CWLD 合并（n=9）** | **7.530** | 0.027 | **−0.129** | **0.064** |

**判读**：CWLD 各臂 7.49–7.58，比 PME 低 **0.08–0.17（1–2%）**，
逐臂 p=0.10–0.70、合并 p=0.064 ⇒ **没有统计显著的退化**，且全部落在
Ca²⁺ 配位数的实验区间内。

⇒ **负对照成立：在一个「固定电荷本来就对」的体系上，CWLD 不破坏正确答案。**
配合 1AAY 的主结果（CWLD 救回 PME 救不了的位点，p=0.005），
两条一起说明**增益来自局部响应，而不是全局过度收紧**。

**⚠ 两条口径**：
1. **这一条不受 §MATH_CHANGE_MAP 那个 closure 不一致的影响。** 那三个臂里
   `aq0_ca_w1`/`aq0_ca_w2` 是 ell=1、`aq0p5_ca_w2` 是 ell=2 —— 该不一致污染的是
   **参数选择**结论（「哪个 a_q2/ca_w 更好」），而负对照问的是
   「**有没有任何** CWLD 配置把 Ca 配位搞坏」。三个配置**跨两个 closure 阶数**全都没坏
   ⇒ 结论反而**更强**（对 ell 不敏感）。
2. 合并的 p=0.064 是「接近但未达 0.05」。**方向是 CWLD 略低**，
   所以诚实的说法是「未检出退化」而不是「证明无差别」 ——
   若要把它讲成「无差别」，需要更多 seed 或一个等价性检验（TOST），不是现在这个检验。

### 9.7 已知技术债

* `engine/v26.py` 是近 2900 行单体（closure 数学 / metadata / 力场装配 / MD 驱动 /
  分析 / 实验矩阵调度）。**拆分暂缓**：同一套数学有多份独立实现，
  动它有让它们静默失同步的风险。**先做 parity，再谈拆分。**
* `archive/` `salvage/` `results_cwld_corrupt/` `results_pswfz2_partial/` 四个目录
  没清理过，也没查过还有没有代码在引用它们。
* 闸门诊断的判据在 C++ 与 `check_gate_diagnosis.py` 里各有一份，**改阈值要改两处**。
* `DEC-002` 状态是「证据已收集、结论待签署」。

---

## 10. 引用纪律：这些数字现在不能用

| 数字 / 说法 | 为什么 | 该用什么 |
|---|---|---|
| 3.391x（插件端到端中位） | 闸门 NO VERDICT（各臂漂速不同） | **3.1–3.6x**，保守下界 3.075x |
| 48.4 / 15.9 GPU-h | 48.4 是硬编码 2080 Ti 基线 | 只引加速比，不引小时数 |
| ~~Δq = 0.75% / 11.5%~~ | ~~无落盘产物，来自未验证的镜像实现~~ | **已解除**（2026-09-07）：镜像已验（§9.1-R），11.5% 已复现为 **11.73%**（12 真配位原子、MD 帧），0.75% 复现为 0.98%（§9.2-R） |
| **Δq 的百分数不带口径** | 同一配置下 13 原子均值 10.85% / 12 原子均值 11.73% / solvated 帧 8.01%，三个都"对" | 引用时必须写明**哪些原子**与**哪一帧** |
| 2.4 的 5 ns 矩阵参数选择结论 | closure 不一致（07-11 是 ell=1，07-14 是 ell=2） | **不要再引用** |
| 首轮 NVE 漂移的 kernel 排序 | minimize bug，被 dt 标度判死 | 重跑后再说 |
| Stage 1 §2 的 $E_F$ 数值表 | 建立在有系统性缺陷的解析 $S_{ZZ}$ 模型上 | 降级为暂定值 |
| **截断组的 $\kappa\approx30$ nm⁻¹** | 低 k 非单调，$k^2/\kappa^2$ 表示不出来；逐点反解跨 18–87 | 直接报 $S_{ZZ}(k_{\min})$ 与形状（§7.4.1） |
| **「低 k 的 $S_{ZZ}$ 差 ~17 倍」** | 由上一条的 $\kappa$ 推出，形式不成立 | 实测比值 22–45×（依长度） |
| **「CWLD 极化只贡献 1.6%、几乎为零」** | 由 $\kappa$ 41.8 vs 42.5 推出；按 $S_{ZZ}(k_{\min})$ 是 10.5% | 「主因是截断；CWLD 至多约 10% 量级的二级影响，显著性待判」（§7.4） |
| **「低 k 外推占比 17.6%」** | 名字暗示那是实测区 | 「$k<k_{\min}$ 的强加 $k^2$ 延拓对 surrogate 积分的贡献占比」 |
| **「偶性 ⇒ $\hat c$ 按 $q^{-3}$ 衰减」** | $q^{-3}$ 只是 $\ell=2$ 那一档 | $\hat c_\ell=O(q^{-(\ell+1)})$，阶由端点零点阶定（§3.6） |
| **「$S_{ZZ}$ 比值发散」** | kernel 极点 ⇏ 平衡态比值发散 | 「无法恢复 Coulomb 极点与 SL 渐近系数」（§4.1） |
| **全类天花板 1.39×** | 由已被推翻的 $E_F$ surrogate 得出 | 「$E_F$ surrogate + 给定 $S_{ZZ}$ 模型 + 该约束类下的下界」（§4.5） |
| **「窗只能动 6%」** | 只是 zmm1/2/3 在 $q\approx1.1$ 挨得近；$q\approx2.6$ 处落差 45% | 当经验佐证，不当 bound（§4.3） |
| **「PSWF 优化 $q>2\pi$」** | $2\pi$ 是本实现绑到 cutoff 的选择，不是 PSWF 的性质 | 「当前 pswfz2 的 objective 与目标错位」（§4.4） |
| **$\hat c$ 的 0.877 / 0.912 / 0.932 与 0.32 / 0.52 / 0.62** | 两阶展开 + 相消误差 | 闭式值（§3.6 / §4.3） |
| **「pswfz2 只需 zmm2 的 1/4.8 模式数」** | 我 2026-09-07 的量级估算，peer 实测**不成立**（节省不单调，最严目标上 0.68×） | **残差比 3.8×**，且只在 $k_c\approx2\pi/r_c$ 附近（§4.7） |
| **「窗的优劣与 $k_c$ 无关」** | 从未这么写过，但 §13.2 的干净分解容易被这样读 | 优劣与 $k_c$ **耦合**：$q\approx8$ 起 zmm3 反超、$q\approx13$ 起 zmm2 反超 pswfz2 |
| **B0 合成体系与生产力审计的「量级也对上」** | 我 2026-09-07 的宽松说法；合成体系系统性压缩 9–13%、单方向 | **只引排序**（zmm1 < zmm2 < pswfz2 < zmm3）（§4.7） |
| **$\hat c_\ell=(2\ell+1)!!j_\ell/q^\ell$ 用于非 ZMM 窗** | 该闭式是 ZMM 专属 | 一般结论是 $\hat c=O(q^{-(\ell+1)})$，由端点零点阶给出（§3.6(i)） |
| Stage 1 §5.1 撤回 / §8.1「证伪 4/4 PASS」 | 撤回本身是错的；PASS 是判据 bug 导致的假阳性 | 以静态力审计为准 |
| 「13 个金属配位原子」 | 实为 **12 配位 + 1 远端 HIE149** | 「13 个化学判据命中 = 12 配位 + 1 远端」 |
| `ZN_water_cwld_*` 的标签 | 跑在 Zn 修复前，CWLD 极化实际关着 | 结论不变，标签错 |
| 「AMOEBA 优于/劣于 CWLD」 | AMOEBA 只有 1 ns × 1 seed，无 seed 间散度 | 只当锚点 |
| 「CWLD vs PME 是干净对照」 | 跨了两套溶剂化（32818 vs 32794） | 等价于换了水的随机种子；**CWLD arm 之间才是干净的** |

---

## 11. 环境

```
/home/ruigengji/miniforge3/envs/openmm_dev
  Python 3.12   OpenMM 8.5.2.dev-36a30cb（CUDA 12 构建，NVRTC 12.9）
                git_revision 36a30cbca54e727b216b606f3c011b67201eb8b4
  numpy 2.4  scipy 1.17  pandas  mdtraj 1.11  torch 2.12(+cu12.9, 可选)
  SWIG 4.5.0 / pybind11 3.0.1 / Cython 3.3.0（插件 Python 绑定用）
  GPU: RTX 2080 Ti（主，驱动 580.178.04）/ RTX 3090（yayoi27，独立构建树 build-cuda-3090）
```

⚠ **OpenMM 版本是钉死的，不要「升级修正」。** mamba 渠道上那个 8.6.0 是
**alpha 预发布、没有 GPU 支持**，生产上用不了。版本以解释器为准
（`openmm.version.full_version`），不是文档。换版本会让插件 kernel **静默算错力**（§7.2）。

测试现状：`pytest tests/ -q` **36 passed**（2026-09-07：原 21 + PCF 13 + lint 门新增 2）；
`openmm-localcwld` ctest **7/7**，Python **35 passed**。

---

## 12. 复现路径

```bash
mamba activate openmm_dev
pip install -e .            # 已装则跳过
lips-paths                  # 确认路径解析
pytest tests/ -q            # 21 passed

# 插件（每台机器一棵构建树）
cd openmm-localcwld && cmake --build build-cuda -j && (cd build-cuda && ctest)
```

**跑主结果那一档**（会真的跑 MD，5 ns × 3 seed）：

```bash
L_IPS_LIGAND_DPOLAR=-0.15 L_IPS_LIGAND_SCOPE=metal \
L_IPS_RUN_TAG=ligm0p15metal_zmm_plugin \
lips-run-zn --system 1aay
```

**只想验证不跑 MD 的三件事**（都是秒级到分钟级）：

```bash
python -m lips.engine.closure          # 窗层自检：ZMM 三档必须全 PASS 且逐位一致
python -m lips.analysis.dens_parity    # P0，已 PASS（§9.1-R）；加 --dcd/--frame 用真实 MD 帧
lips-force-audit --help                # 静态力审计，零 Integrator.step()
```

**复算 1CKK 旧数据**：`L_IPS_DATA_DIR=/home/ruigengji/L-IPS/2.4 lips-block-time`
（只影响**读**输入轨迹；输出仍写到 `RESULTS_DIR`，不会污染 2.4）。
