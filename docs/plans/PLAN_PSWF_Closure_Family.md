# 窗函数 closure 家族 —— 总体设计（PSWF 路线）

> 2026-08-31 建，2026-09-01 更新。
> 状态（2026-09-01 最新）：**Stage 0 已完成并接入生产；Stage 1 判据已被静态力审计推翻。**
> pswfz2 在力精度上每个通道都比 zmm2 差 3–15%，无收益；唯一未测通道是 NVE 能量漂移
> （`closure_nve_drift.py` 已就绪待跑）。详见 `STAGE1_CLOSURE_KILLSWITCH_REPORT.md` §11。
> 历史状态：**Stage 1 kill switch 已执行但未判完（阻塞于实测 S_ZZ）。见
> [`STAGE1_CLOSURE_KILLSWITCH_REPORT.md`](STAGE1_CLOSURE_KILLSWITCH_REPORT.md)。**
> 生产脚本仍一字未动；`closure_windows.py`（窗层）与
> `stage1_closure_kill_switch.py`（判定脚本）已落地并自验。
>
> **2026-09-07 路线重定级（先读 §11）**：原目标「靠换窗恢复 Coulomb 长波物理」**已判死**；
> 另见 **§14 Prolate Closure Family (PCF)**：$\chi_{\ell,c}=(1-x^2)^\ell\psi_0^c/\!\int$，
> 把 **ZMM 变成 $c\to0$ 的退化极限**而不是被 PSWF 打补丁的对象；
> $\ell$ 定谱衰减指数 $q^{-(\ell+1)}$、$c$ 定带内形状，**参数职责分离**。
> 它的卖点是「a spectrally tunable, smooth, compact real-space closure」，
> **不背长程静电的包袱、也不宣称最优**（约束最优仍是 `pswfz2`）。
> closure-family 降级为 **real-space kernel designer + hybrid 方法的前半截**，
> 并新增两条活路线：**§12 有限 $k$ 的 closure 优化**（便宜，先做）与
> **§13 local closure + 少量 low-$k$ reciprocal 修正**（最值得做）。
> §13.1 推出一个关键结果：reciprocal 修正的权重**不是自由参数**，
> 就是窗自己的余弦变换 $w(k)=\hat c(kr_c)$；由此带内静电变成精确 Coulomb 且与窗无关，
> **窗只剩「局域力精度」与「带外残差」两个作用** —— PSWF 的目标在 hybrid 下才是对的。
>
> **2026-09-07 追加：B0 与 C0–C2 已实测**，结果在
> [`../reports/REPORT_HYBRID_LOWK_B0_2026-09-07.md`](../reports/REPORT_HYBRID_LOWK_B0_2026-09-07.md)。
> **B0 PASS**（§13 的形式与系数对；375 个模式把力误差从 14–18% 压到 0.19–1.06%；
> ⚠ 固定电荷 + 严格中性合成体系 ⇒ **H2 的三条 response 通道一条都没验**）、
> **C0/C1/C2 PASS**。但三条计划里的表述被实测修正，见 §13.3 / §13.2 / §14.6 的 ⚠ 段。
> 复现：`lips-hybrid-lowk`（B0）、`python src/lips/engine/closure.py`（C0–C2）。
>
> **2026-09-07 追加指针**：「PSWF 能不能解决长波长屏蔽不足」这个问题的完整回答在
> [`../reports/PROJECT_STATUS_2026-09-07.md`](../reports/PROJECT_STATUS_2026-09-07.md) §4。
> 结论收紧为：**紧支撑 kernel 的 $\hat U(0)$ 有限，恢复不了 Coulomb 的 $4\pi/k^2$ 极点
> 与 SL 渐近系数；但有限 $k$ 上窗仍有真自由度（$q\approx2.6$ 处窗间落差 45%）。**
> 同时作废三条早前的越界表述：「$S_{ZZ}$ 比值发散」、「窗只能动 6%」、
> 「全类天花板 1.39×」（后者依赖已被推翻的 $E_F$，只能叫 surrogate 下界）。
> 本文 §3.2 那条「$\hat S\to4\pi/k^2$，任何 $\chi$ 都改不了」是**对**的，
> 而且它比 $\hat c(0)=1$ 更一般——依据是紧支撑，不是归一化。
>
> 目的：在不动现有 ZMM 路线的前提下，把 pair closure 抽象成"窗函数族"，
> 并在这一层上开发基于 PSWF 的新 closure。
>
> 配套文档：`MATH_CHANGE_MAP.md`（改数学时的同步清单）、
> `PLAN_LocalCWLDForce_OpenMM_Plugin.md` §18.3（插件 kernel 的唯一规格来源）。

---

## 0. 一句话

现有的 Sakuraba ZMM closure 已经**字面上**是窗函数正规形式
$U(r)=\big(1-A(r/r_c)\big)/r$；所谓"换成 PSWF"精确地就是换掉窗 $\chi$。
因此正确做法不是加一个 `if`，而是把这一层显式化，让 ZMM / PSWF / 约束 PSWF
成为同一层的三个实例——消融对照因此是免费的。

---

## 1. 出发点：现有 closure 已经是窗函数正规形式

`test_lips_vs_pmeV2.6.py:49` 的 `zmm_uclosure()` 三个分支，按 $x=r/r_c$ 提出 $1/r$ 后：

$$U_\ell(r)=\frac{1-A_\ell(x)}{r},\qquad A_\ell(x)=\int_0^x\chi_\ell(t)\,dt$$

| $\ell$ | 代码里的名字 | $\chi_\ell(x)$ | $A_\ell(x)$ | 代码常数项 |
|---|---|---|---|---|
| 1 | ZDipole/RF-like | $\tfrac32(1-x^2)$ | $\tfrac32x-\tfrac12x^3$ | $-1.5/r_c$ |
| 2 | ZQuad **(生产默认)** | $\tfrac{15}{8}(1-x^2)^2$ | $\tfrac{15}8x-\tfrac54x^3+\tfrac38x^5$ | $-\tfrac{15}{8r_c}$ |
| 3 | ZOct | $\tfrac{35}{16}(1-x^2)^3$ | $\tfrac{35}{16}x-\tfrac{35}{16}x^3+\tfrac{21}{16}x^5-\tfrac5{16}x^7$ | $-\tfrac{35}{16r_c}$ |

即

$$\chi_\ell(x)=c_\ell\,(1-x^2)^\ell,\qquad c_\ell=\frac{(2\ell+1)!!}{(2\ell)!!}$$

$c_\ell$ 由归一化 $\int_0^1\chi_\ell=1$ 定死，**而它恰好就是代码里那个常数项的分子**
（1.5、15/8、35/16）。三阶逐项验算全部对上，无一例外。

两个立即可读出的结构性事实：

- $A_\ell(1)=1\ \Rightarrow\ U(r_c)=0$（能量在 cutoff 归零）。
- $U'(r_c)=-\chi(1)/r_c^2\ \Rightarrow$ **力在 cutoff 连续 $\iff\chi(1)=0$**；
  更高阶光滑性 $\iff\chi$ 在 $x=1$ 处零点阶数更高。
  $(1-x^2)^\ell$ 在 $x=1$ 有 $\ell$ 阶零点——这正是 ZMM 系数被解出来的条件。

代码注释写的构造条件是 $(d^mU/dr^m)|_{r_c}=0,\ m=0..\ell$；Sakuraba 的物理解释是
排除区多极矩逐阶抵消。在窗函数语言里两者是同一件事的两种说法。

> 副产品：CWLD 的密度核 $K(r)=(1-(r/r_{\rm env})^2)^2$ 就是 $\chi_2$ 去掉归一化。
> 同一个窗在代码里出现了两次，但**这两处不要合并**——它们的物理约束不同
> （密度核只需紧支撑，closure 还要背静电求和规则）。

---

## 2. 目标与非目标

**目标**

1. 把 pair closure 抽象成窗函数层，ZMM 成为该层的一个实例。
2. 在该层上实现 PSWF 系窗，开发出一个新的 real-space、pair-additive、无 FFT 的 closure。
3. 全程保留 ZMM 入口，老结果可无损复现。

**非目标（写死，防止范围蔓延）**

- 不改密度核 $K(r)$、不改 $Q(\text{dens})$ 映射、不改链式响应力的结构。
  → 因此 `MATH_CHANGE_MAP.md` 的 **#3 分析侧镜像不受影响**，DEC-002 不作废。
- 不在本轮引入 FFT / reciprocal space。终点仍必须是
  $U=\sum_{i<j}^{r_{ij}<r_c}Q_iQ_jK(r_{ij})$。
- 不改 CWLD 的默认参数。参数重选是 Stage 3 之后的事。

---

## 3. 核心理论问题：丢掉 $S$ 到底丢了什么

**这一节是整条路线的成败点。代码只是它的下游。**

### 3.1 被丢项的精确形式

split：$1/r=L(r)+S(r)$，$L(r)=\big(1-A(x)\big)/r$，$S(r)=A(x)/r$，$x=r/r_c$。

注意 $A(x)=1$ 对 $x\ge1$，所以 $S(r)=1/r$ **精确地**一直延伸到无穷远——
$S$ 可 pairwise 求和，但**不是有限 cutoff 内能加完的东西**。周期体系里它是

$$U_S=\tfrac12\sum_{i}\sum_{(j,n)\neq(i,0)}q_iq_j\,S\big(|\mathbf r_{ij}-\mathbf nL|\big)$$

Poisson 求和后（中性体系，$k=0$ 项为零）：

$$U_S=\frac1{2V}\sum_{\mathbf k\neq0}\hat S(k)\,|\hat\rho(\mathbf k)|^2\;-\;\tfrac12 S(0)\sum_iq_i^2$$

其中 $\hat\rho(\mathbf k)=\sum_jq_je^{-i\mathbf k\cdot\mathbf r_j}$。
两个要点：

- $S$ 在原点**正则**：$S(0)=\chi(0)/r_c$（因为 $A(x)\simeq\chi(0)x$）。
  所以自能项是有限的、且**直接由 $\chi(0)$ 决定**。
- $\hat S(k)=4\pi/k^2-\hat L(k)$，$\hat L$ 因 $L$ 紧支撑而处处光滑。

> ### ⚠ 2026-09-01 勘误：§3.2–§3.4 的推导已被 Stage 1 修正
>
> 1. **§3.2 的结论要收回一半。** 分部积分后有闭式
>    `Ŝ(k) = (4π/k²)·ĉ(q)`，`ĉ(q)=∫₀¹χcos(qx)dx`（数值验到 9e-14）。于是
>    `E_F ∝ ∫ĉ(k·rc)²S_ZZ(k)dk` —— **正是 PSWF 的能量集中问题本身**，
>    只是权重从锐截断换成物理的 `S_ZZ`。PSWF 的判据是正确判据的近似，
>    不是"不相干的判据"。（"`Ŝ` 在 k→0 必然发散、任何 χ 改不了"这一条仍然成立。）
> 2. **§3.4 的约束写错了。** ZMM 的条件是**端点零点**
>    `χ(1)=…=χ^(ell-1)(1)=0` 加归一化，**不是**矩条件。因此可行空间可写成
>    `χ=(1-x²)^ell·P(x)`，端点条件自动满足，只剩一条线性约束——
>    这也让优化变成良态的小型 QP。
> 3. **§3.4 的 `L_PSWF + C_closure` 分解没有额外自由度。** 任何在 rc 归零的
>    pair kernel 都等价于某个窗（令 `A(x)=1-r·U(r)`），所以 `C_closure` 就是换窗。
> 4. **§8 风险 1（端点连续性代价）不成立**：实测只值 3%（0.519 vs 0.533）。
>
> 以下 §3.2–§3.4 原文保留，作为推导过程的记录。

### 3.2 为什么 PSWF 的最优判据不是我们要的判据

$\hat L$ 在小 $k$ 展开 $\hat L(k)=\hat L(0)-\tfrac{k^2}6\!\int r^2L\,d^3r+\dots$，于是

$$\hat S(k)=\frac{4\pi}{k^2}-\hat L(0)+O(k^2)$$

**$\hat S(k)$ 在 $k\to0$ 必然发散成 $4\pi/k^2$，任何 $\chi$ 都改不了这一点**——
因为 $S$ 按定义扛着完整的 $1/r$ 尾巴。所以"把 $\hat S$ 做小"这个想法在小 $k$ 端是死的。

截断之所以还能成立，靠的是**体系自身的屏蔽**：中性导电液体的电荷结构因子满足
Stillinger–Lovett，$S_{ZZ}(k)\to k^2/\kappa_D^2$，恰好抵消 $4\pi/k^2$，使被丢项有限。

**这直接推翻了"PSWF 最优所以更好"这个朴素论证：**

| | 最优性判据 | 为谁服务 |
|---|---|---|
| ZMM $(1-x^2)^\ell$ | 排除区低阶多极矩归零 / $r_c$ 处高阶光滑 | **截断本身**（closure） |
| PSWF $\chi_\alpha$ | $\hat\chi$ 带限最紧，$\|\hat\chi\|_{|k|>k_c}$ 最小 | **FFT 网格大小**（我们没有 FFT） |

原论文里 $S_{\rm PSWF}$ **被精确算掉了**，只是算得便宜；PSWF 从未被要求"可忽略"。
我们要丢掉它，用的是完全不同的许可证。

### 3.3 正确的判据

对各向同性液体做系综平均，$\langle|\hat\rho(\mathbf k)|^2\rangle=N\langle q^2\rangle S_{ZZ}(k)$，
被丢的每粒子能量是 $\chi$ 的泛函：

$$\boxed{\;\frac{\Delta U[\chi]}{N}=\frac{\langle q^2\rangle}{4\pi^2}\int_0^\infty\!\!dk\;k^2\,\hat S(k)\,S_{ZZ}(k)\;-\;\frac{\langle q^2\rangle}{2}\,\frac{\chi(0)}{r_c}\;}$$

> 系数按标准约定（$\sum_{\mathbf k}\to\frac V{2\pi^2}\int k^2dk$）推出，
> **实现前必须独立复核一遍**，这里只定形式不定数值。

设计自由度的用法应该是：把残差拆成
(i) 可被一个光滑 pair 项 $C_{\rm closure}(r)$ 在 $r<r_c$ 内吸收掉的平均场部分，
(ii) 真正随构型变化的误差。**最小化 (ii)**，而不是最小化 $\Delta U$ 本身。

### 3.4 三个候选窗族

| name | $\chi(x)$ | 角色 |
|---|---|---|
| `zmm1/2/3` | $c_\ell(1-x^2)^\ell$ | **现有基线**，绝不改动 |
| `pswf` | $\chi_{\rm PSWF,\alpha}$（端点修正后） | **消融对照**：证明纯带限判据不够 |
| `pswf-zmm` | 带矩约束的 PSWF | **候选新方法** |

`pswf-zmm` 的变分问题（Stage 1 的正题）：

$$\chi^\star=\arg\min_\chi\ \big\|\hat\chi\big\|^2_{|k|>k_c}
\quad\text{s.t.}\quad
\begin{cases}
\int_0^1\chi(x)x^{2m}dx=\mu_m,& m=0..\ell\\[2pt]
\chi(1)=\chi'(1)=\dots=0 & (\ell\ \text{阶零点})
\end{cases}$$

目标泛函二次、约束线性 ⇒ 带约束广义特征值问题，有闭式解法。
物理含义：**localization 的最优性由 PSWF 负责，long-range closure 的守恒性由矩约束负责**，
两件事第一次被分开表述——而不是像 $(1-x^2)^\ell$ 那样让一个多项式同时硬扛。

### 3.5 Kill switch（重要）

**PSWF 系不保证赢。** 上面的泛函是个一维积分，给定 $S_{ZZ}(k)$ 就能直接算，
**不需要跑任何 MD**。Stage 1 必须先做这个，若不能明显优于 `zmm2`，
**整条路线就地终止，不进 Stage 2**。

> **2026-09-01 执行结果：初判 FAIL，当日修订为「暂缓，未判完」。**
> 判据与 Sakuraba ZMM 论文在同一工作点的 ell 推荐冲突，阻塞在实测 S_ZZ 上，
> 详见报告 §7。以下初判数值降级为暂定值。 门槛是 gain_rms ≥ 2x（E_F 比 ≤0.25），
> 实测**全类天花板 1.39x**（连允许 rc 处力跳变都只到 0.519），对
> s∈[0.5,30] 与两种 $S_{ZZ}$ 形状一致。且 PSWF（0.521）与直接优化的多项式窗
> （0.536）在同一水平，PSWF 本身没带来任何额外东西。
> **按预先登记的规则：PSWF 专项终止，不进 Stage 2。**
> 两个保留的副产品（`L_IPS_ZMM_ELL=1` 零成本对照、优化多项式窗 1.35x）
> 见报告 §5，是否推进需用户决定。详见
> [`STAGE1_CLOSURE_KILLSWITCH_REPORT.md`](STAGE1_CLOSURE_KILLSWITCH_REPORT.md)。

$S_{ZZ}(k)$ 两个来源：
1. 解析模型（Debye–Hückel / Stillinger–Lovett 形式）—— 快，我可以直接做。
2. 从 `/home/ruigengji/L-IPS/2.4/` 现有 dcd 实测 —— 更可信，需要你在节点上跑。

---

## 4. 架构设计

### 4.1 窗函数层

新模块（暂定 `closure_windows.py`，主脚本与插件共享或各自镜像一份），
每个窗注册为一个对象，提供：

```
name        -> str
chi(x)      -> χ(x),        x∈[0,1]
A(x)        -> ∫₀ˣ χ
U(r, rc)    -> (1 - A(r/rc)) / r          能量
dUdr(r, rc) -> 解析导数（插件侧手写实现需要）
lepton(rc)  -> Lepton 表达式字符串（主脚本 CustomGBForce 需要）
coeffs()    -> 多项式系数元组（fast evaluator 的系数表需要）
```

**自检函数**（任何新窗注册时自动跑，不过不给用）：

1. 归一化 $\int_0^1\chi=1$（否则 $U(r_c)\neq0$）
2. 端点 $\chi(1)=0$、$\chi'(1)=0$ —— **力连续性**，见 §9 风险 1
3. 矩 $\int_0^1\chi x^{2m}dx$ 打印出来备查
4. `dUdr` 与 `U` 的中心差分一致（1e-8 量级）
5. $r\to0$ 时 $U\to1/r$（前导奇异性未被破坏）

### 4.2 入口点

```
L_IPS_CLOSURE = zmm (默认) | pswf | pswf-zmm
  └ =zmm 时继续读 L_IPS_ZMM_ELL ∈ {1,2,3}，默认 2 —— 语义一字不变
  └ =pswf / pswf-zmm 时读 L_IPS_PSWF_ALPHA 等新变量（Stage 1 后确定）
```

**不设 `L_IPS_ZMM_ELL=4` 之类的偷渡口**。新方法走新变量，日志里必须打印
完整 closure 标识（窗名 + 全部参数 + rc），进 csv 的 metadata 列。

### 4.3 纪律：老路径一个字节都不动

`zmm_uclosure()` 现有三行硬编码字符串**不重写**成 $(1-A)/r$ 形式。
数学上等价，但 Lepton 浮点求值顺序会变，fixture 的 1e-12 断言可能变红，
白白制造一次"到底是重构错了还是新方法错了"的调试。

做法：`zmm` 分支继续走老字符串；窗层只服务新窗；**另写一个测试**证明
窗层的 `zmm2` 与老字符串解析等价（在密网格上比到 1e-14）。
这个测试同时也是窗层实现正确性的第一个证据。

### 4.4 插件侧

- `Globals` 加 `closure_kind: str` 字段；`zmm_order` 语义保持不变（仅 `zmm` 时有意义）。
- `reference.py` 加 `window_uclosure` / `window_duclosure_dr`，
  老的 `zmm_uclosure` / `zmm_duclosure_dr` **原样保留**（fixture 直接引用它们）。
- `evaluator.py` 的系数表从 `dict[int]` 变成按窗名分派；`zmm` 三档系数原样。
- `PLAN_LocalCWLDForce_OpenMM_Plugin.md` §18.3 从"一个公式"扩成"一个族"，
  **追加不覆盖**——将来写 CUDA kernel 的人必须能同时看到两条规格。

---

## 5. 逐文件改动清单

| # | 文件 / 位置 | 改什么 | 阶段 |
|---|---|---|---|
| 1 | `test_lips_vs_pmeV2.6.py:44-46` | 加 `L_IPS_CLOSURE` 解析与校验 | S0 |
| 2 | `test_lips_vs_pmeV2.6.py:49-66` | **不动**；新增 `pair_uclosure()` 分派器 | S0 |
| 3 | `test_lips_vs_pmeV2.6.py:1195` | `{zmm_uclosure()}` → `{pair_uclosure()}`（exact 引擎） | S0 |
| 4 | `test_lips_vs_pmeV2.6.py:1247` | 同上（fast 引擎）；`1191-1192` 的打印改成通用标识 | S0 |
| 5 | `closure_windows.py`（新） | 窗层 + 自检 + `zmm` 三档注册 | S0 |
| 6 | `reference.py:34,55` | 加 `window_*`，老函数保留 | S2 |
| 7 | `reference.py:252` | 按 `closure_kind` 分派 | S2 |
| 8 | `evaluator.py:81` | 系数表按窗名分派 | S2 |
| 9 | `evaluator.py:~200` | `closure = invr + polynomial` 走分派结果 | S2 |
| 10 | `PLAN_...md` §18.3 / §31 | closure 规格扩成族 | S2 |
| 11 | `MATH_CHANGE_MAP.md` | 加一行指向本文档 | S0 |

`test_lips_vs_pmeV2.5.py`、`legacy_versions/`、`block_time_check_v26.py` **不动**
（后者只重算 q-profile，不碰 closure）。

---

## 6. 冻结物处理

| 冻结物 | 处理 |
|---|---|
| 9 个 fixture + `sha256sums.txt` | **原样保留、必须继续绿**。这是回归保证。新 closure 生成**新文件**，sha256sums.txt **追加**不改写 |
| DEC-002（密度核） | 不受影响（我们不动 $K(r)$） |
| `README_v2.6_1CKK_CONCLUSION.md` 等 | 结论仅对 `zmm2` 有效。新 closure 的结果**另开文档**，不要混写 |
| 新增 `DEC-004-closure-family.md` | 记录：窗层抽象、为什么保留 ZMM 入口、PSWF 判据不匹配问题（§3.2）、Stage 1 的 kill switch 结果 |

---

## 7. 分阶段计划

### Stage 0 · 基建（纯工程，数值零变化）

窗层 + 入口 + 分派器 + 等价性测试。

**门禁**：`L_IPS_CLOSURE=zmm` 下所有输出与改动前逐位相同；
9 个老 fixture 全绿；窗层 `zmm2` 与老 Lepton 串在密网格上一致到 1e-14。

> 做完不会有任何"进展感"，但它是之后所有对照的地基。

### Stage 1 · 数学（独立脚本，不进生产代码）

1. 核对 PSWF 定义（$\chi_\alpha$ 的归一化、$\alpha$ 的含义、**端点值**）——见 §9 风险 1。
2. 实现 $\Delta U[\chi]$ 泛函（§3.3），用解析 $S_{ZZ}$ 对 `zmm1/2/3` + `pswf` 扫描。
3. **Kill switch 判定**（§3.5）。
4. 通过则解带约束变分问题得 $\chi^\star$，验矩与端点。
5. $A(x)$ 拟合解析代理（多项式/有理式），**不要 1024 点样条表**——见 §9 风险 2。

**门禁**：kill switch 通过；$\chi^\star$ 过窗层全部 5 项自检。

### Stage 2 · 接线

三份实现同步（Lepton 串 / reference 手写解析导数 / fast 系数表）+ 新 fixture +
plan §18.3 扩写 + DEC-004 落笔。

**门禁**：三份实现互相 parity（沿用 LCWLD-020/030 那套逐元素比对方法）；
新 fixture 的 U/dU 与独立中心差分一致。

### Stage 3 · 验证

**必须按顺序，先纯静电后 CWLD。** closure 和隐式极化**可以分阶段隔离测试**
（固定 $Q$ 时只有 pair kernel 在变），一起改就没法归因。

> ⚠ **措辞更正（2026-09-07）**：原文写的是「正交的两个变量」，**数学上不对**。
> 完整 CWLD 里 $\lambda_i=C\sum_j Q_jU_\ell(r_{ij})$、
> $g_i=\lambda_i\,dQ_i/d\,\mathrm{dens}_i$，**换 $U_\ell$ 会直接改变链式响应力**。
> 「可分阶段隔离测试」成立，「正交」不成立。

| 关 | 内容 | 判据 |
|---|---|---|
| 3a | NaCl 晶格 Madelung 能 | 与解析值相对误差 |
| 3b | 纯水盒 vs PME（能量、$g_{OO}$、介电常数） | 与 PME 一致 |
| 3c | **NVE 能量漂移** | 专抓 $\chi(1)\neq0$ 导致的力跳变 |
| 3d | 1CKK，与 `zmm2` 同协议对照 | 5ns×3seed |
| 3e | ABFE decharge 能量一致性 | 见 §9 风险 3 |

---

## 8. 风险与已知坑

1. **prolate 函数端点不为零。** $U'(r_c)=-\chi(1)/r_c^2$，$\chi_\alpha(1)\neq0$ ⇒
   力在 cutoff 跳变 ⇒ NVE 能量漂移。ESP 能容忍是因为它把 $S$ 精确算掉了，
   我们不算就不能容忍。**Stage 1 第一件事就是核对这一点**；若确实不为零，
   必须用端点加窗的变体（这也正好被 §3.4 的 $\chi(1)=\chi'(1)=0$ 约束覆盖）。
2. **不要用 1024 点样条表。** DEC-002 已有实测：tabulated kernel 的 $dK/dr$
   在边界有 ~1e-2 的样条振铃。那还只是密度核；放到**静电力**上会直接进 force。
   必须走解析代理拟合。
3. **ABFE decharge 一致性。** $U$ 的常数项 $-c/r_c$ 贡献 $q_iq_j\times$const，
   自能项由 $\chi(0)$ 决定（§3.1）。换窗 = 换自能常数 = 换 $\Delta G_{\rm decharge}$ 的零点。
   净电荷修正必须跟着重新推，不能沿用 ZMM 的。
4. **新 closure = 新基线。** `a_q2=0.5, ca_source_weight=2.0` 是在 `zmm2` 下选出来的，
   换 closure 后参数选择原则上要重选。别默认继承。
5. **$\alpha$ 不能抄论文。** 论文的 $\alpha$ 是在"实空间 cutoff 成本 vs FFT 网格成本"
   之间取的最优；我们没有 FFT，目标函数换成 §3.3 的 $\Delta U[\chi]$，最优 $\alpha$ 必然不同。
6. **论文原文我没读到。** 本文档中所有关于 $\chi_{\rm PSWF}$ 具体性质的陈述
   （端点值、$\alpha$ 定义、归一化约定）**全部标记为待核对**，Stage 1 第一步就是拿原文对。

---

## 9. 待定决策

| # | 决策 | 状态 |
|---|---|---|
| D1 | **方法名称**（决定 env 值、fixture 前缀、DEC 编号、将来论文里的叫法） | **待你定** |
| D2 | 矩约束阶数 $\ell$ 取几（与 `zmm2` 对齐取 2？还是扫描） | Stage 1 后定 |
| D3 | $S_{ZZ}(k)$ 用解析模型还是实测 | 建议先解析、后实测复核 |
| D4 | 窗层是主脚本与插件共享一份文件，还是各自镜像 | Stage 0 前定 |
| D5 | 是否同时把密度核 $K(r)$ 也纳入窗层 | **建议不**，见 §1 副产品 |

---

## 10. 给下一个会话的接手须知

- 本文档是这条路线的**唯一设计来源**。开工前先读它 + `MATH_CHANGE_MAP.md`。
- 当前状态：**Stage 1 kill switch 已执行并 FAIL；Stage 0 未接入生产脚本，
  `test_lips_vs_pmeV2.6.py` 一字未动。** 已落地的是两个独立模块：
  `closure_windows.py`（窗层，ZMM 三档系数与 v2.6 冻结值逐位断言）、
  `stage1_closure_kill_switch.py`（判定脚本）。
- **别在没读 `STAGE1_CLOSURE_KILLSWITCH_REPORT.md` 之前重启这条路线。**
  §3.2–§3.4 的原始推导有三处错误，已在上面的勘误块里改正。
  kill switch 确实按设计发挥了作用：整条路线被自己的数学否掉，省下了 Stage 2/3。
- 环境/运行纪律：CUDA-only，无调度器；MD 与轨迹分析由用户在节点上跑，
  助手只写代码 + 跑 pytest / 静态 System 构建（见记忆 `feedback-user-runs-md-locally`）。

---

## 11. 路线重定级（2026-09-07，用户决定）

> ⚠⚠ **本节第一版的前提是错的（2026-09-07 用户指出，已改）。** 我曾写「这份计划隐含的
> 目标是靠换窗恢复 Coulomb 长波物理，该目标已判死」——**那个目标不在 §2 里**。
> （成文时间不作断言：用户指出本文件是新版本，而 `archive/` 里没有旧副本可比对；
> 但这不影响更正的实质 —— 现行 §2 里没有那一条，是我给它安的。）
> §2 的三条目标是「把 closure 抽象成窗层、ZMM 成为其中一个实例」、
> 「在该层上做出一个 real-space / pair-additive / **无 FFT** 的新 closure」、
> 「保留 ZMM 入口可无损复现」。**「长波物理」是我给它安的目标，然后又宣布它死了。**
> 真实发生的事是：Stage 1 的判据（$E_F$ = 截断误差）让路线**看起来**是在追长程精度，
> 于是后续讨论围着那个不存在的目标转。下面按 §2 的原始三条重新计分。

### 11.0 按计划自己的三条目标计分

| §2 的目标 | 状态 |
|---|---|
| **1. 把 pair closure 抽象成窗函数层，ZMM 成为该层的一个实例** | ✅ **已完成**。`engine/closure.py` 的窗层；且 §14 的 PCF 让 ZMM 成为 $c\to0$ 的**退化极限**，比"实例"更强——每条 ZMM 陈述都自动是 PCF 的特例 |
| **2. 在该层上做出一个 real-space / pair-additive / 无 FFT 的新 closure** | ⚠ **形式上完成、精度上没有收益**。`pswfz2` 与 PCF 都已实现并过自检，但静态力审计上 pswfz2 差 zmm2 3.7%、PCF 的 $c$ 单调变坏，**族内最优落在退化边界 $(\ell{=}1,c\to0)$**（= zmm1） |
| **3. 全程保留 ZMM 入口，老结果可无损复现** | ✅ **已完成**。ZMM 三档冻结字符串逐位未动、默认路径不导入窗层、别名纪律见 §14.4 |

⇒ **正确的总结是：目标 1 与 3 达成，目标 2 达成了"做出来"但没达成"更准"。**
不是"路线判死"，而是**「新 closure 在纯截断路线上没有精度收益」这一条实测结论**。

### 11.1 一条确实成立的解析限制（但它不是原目标）

只要还是紧支撑 pair kernel，$\hat U(0)$ 就有限，补不回 $4\pi/k^2$ 的非解析极点
（`../reports/PROJECT_STATUS_2026-09-07.md` §4.1）。
**这条限制是对的，但它约束的是一个 §2 从未承诺的目标** —— 记在这里是为了防止将来有人
再把"换窗解决长程"当成本路线的卖点，**不是**对本路线的判决。

### 11.2 三条后续路线（其中一条越界，必须显式决策）

closure-family 本身的用途：

| 路线 | 状态 | 继续？ |
|---|---|---|
| PSWF 单独解决 $k\to0$ screening | **死** | ❌ |
| low-$k$ 加权 closure，改善有限盒误差 | **活，属于便宜优化** | ✅ 做一轮 kill-switch |
| **closure + 少量 low-$k$ reciprocal modes**（§13） | **很活**，但 ⚠ **越界** | 见下方越界声明 |
| 按原 band-concentration objective 再做 pswfz2 | 基本没必要 | ❌ |
| zmm1/2/3 继续扫 | 作为 baseline | ✅ 但不当新方法 |
| **PCF：把 closure 推广成 $(\ell,c)$ 两参数 family（§14）** | **活，属于「定义与叙事」层面的贡献** | ✅ 与 §13 互补：PCF 提供旋钮，§13 让旋钮值钱 |

> ⚠⚠ **§13 撞在 §2 写死的非目标上**（2026-09-07 补记）：§2 明写「**不在本轮引入
> FFT / reciprocal space**，终点仍必须是 $U=\sum_{i<j}^{r_{ij}<r_c}Q_iQ_jK(r_{ij})$」。
> §13 引入低-$k$ 倒空间修正，**直接违反这一条**。它是我（assistant）提出的**范围扩展**，
> 不是本计划的自然延续。
> ⇒ **要做它必须是一次显式的范围决策**（"本轮"的边界是否重划），
> 不能因为它"看起来最有价值"就默认继续。B0 已 PASS 只说明**形式与系数是对的**，
> 不构成开工许可。§14 的 PCF 则**不越界**：它只换窗，仍是 real-space / pair-additive / 无 FFT。

**定级的现实依据**：`zmm1` 已经在静态力审计上赢 `zmm2`，而且第一壳（0.844）/
第二壳（0.788）/蛋白重原子（0.789）不是小赢。所以 pure-closure 这条线最合理的目标
**不再是「证明 PSWF 牛」**，而是：

> **能不能找到一个 closure，同时 (a) 保留 zmm1 的局域力优势、
> (b) 显著压低最低几个 $k$ mode 的结构误差、(c) NVE 不恶化。**
> **做不到就停。** 做到一点但长波仍明显错，它正好变成路线 B 的 real-space 前半截。

⇒ **本路线从「主物理解决方案」降级为「real-space kernel designer + hybrid 方法的前半截」。**
这比原来的「PSWF magically replaces PME」靠谱得多。

**并且不再迷信「PSWF」这个名字**：真正的对象叫 **closure optimization**，
PSWF 只是一种 basis / optimizer 候选。Chebyshev、Legendre、直接参数化都等价：

$$
\chi(x)=(1-x^2)^\ell\bigl(a_0+a_2x^2+\cdots+a_{2m}x^{2m}\bigr)
$$

（$(1-x^2)^\ell$ 保证端点条件自动满足，只含偶次幂保证 §3.5 的偶性前提，
于是可行空间就是系数 $a$ 的一个仿射子空间——归一化 $\int_0^1\chi=1$ 是唯一线性约束。）

---

## 12. 路线 A：有限 $k$ 的 closure 优化（便宜，先做）

### 12.1 目标函数

关键的重新定义：**$\mathcal K_{\rm box}$ 不是假想的连续 $k\to0$，而是这个盒子实际存在的
离散 modes**，从 $k_{\min}=2\pi/L$ 起。

$$
\min_{\chi}\ \Bigl[\sum_{k\in\mathcal K_{\rm box}} w_k\,|\hat c(k r_c)|^2
\;+\;\lambda_F E_{\rm force}\;+\;\lambda_s E_{\rm smooth}\Bigr]
$$

* 第一项：有限盒里实际能访问的长波模式上的截断误差（$w_k$ 可取实测 $S_{ZZ}(k)$ 或
  模式多重度）。
* $E_{\rm force}$：局域力误差（**不能省**，否则会重演 pswfz2「谱好了力差了」）。
* $E_{\rm smooth}$：$r_c$ 处光滑性/NVE 代理。

**这条路线不是「恢复 Coulomb」，而是：在不可避免缺失 long-range pole 的前提下，
让有限盒里的错误尽量小。** 这个陈述完全成立。

### 12.2 两道必过的门（$E_F$ 不再当最终判据）

$$
\text{有限-}k\ S_{ZZ}\quad+\quad\text{直接静态力审计}
$$

理由：$E_F$ 已被证明在 2–5% 量级不可靠、可能翻符号
（`STAGE1_CLOSURE_KILLSWITCH_REPORT.md` §11.3）。优化可以用它当**廉价的搜索代理**，
但**判定必须用 `lips-force-audit` + 实测 $S_{ZZ}$**。
当前 `pswfz2` 在直接力审计里每个通道都比 zmm2 差 3–15%，正说明
**原 band-concentration objective 与本项目的目标错位**。

### 12.3 kill switch

优化解若不能同时满足「力审计不差于 zmm1」与「$S_{ZZ}(k_{\min})$ 明显改善」
⇒ **停，把结论写进本节，不进生产**。

---

## 13. 路线 B：local closure + 少量 low-$k$ reciprocal 修正（最值得做）

不是重做完整 PME，而是**只补真正缺掉的那几个长波模式**。
数据已经指明缺口位置：$k\lesssim2\pi/r_c\approx5.24$ nm⁻¹ 异常最大，往上逐渐接近 PME
（`PROJECT_STATUS_2026-09-07.md` §7.4）。

$$
U_{\rm total}=U_{\rm local}^{\rm optimized}+U_{\rm low\text{-}k},\qquad
U_{\rm low\text{-}k}=\frac1{2V}\sum_{\mathbf k\in\mathcal K_{\rm low}}
\frac{4\pi}{k^2}\,w(k)\,\bigl|\rho_q(\mathbf k)\bigr|^2
$$

### 13.1 $w(k)$ **不是自由参数**，它就是窗自己的余弦变换（2026-09-07 新推）

由 §3.1 的 split $1/r=L+S$、$L=U_\ell$、$\hat S(k)=\dfrac{4\pi}{k^2}\hat c(kr_c)$：

$$
\hat U_\ell(k)=\frac{4\pi}{k^2}\bigl(1-\hat c(kr_c)\bigr)
$$

**每个模式上缺掉的量恰好是 $\hat S(k)$**，所以

$$
\boxed{\;w(k)=\hat c(k r_c)\;}
$$

而 $\hat c$ 有闭式（`PROJECT_STATUS_2026-09-07.md` §3.6）：
$\hat c_\ell(q)=(2\ell+1)!!\,j_\ell(q)/q^{\ell}$，
**用级数算，不要用三角闭式**（$q\lesssim2$ 相消两三个量级）。

> 自洽性检查（已手算核对）：$q\to0$ 时 $1-\hat c=q^2m_2/2$，于是
> $\hat U_\ell(0)=2\pi r_c^2m_2$ 有限；对 $\ell=1$ 直接积分
> $4\pi\int_0^{r_c}U_1r^2dr=4\pi r_c^2/10$，与 $m_2=1/5$ 给的值**逐位相同**。
> 这同时正面印证了 §4.1「紧支撑 ⇒ $\hat U(0)$ 有限」。

### 13.2 由此得到一个很干净的设计分解（2026-09-07 新推）

带内（$k<k_c$）修正之后：

$$
\hat v_{\rm total}(k)=\frac{4\pi}{k^2}\bigl(1-\hat c\bigr)+\frac{4\pi}{k^2}\hat c
=\frac{4\pi}{k^2}\qquad(k<k_c)
$$

⇒ **带内静电变成精确 Coulomb，而且与窗的选择完全无关**（任何 $\chi$ 都一样）。
残差只剩带外：

$$
\Delta\hat v(k)=\frac{4\pi}{k^2}\hat c(kr_c)\qquad(k>k_c\ \text{未修正的模式})
$$

⇒ **窗的选择只影响两件事**：

| 窗影响什么 | 判据 |
|---|---|
| 实空间**局域力精度** | `lips-force-audit`（zmm1 目前最好） |
| **带外残差** | $\int_{q>q_c}\hat c^2S_{ZZ}\,dq$ ← **这正是 PSWF 的能量集中目标** |

**所以 PSWF 的目标在这里才是对的**：纯 cutoff 下它牺牲的带内部分恰恰是要命的部分
（所以力审计判它差）；而在 hybrid 下带内已被显式修正，**带外残差就是唯一的长程误差**。

**pswfz2 的 13.7× 带外能量削减因此从「买了个没用的东西」变成「就是本路线的收益指标」。**

> ⚠ **2026-09-07 实测修正：这个分解是对的，但「带外残差」必须在实际用的 $k_c$ 上算。**
> 实测 $\hat c$ 包络：pswfz2 在 $q\approx8$ 被 zmm3 反超、$q\approx13$ 被 zmm2 反超，
> $q=50$ 时比 zmm3 差 14×。机制：它解的是 $\min\int_{q>2\pi}\hat c^2$，那个积分由紧挨着
> $2\pi$ 的头几个振荡主导 ⇒ 最优解**压带边、换胖尾巴**。于是 hybrid 主表里 pswfz2
> 在 $k_c\gtrsim10$ 处**平台化**。
> ⇒ **选窗与选 $k_c$ 是联合决策**：$k_c\approx2\pi/r_c$（375 模式，便宜）选 pswfz2；
> $k_c\gtrsim10$ 则**提高 $\ell$ 更划算**（zmm3 反超全部候选）。
> 详见 `../reports/REPORT_HYBRID_LOWK_B0_2026-09-07.md` §5。

### 13.3 收益量化：好窗 ⇒ 需要补的模式更少（量级估算）

$\hat c_\ell^2$ 平均按 $q^{-2(\ell+1)}$ 衰减 ⇒ 带外残差 $\propto q_c^{-(2\ell+1)}$，
$\ell=2$ 时 $\propto q_c^{-5}$；而模式数 $\propto k_c^3$。于是

$$
\text{pswfz2 的 }13.7\times\text{ 带外削减}\;\equiv\;q_c\ \text{放大}\ 13.7^{1/5}=1.69\times
\;\equiv\;\text{模式数}\ 1.69^3\approx4.8\times
$$

⇒ **同样的长程残差，pswfz2 只需要 zmm2 约 1/4.8 的 reciprocal 模式数。**

> ⚠ **2026-09-07 实测作废这个数**（`../reports/REPORT_HYBRID_LOWK_B0_2026-09-07.md` §4）。
> 直接量的节省倍率不单调、且在最严的目标上翻成劣势：目标 rel 3e-2/1e-2/3e-3/1e-3 分别为
> 1.00× / 1.00× / 3.34× / **0.68×**。**「4.8×」不要再引用。**
> 能站住的只有一条：在标称带宽 $k_c=2\pi/r_c$ 上，pswfz2 的**残差**比 zmm2 低 **3.8×**
> （1.89e-3 vs 7.12e-3）—— 是残差比，不是模式数比。
> 原因见 §13.2 的 ⚠ 段：约束最优解压带边、换一条更胖的尾巴。
这就是把原来那个错误目标

$$
\text{PSWF}\Rightarrow\text{long range disappears}
$$

换成正确目标

$$
\boxed{\text{optimized closure}\Rightarrow\text{long-range correction needs fewer modes}}
$$

之后，PSWF 的价值**重新出现且可量化**。⚠ 这是量级估算，不是实测；本项目的教训 #8
是「看代码推机制，四次都错」——**要引用必须先测**。

### 13.4 规模：几百个模式，不是 3D mesh

$L=6.805$ nm（由实测 $k_{\min}=0.9234$ nm⁻¹ 反推），$k_c=2\pi/r_c=5.236$ nm⁻¹：

$$
|\mathbf n|_{\max}=k_c/k_{\min}=5.671
$$

**精确整数计数**（$0<|\mathbf n|\le5.671$）：**750 个格矢，独立模式 375 个**
（$\rho(-\mathbf k)=\rho^*(\mathbf k)$）。

成本对照（32794 原子、95.6 原子/nm³）：

| | 量 |
|---|---|
| 一趟 $r_c=1.2$ nm pair 遍历 | 每原子 ~692 邻居 ⇒ **~1.13e7 对** |
| 375 个模式的直接 structure factor | **~1.23e7 个原子×模式乘积** |

⇒ 只能说 **interaction-count scale comparable to one neighbor-pair traversal**
（相互作用**计数**同量级），**不能说「计算成本同量级」**（2026-09-07 更正）：
一个 atom-mode 项要做 phase / sin-cos / 复数累加 / 施力，与一个**已经有邻居表**的
pair 操作常数完全不同。**375 modes 的 naive 实现很可能远贵于 1.13e7 个 pair 相互作用。**
量级估算 ≠ 成本估算，**未实测前不要引用为性能依据**（本项目性能估算已错过四次）。

* **两趟结构**：① 累加 $\rho(\mathbf k)$（对原子 reduce）；② 施力（原子 × 模式）。
  这是 Ewald 倒空间求和的标准形态，只是没有 Gaussian 屏蔽、换成窗 $\hat c$。
* **⭐ MTS：candidate，必须实测，不能假定**（2026-09-07 更正）。
  对**固定电荷**，$\rho_{\mathbf k}(t)=\sum_iq_ie^{-i\mathbf k\cdot\mathbf r_i(t)}$
  在低 $k$ 确实天然慢。**但 CWLD 是 $Q_i(t)=Q_i[\mathrm{dens}_i(t)]$**，于是即使 $k$ 很小，

  $$\rho_{\mathbf k}(t)=\sum_i Q_i(t)\,e^{-i\mathbf k\cdot\mathbf r_i(t)}$$

  也可能因**局部水重排让 $Q_i$ 快速变化**而含高频成分。
  ⇒ **「low $k$ ⇒ slow force」对 CWLD 不自动成立**，
  §7.4 那个「$\kappa$ 在 ns 尺度才漂」只说明**位置**的长波演化慢，不覆盖电荷响应。
  正确表述：*MTS candidate; low spatial frequency suggests slow positional evolution,
  but CWLD charge-response dynamics may introduce fast components and must be measured.*

  **验证很便宜**：存 $E_{\rm low\text{-}k}(t)$、$\mathbf F_{\rm low\text{-}k}(t)$、
  $\lambda_i^{\rm low\text{-}k}(t)$，做时间自相关 / 功率谱，再决定 2/4/8 fs 的 outer step
  能不能用。**这一步没做之前，不要把 MTS 写成路线 B 的优势来源。**
* 概念上的差别值得写进论文：不是「real-space + huge reciprocal mesh」，而是
  **local electrostatics handles intermediate/high $k$;
  explicit low-rank Fourier modes only repair nonlocal physics.**
  原 Nature Communications 那条线主要省 FFT/grid；这里可以**只显式维护低阶 Fourier
  modes、完全不用 FFT**。

### 13.4b 两个阻塞项（开工前必须解决，H2 比 H1 更基础）

> **H2 — response-consistent reciprocal correction（2026-09-07 用户提出，列为最高阻塞项）**
>
> **倒空间能量里所有对 $Q_i$ 的依赖 —— pair 项、self 项、背景约定三处 —— 必须全部进入
> $\lambda_i=\partial E/\partial Q_i$，否则 CWLD 的 chain force 不是这个 Hamiltonian 的导数。**
>
> 失效模式极其隐蔽：**能量看起来补对了（带内精确恢复 $4\pi/k^2$），力却不守恒。**
> 这类错误不会让轨迹立刻炸，只会让 NVE 慢慢漂 —— 而我们刚花了三轮才知道
> 那个通道在 fp32/短窗下**根本测不出来**（`PROJECT_STATUS` §9.3-R/R2）。
>
> **先说判据本身，再列通道 —— 因为列表会漏，判据不会**（2026-09-07 收紧）：
>
> $$\boxed{\ \lambda_i=\frac{\partial U_{\rm total}}{\partial Q_i}\ \text{，其中 }U_{\rm total}\ \text{是完整能量，含每一个「修正项」}\ }$$
>
> **枚举方式必须是「照着能量表达式逐项求导」，不是「把项分成物理项和修正项」。**
> 前三个通道我列过、第四个是 peer 补的、而两次遗漏都出自同一个心理动作：
> **在代码里它看起来像个 correction，于是没人对它求导。**
>
> | # | 通道 | $\partial/\partial Q_i$ | 为什么容易漏 |
> |---|---|---|---|
> | 1 | **reciprocal pair** $\tfrac1{2V}\sum_{\mathbf k}\tfrac{4\pi}{k^2}\hat c\,\|\rho_{\mathbf k}\|^2$ | $2\,\mathrm{Re}[e^{-i\mathbf k\cdot\mathbf r_i}\rho^*_{\mathbf k}]$ | 唯一"显然"的一个 |
> | 2 | **reciprocal self** $-\tfrac1{2V}\sum_{\mathbf k}\tfrac{4\pi}{k^2}\hat c\sum_iQ_i^2$ | $\propto 2Q_i$ | 固定电荷下是常数、对力零贡献 ⇒ PME 的习惯就是不管它，**这个习惯在 CWLD 下直接错** |
> | 3 | **背景 / 中性化约定**（丢 $\mathbf k=0$ 隐含的中性背景） | $\propto 2\sum_iQ_i$ | 同上；**且恰好在 H1 被违反时非零** ⇒ 背景项就是 H1 与 H2 的耦合处 |
> | 4 | **排除对扣除** $-\sum_{(i,j)\in\rm excl}Q_iQ_jS_{\rm band}(r_{ij})$（§13.5-2） | $-\sum_{j\in\rm excl(i)}Q_jS_{\rm band}(r_{ij})$ | **peer 2026-09-07 补，我已独立验过导数**。它是**实空间**项，所以不在"倒空间能量"这个说法覆盖的范围里 —— 但失效方式完全相同。1AAY 有 **43338 个** bonded exclusions，量不小 |
>
> ⚠ **一个反例，专门记下来防止「顺手一起修」**：现有 CWLD 的 RF 抵消项
> $-q_{{\rm base},i}q_{{\rm base},j}(1/r-1/r_c)$ 用的是 **qbase 不是 $Q$**
> ⇒ $\partial/\partial Q_i=0$，**它确实不进 $\lambda_i$**。
> 它长得最像该进的那一项，却是唯一不进的。
>
> 另外两项本来就已经在规格里，一并列全以便机械核对：
> Q penalty（§18.6，启用时进 $\lambda_i$）、以及若加分子级中性约束（§13.5-4）也进 $\lambda_i$。
>
> 因此力必须写成
>
> $$\frac{dE}{d\mathbf R}=\Bigl.\frac{\partial E}{\partial\mathbf R}\Bigr|_Q
> +\sum_i\underbrace{\frac{\partial E}{\partial Q_i}}_{\lambda_i^{\rm recip}}
> \frac{\partial Q_i}{\partial\mathbf R},
> \qquad
> \boxed{\lambda_i=\lambda_i^{\rm real}+\lambda_i^{\rm low\text{-}k}}$$
>
> 再进 dens chain（$g_i=\lambda_i\,dQ_i/d\,\mathrm{dens}_i$）。
> ⚠ 具体 normalization 要按最终的 $\pm\mathbf k$ 约定定；**形式在此定死，数值实现前独立复核一遍。**

> **H1 — 净电荷 $\sum_iQ_i\neq0$**（原 §13.5-4）：倒空间求和丢掉 $\mathbf k=0$ 等价于假设
> 中性背景，而 CWLD 的 $\Delta q$ 实测系统性同号（每水 $+8\times10^{-4}$ e）。
> 必须先做完 `PROJECT_STATUS` §9.5-1 的 $\sum_i\Delta q_i$ 直接测量。
> **注意 H1 与 H2 相扣**：中性化背景项本身就是 H2 的第三个通道。

⚠ **B0 已 PASS 不覆盖这两项**：B0 用的是严格中性的合成体系，且只验了
$w(k)=\hat c(kr_c)$ 的**形式与系数**，没有 CWLD 的 $Q(\mathrm{dens})$ 响应。

### 13.5 五条实现约束（每一条都对应本项目已经踩过的坑）

1. **$U_{\rm low\text{-}k}$ 必须进 $\lambda_i$**（= H2，已升级为阻塞项，见 §13.4b）。 CWLD 的 $Q_i$ 依赖 dens，所以
   $\partial U_{\rm low\text{-}k}/\partial Q_i$ 会通过
   $g_i=\lambda_i\,dQ_i/d\,\mathrm{dens}_i$ 产生**额外的链式响应力**：

   $$
   \frac{\partial|\rho(\mathbf k)|^2}{\partial Q_i}
   =2\,\mathrm{Re}\bigl[e^{-i\mathbf k\cdot\mathbf r_i}\rho^*(\mathbf k)\bigr]
   $$

   ⚠ **只加能量、漏掉这一项就是错的实现**——与 §18.6 对 Q penalty 的要求同一类错误。
   （系数按标准约定推出后**实现前必须独立复核一遍**，这里只定形式不定数值。）
2. **排除对必须在实空间扣掉。** 倒空间求和无条件包含全部对，包括 43338 个 bonded
   exclusions；必须减掉 $\sum_{(i,j)\in\rm excl}Q_iQ_j S_{\rm band}(r_{ij})$，
   $S_{\rm band}$ 是带限反变换（做法与 PME 用 `erf` 扣排除对完全同构）。
   **这正是 $E_F$ 判据当年翻车的同一个坑**（§11.3：全对求和 vs 非排除对）。
3. **self 项**：$S(r\to0)=\chi(0)/r_c=c_\ell/r_c$ 有限，需扣
   $-\tfrac12(c_\ell/r_c)\sum_iQ_i^2$ 的带限对应量。
   ⚠ **它对 CWLD 不是常数**（$\partial Q_i^2/\partial Q_i=2Q_i$）⇒ 进 $\lambda_i$，见 H2。
4. **净电荷 $\sum_iQ_i\neq0$ 是前置阻塞项。** 倒空间求和丢掉 $\mathbf k=0$ 等价于假设
   中性背景，而 CWLD 的 $\Delta q$ 实测系统性同号
   （每水 $+8\times10^{-4}$ e，`PROJECT_STATUS_2026-09-07.md` §3.4.3）。
   ⇒ **必须先做完 §9.5-1 那个 $\sum_i\Delta q_i$ 的直接测量**，
   确认总净电荷量级，再决定要不要在 $Q$ 上加分子级中性约束
   （注意约束本身也会进 $\lambda_i$）。
5. **不要用 $E_F$ 当验收门。** 同 §12.2。

### 13.6 分级 kill switch

| 关 | 判据 | 不过怎么办 |
|---|---|---|
| B0 | 纯 numpy 单帧：$U_{\rm local}+U_{\rm low\text{-}k}$ 对**精确 Ewald** 的逐原子力误差，随 $k_c$ 单调下降并在 $k_c=2\pi/r_c$ 处显著优于纯 closure | 形式或系数推错，回 13.1 |

**B0 = PASS（2026-09-07，`lips-hybrid-lowk`）**：$\hat L=(4\pi/k^2)(1-\hat c)$ 的恒等式逐点
对到 1.6e-12；375 个独立模式把 rel_rms 力误差从 14–18%（纯 closure）压到
0.19–1.06%（13.6×–90.5×）；$k_c\to60$ 单调收敛到精确 Ewald。
⚠ **合成体系上验不到的两条**：排除对扣除（§13.5-2，合成体系没有 bonded exclusion）
与 self 项（§13.5-3，与坐标无关 ⇒ 对力零贡献）—— 这两条**仍然待做，必须换真体系**。
§13.5-1 的 $\lambda_i$ 链式响应力也完全没碰。
| B1 | 实测 $S_{ZZ}(k_{\min})$ 回到 PME 量级（当前差 22–45×） | 补的模式不够 / $w(k)$ 错 |
| B2 | 静态力审计不恶化；NVE 漂移不恶化 | 带限反变换的实空间扣除有问题 |
| B3 | 端到端速度：相对纯 CWLD 插件的额外成本 ≤ 一趟 pair 遍历（含 MTS 后） | 回 13.4 的两趟结构做归因（**单变量探针**） |

**B0 是最便宜的一关，也必须最先过**：纯 numpy、单帧、零 MD。

### 13.7 与路线 A 的关系

$$
\text{路线 A}=\text{路线 B 的 real-space 前半截}
$$

所以两者**不是二选一**：先做 A 的 kill-switch（便宜），
它的产物（一个在有限 $k$ 上更好、且局域力不差于 zmm1 的窗）直接就是 B 里的
$U_{\rm local}^{\rm optimized}$。而 B 的存在反过来**放宽了 A 的要求**——
带内既然会被显式修正，A 就不必再去纠结带内，只需管
「局域力精度 + 带外残差」这两件事（§13.2 的分解）。

---

## 14. Prolate Closure Family (PCF) —— closure 层的**主表述**

> ⚠ **本节的定位在 2026-09-07 被更正过一次，因果曾被写反。**
> 第一版把 PCF 摆成「路线 C」、一个在精度路线失败之后的退路。**不对。**
> 用户的立场自始就是「**PSWF 是一种比 ZMM 优雅得多的表达式**」——
> 也就是 §2 目标 1（「把 closure 抽象成窗层、ZMM 成为该层的一个实例」）**本身**，
> 而不是它的备选。精度测试（力审计）是一次**旁证检查**，它返回了负结果，
> 而那个负结果**不触及这个目标** —— 目标从来不是「更准」。
> ⇒ 正确的读法：**PCF 是 §2 目标 1 的完成形态**（ZMM 从「一个实例」进一步变成
> 「$c\to0$ 的退化极限」），不是任何东西的退路。

> 2026-09-07，用户提出。**动机是「ZMM 那个低阶多项式太丑，想要一个数学上更漂亮、
> 可系统调参的 family」，它明确 *不* 背负「解决长程静电」的包袱**（那本来也不是本计划的目标，
> 见 §11 的更正框）。卖点就一句：**a spectrally tunable, smooth, compact real-space closure。**
>
> **四条可精确陈述的收益**（不是审美修辞）：
> 1. **两个物理角色分开了**：$\ell$ 定 $r_c$ 处正则性（⇒ 谱衰减指数 $q^{-(\ell+1)}$），
>    $c$ 定带内形状。**ZMM 里无法只动其中一个** —— 想提高 cutoff 处光滑度就被迫
>    接受一个特定谱形状。这是「参数化选对了坐标系」。
> 2. **ZMM 是极限点而非例外**：每条 ZMM 陈述自动成为 PCF 的特例。
> 3. **来源从「解端点方程」升级为「变分原理 + 端点因子」。**
> 4. **这次重写已经付过一次帐**：正是为写 PCF 才发现「ZMM = 满足端点条件的最低次多项式」
>    **是错的**（$(1-x)^\ell$ 次数更低），必须补「$\chi$ 为偶」才成立（§3.5）。
>    ⇒ 换表述本身修掉了一个真实的理解漏洞。
>
> **而且是免费的**：B1 表示（$\psi_0^c$ 拟合成偶次多项式，残差 4e-15）让 Lepton 串 /
> 插件系数表 / 解析导数**一个字节都不用改**（§14.7）。
>
> **副产品：族给了陈述结果的坐标系。** 今天那张力审计表可以写成
> 「**在局域力精度判据下，族内最优位于 $(\ell{=}1,\,c\to0)$**」，
> 比「试了 zmm1/2/3 与 pswfz2，zmm1 赢」强得多 ——
> 而「最优落在退化边界上」本身就有信息量（局域力精度偏好最低正则性、最宽谱）。

### 14.1 定义

$$
\boxed{\;\chi_{\ell,c}(x)=\frac{(1-x^2)^\ell\,\psi_0^c(x)}
{\displaystyle\int_0^1(1-t^2)^\ell\,\psi_0^c(t)\,dt}\;,\qquad 0\le x\le1\;}
$$

其余照旧：$A(x)=\int_0^x\chi_{\ell,c}$，$U_{\ell,c}(r)=\bigl(1-A(r/r_c)\bigr)/r$。
$\psi_0^c$ 是带宽 $c$ 的零阶 prolate（PSWF），即余弦集中算子

$$
K_c(x,y)=\int_0^c\cos(qx)\cos(qy)\,dq=\tfrac12\Bigl[\tfrac{\sin c(x-y)}{x-y}+\tfrac{\sin c(x+y)}{x+y}\Bigr]
$$

的最大特征函数（`closure.py::pswf_window` 已经在算这个，只是没乘 $(1-x^2)^\ell$）。

### 14.2 为什么这个形式比 ZMM 好看：两个参数的职责终于分开了

$$
\underbrace{(1-x^2)^\ell}_{\text{real-space regularity}}\;\cdot\;
\underbrace{\psi_0^c(x)}_{\text{spectral shaping}}
$$

而且这个分工**可以精确陈述**（用 §3.5 的偶性论证 + `PROJECT_STATUS_2026-09-07.md` §3.6 的渐近）：

| 参数 | 控制什么 | 精确表述 |
|---|---|---|
| $\ell$ | cutoff 正则性 + 高 $q$ 衰减阶 | $\chi$ 在 $x=1$ 的零点阶 ⇒ **envelope** $\hat c(q)=O\bigl(q^{-(\ell+1)}\bigr)$ |
| $c$ | 有限 $q$ 形状 + 渐近系数 | 不改变 $x=1$ 的零点阶 ⇒ 只改**前导系数**与带内形状 |

⚠ **必须写成 envelope，不能写成逐点**（2026-09-07 用户指出）。准确表述：

> **For finite $c$ with $\psi_0^c(1)\neq0$, $c$ does not change the asymptotic envelope
> order $O(q^{-\ell-1})$; it changes the leading coefficient and the finite-$q$ spectral shape.**

理由：$\hat c$ 含 $\sin q/\cos q$ 振荡因子，**在某些离散 $q$ 上前导项本身可能恰好为零**。
说「指数」时指的是包络，不是每一个点。
（peer 实测的斜率 −2.000 / −2.952 / −4.000 正是**包络**斜率，与此一致。）

（$\psi_0^c(1)\neq0$——prolate 在端点不为零，这正是它**不能裸用**的原因；
乘上 $(1-x^2)^\ell$ 之后端点条件全部恢复。$\psi_0^c$ 作为 $[-1,1]$ 上的偶函数
在 $[0,1]$ 上有 $\psi'(0)=0$，所以偶性前提也满足 ⇒ 原点侧不掉一阶。）

### 14.3 定义性质：$\mathrm{ZMM}_\ell=\mathrm{PCF}_{\ell,0}$

$$
\boxed{\ \mathrm{ZMM}_\ell=\mathrm{PCF}_{\ell,\,0}\ }
$$

**这不是一条观察，是 PCF 的定义性质**（2026-09-07 用户升格）。它让整篇叙事换一个层次 ——
不需要说「we replace ZMM with PSWF」，而是：

> **We embed the conventional ZMM closures into a spectrally tunable prolate closure
> family, with ZMM recovered exactly in the zero-bandwidth limit.**

两种写法给出同一个极限（取哪个算子看实现，`closure.py` 用的是后者）：

| 算子 | $c\to0$ 的退化 |
|---|---|
| $\mathcal F_c[\phi](x)=\int_{-1}^{1}\phi(t)e^{icxt}dt$（ESP 的定义） | $e^{icxt}\to1$ ⇒ **rank-one 常数算子** ⇒ 最大特征函数 → const |
| $K_c(x,y)=\int_0^c\cos(qx)\cos(qy)dq$（本项目余弦集中核） | $\sin(cu)/u\to c$ ⇒ $K_c\to c$（常数核，秩 1）⇒ 同上 |

于是 $\psi_0^c\to\text{const}$，$\chi_{\ell,c}\to c_\ell(1-x^2)^\ell=\chi_\ell^{\rm ZMM}$。

**C0 回归测试就是这条性质的工程化**：$c\to0$ 必须逐位/机器精度地还原冻结的 ZMM 系数
（peer 实测差 ≤1.8e-13）。

✅ **别名已实现（2026-09-07，peer）**：`pcf_window(ell, 0)` 在 `c==0` 上短路，
**直接返回冻结的 `zmm_window(ell)` 对象本身** —— 所有冻结断言与逐位相同的 Lepton 串
一个字节未动，小 $c$ 的条件数问题也就不存在。`pcf{1,2,3}_c0` 已注册为显式 registry 键。
别名**故意不改 `.name`**（仍是 `zmm{ell}`），因此它不可能在报告里伪装成一个独立的窗；
而 `force_audit` 按 CLI 字符串给行打标签，所以 `--kernels pcf1_c0,pcf1_c2,…`
在 CSV 里读起来是 PCF、跑的是冻结路径。已核：系数与 lepton 逐位相同。

⇒ **「族内最优位于 $(\ell{=}1,c\to0)$」现在是一句可执行、可复现的话**，不再只是散文。

#### 14.3b 三个参数的职责（hybrid 下才完整）

$$
\boxed{\ \underbrace{\ell}_{\text{local regularity}}\ ,\qquad
\underbrace{c}_{\text{spectral partition}}\ ,\qquad
\underbrace{k_c}_{\text{exact reciprocal bandwidth}}\ }
$$

hybrid 内：

$$
k\le k_c:\quad \hat v_{\rm local}+\hat v_{\rm recip}=\frac{4\pi}{k^2}
\quad\textbf{（与 closure 选择无关）};\qquad
k>k_c:\quad \delta\hat v(k)=\frac{4\pi}{k^2}\hat c_{\ell,c}(kr_c)
$$

⇒ **只有在这里，把 $\hat c_{\ell,c}$ 往 correction band 内压才真的转化为「更少 reciprocal
modes」**；而在纯 cutoff 下，同一个动作换来的是「谱判据好了 13.7×、力反而差 3–14%」那个尴尬
（§14.6 实测）。

**整条逻辑因此是**：

$$
\text{ZMM}\subset\text{PCF}\ \longrightarrow\
\text{spectral partition designer}\ \longrightarrow\
\text{minimal low-}k\text{ exact correction}
$$

### 14.3c $c\to0$ 极限的推导细节

$c\to0$ 时 $K_c(x,y)\to c$（常数核，秩 1）⇒ 最大特征函数 $\psi_0^c\to$ const ⇒

$$
\chi_{\ell,c}(x)\;\xrightarrow{c\to0}\;c_\ell(1-x^2)^\ell=\chi_\ell^{\rm ZMM}
$$

⇒ **ZMM1/2/3 = PCF$(\ell,c\to0)$，是 family 的极限成员。**
所以 PCF 是 ZMM 的**严格推广**，叙事方向也就反过来了：
不是「PSWF 修补 ZMM」，而是「ZMM 是 PCF 的退化 baseline」。

### 14.4 命名：不要再叫 `pswfz2`

`pswfz2` 听起来像「PSWF 修补 ZMM2」。若要当方法写论文：

* **Prolate Closure Family (PCF)**，或
* **PSWF-regularized finite-range Coulomb closure**

ZMM1/2/3 只作 comparator。论文里的叙事句（用户原话，值得照抄）：

> Conventional ZMM closures correspond to low-order polynomial windows fixed entirely by
> endpoint conditions. We generalize the closure to a PSWF-modulated family, separating
> cutoff smoothness from spectral shape.

⚠ **改名的纪律**（这仓库被静默改名坑过，见 `MATH_CHANGE_MAP.md`）：
`WINDOW_REGISTRY` 里的键 `"pswfz2"`、常量 `FROZEN_PSWFZ2_CHI_EVEN` 及其冻结断言、
环境变量取值 `L_IPS_CLOSURE=pswfz2`、以及各报告里引用的这个名字
**一律保留为别名，不要重命名**；新名字**新增**注册（建议 `pcf{ell}_c{c}`）。

### 14.5 必须说清的一件事：PCF **不是**约束下的最优窗

这两个是不同的对象，**论文里不能混**：

| 对象 | 是什么 | 长处 | 短处 |
|---|---|---|---|
| `pswfz2`（现有） | **约束最优**：在 $\chi=(1-x^2)^2P(2x^2-1)$ 内解 $\min\int_{q>c}\hat c^2$ s.t. $\int\chi=1$ | 该约束类里带外能量真的最小（对 zmm2 好 13.7×） | 7 个冻结浮点系数，**难看**，没有 family 结构 |
| **PCF$(\ell,c)$**（本节） | **PSWF-调制**：直接乘 $(1-x^2)^\ell$ 再归一 | 两参数、职责分离、ZMM 是极限、形式漂亮 | **乘 $(1-x^2)^\ell$ 破坏了 $\psi_0^c$ 的最优性** ⇒ 同 $\ell$ 下带外集中度**必然不如** `pswfz2` |

⇒ **PCF 可以说「更一般、更漂亮、可系统调参」，不能说「最优」。**
（这正是本项目连续三轮在收紧的那类越界表述。）

### 14.6 现实预期：纯 closure 下优化器很可能选 $c\to0$

> ✅ **2026-09-07 实测证实，见 `../reports/PROJECT_STATUS_2026-09-07.md` §7.3-R。**
> 1AAY 上 9 个 kernel 的静态力审计：$\ell=1$ 一族随 $c$ **单调变坏**
> （$c\to0$=zmm1 0.926 < c=2 0.947 < c=4 1.025 < c=2π 1.125，对 zmm2 归一），
> $\ell$ 也是越低越好。**没有任何 PCF 成员打得过 zmm1。**
> ⇒ 本节的预测**成立**，且它是可被打脸的（若某个 $c>0$ 更好，预测就死了）。
> **PCF 在纯 closure 上没有精度收益，收益只可能在 §13 的 hybrid 下。**

静态力审计的既有事实：**带外集中度更好的 `pswfz2` 在局域力上每个通道差 zmm2 3–15%**。
PCF 在 $c\to0$ 端就是 ZMM。所以在**纯 cutoff** 判据下，
$c$ 这个旋钮**大概率会被优化器拧回 0**——也就是拧回 ZMM。

**这不是坏消息，是把两条路线接起来的地方**：

$$
\text{§13 的 hybrid 让带内变精确} \;\Rightarrow\;
\text{带外残差成为唯一的长程误差} \;\Rightarrow\;
c\ \text{这个旋钮才开始值钱}
$$

> ⚠ **2026-09-07 实测修正**（`../reports/REPORT_HYBRID_LOWK_B0_2026-09-07.md` §6）：
> 「优化器把 $c$ 拧回 0」**只在 $\ell=3$ 成立**。$\ell=1$ 与 $\ell=2$ 有真的**内点最优**
> （$c\approx4$、$c\approx2.5$），把带外能量压到 zmm2 的 0.163 / 0.234（6.1× / 4.3×）。
> 所以 $c$ 这个旋钮在低 $\ell$ 上确实值钱。
> **但本节的结论不变**：hybrid 主表里 `pcf1_c4` 虽在 $k_c\ge10$ 处优于 pswfz2
> （7.26e-4 vs 1.12e-3），仍不如 zmm3（6.72e-4）⇒ **hybrid 下的胜负手是 $\ell$，不是 $c$。**
> 另：§14.2 的职责分离（$\ell$ 定衰减指数、$c$ 不改指数）**数值成立**，
> 包络斜率 $=-(\ell+1)$ 在 $c\in[10^{-6},2\pi]$ 上一致到 3 位。

⇒ **PCF 提供旋钮，§13 让旋钮有意义。** 若只做纯 closure、不做 §13，
那 PCF 的价值就仅限于「一个更好看、更一般的表述」——**这本身也是正当的论文贡献，
但要如实说成表述层面的贡献，不要包装成精度收益。**

### 14.6b 判读预登记：ℓ=1 的 $c$ 细扫（**在看到数据之前写的**）

> 2026-09-07。遵 DEC-005 的纪律：**先定判读，再看结果**，否则判读会被结果拟合。
> 正在跑：`lips-force-audit --kernels pcf1_c0,pcf1_c0.25,pcf1_c0.5,pcf1_c1,pcf1_c1.5,pcf1_c2`
> （peer 已 pre-flight：六个名字全解析、全过 `self_check(strict)`，
> B1 拟合残差 3.6e-15…5.2e-14，**比 C1 门槛 1e-12 低 3–4 个量级**，没有糊的行。）

**为什么只扫 0–2**：这一档问的是**力精度判据下族内最优在哪**，而现有 4 点
（$c$=0/2/4/2π）已显示 $c\ge2$ 单调变坏，所以未采样的空白只剩 $(0,2)$。

**两种结果，各自的含义（现在写死）**：

| 结果 | 结论 | PCF 的定位 |
|---|---|---|
| **存在 $c>0$ 使 rel_rms < `pcf1_c0`** | 力判据下族内最优是**内点** ⇒ **PCF 在 ZMM 自己的判据上赢了 ZMM**（$c$ 是 ZMM 里不存在的自由度） | 从「表述层面」升到「精度层面」 |
| **全部 $c>0$ 都 ≥ `pcf1_c0`** | 边界最优，证据从 4 点变 7 点 | 维持「表述层面」，§14.5/§14.6 不变 |

⚠⚠ **第二种结果不构成对 C2 的反驳 —— 这一条必须先说清**（peer 2026-09-07 提出）。
在**带外能量**判据上，ℓ=1 在同一段 $c$ 上是**单调变好**的：

$$c=0\ldots2:\quad 2.139\ /\ 2.115\ /\ 2.046\ /\ 1.797\ /\ 1.458\ /\ 1.103$$

且其最优在**区间之外**（$c\approx4\text{–}5$，值 0.163）。两个推论：

1. **$[0,2]$ 并不 bracket 带外判据的最优** ⇒ 若要找**谱**最优，得扫到 $c\approx5$。
   但那只在 hybrid 下有意义（§13，范围待裁），**不在本档问题之内**。
2. **两个判据在这一段上本来就该反向。** 所以若力判据把最优放在 $c\to0$，
   那是 §13.2 那个分工的**又一次确认**，不是 C2 出错。
   而且这次比 pswfz2-vs-zmm2 那个对照**更干净** —— $\ell$ 固定，只动形状旋钮。

⇒ **把第二种结果读成「C2 错了」或「$c$ 这个旋钮没用」，是要避免的误读。**
正确读法是：**同一段 $c$ 上，谱判据变好 1.94×、力判据变坏** ——
这是一个**正面的结构性结论**（职责分离的定量证据），比「PCF 没有精度收益」信息量更大。

### 14.6c ✅ 细扫结果：边界最优成立，**但 $c=0$ 是驻点而非端点**（2026-09-07）

产物 `closure_force_audit_ZN_1aay_cwld_ligm0p15metal_zmm_seed0_double_6kb799c8.csv`。
按 §14.6b 登记的判读：**第二种结果** —— 无内点最优，**每个通道都严格单调**
（rel_rms / water_O / water_H / protein_heavy / Zn²⁺ / shell1 全单调），$c=0$ 最小。
⇒ 「族内最优位于 $(\ell{=}1,c\to0)$」的证据从 4 点变 7 点；§14.5/§14.6 不变。

| $c$ | rel_rms | 相对 $c=0$ 劣化 | 带外能量（peer） | 相对 $c=0$ |
|---|---|---|---|---|
| **0**（=zmm1） | **4.7889e-02** | — | **2.139** | — |
| 0.25 | 4.7898e-02 | +0.019% | 2.115 | −1.1% |
| 0.5 | 4.7926e-02 | +0.077% | 2.046 | −4.3% |
| 1 | 4.8073e-02 | +0.384% | 1.797 | **−16.0%** |
| 1.5 | 4.8408e-02 | +1.084% | 1.458 | −31.8% |
| 2 | 4.8987e-02 | +2.293% | 1.103 | −48.4% |

**但真正值得记的是：两个判据在 $c\to0$ 都是 $O(c^2)$，符号相反。**
把劣化/增益除以 $c^2$，$c\to0$ 时收敛到常数：

| $c$ | 力判据 $\Delta/c^2$ | 带外 $\Delta/c^2$ |
|---|---|---|
| 0.25 | +0.00301 | −0.1795 |
| 0.5 | +0.00309 | −0.1739 |
| 1 | +0.00384 | −0.1599 |

**这有一个干净的原因**（本次汇总推导）：集中算子
$K_c(x,y)=\tfrac12\bigl[\tfrac{\sin c(x-y)}{x-y}+\tfrac{\sin c(x+y)}{x+y}\bigr]$
可写成 $c\cdot\bigl[\ldots\bigr]$，而**整体的正标量因子 $c$ 不改变特征向量**；
括号里是 $\mathrm{sinc}(cu)=1-\tfrac{(cu)^2}{6}+\cdots$，**对 $c$ 是偶的**。于是

$$\psi_0^c=\text{const}+O(c^2)\quad\Rightarrow\quad
\chi_{\ell,c}=\chi_\ell^{\rm ZMM}+O(c^2)$$

⇒ **$\chi$ 的任何光滑泛函在 $c=0$ 处一阶导数为零。**
所以 $c=0$（=ZMM）**不只是采样区间的端点，而是每个光滑判据的驻点** ——
力判据在那里取**极小**、带外能量在那里取**极大**。数值上两个 $\Delta/c^2$ 都收敛，正是这条。

$$\boxed{\ \text{ZMM 位于一个驻点：族内**局域力最优**同时**谱最劣**，两侧都是 }O(c^2)\ }$$

**这把「职责分离」从经验观察变成了结构事实**，而且给出一个可用的兑换率：

$$\frac{|\text{带外增益}|}{\text{力劣化}}\ \xrightarrow[c\to0]{}\ \frac{0.180}{0.0030}\approx\mathbf{60}
\qquad(\text{在 }c=1\ \text{上是 }16.0\%/0.384\%\approx\mathbf{42})$$

⇒ **拧 $c$ 在力精度上极便宜、在谱上很有效。**
这**不改变** §14.5/§14.6 的结论（纯 closure 下没有精度收益，最优仍在 $c=0$），
但它把那个结论从「阴性结果」变成**正面的结构结论**：
$c$ 不是"没用的旋钮"，而是**一个兑换率约 40–60:1 的旋钮，只是纯 closure 路线上没人要买它换的东西**。
而 hybrid（§13）里要买的恰恰就是带外残差 —— **这正是 PCF 的价值在 hybrid 才出现的定量形式。**

⚠ 仍然不能说 PCF 更准：兑换率有利 ≠ 在纯 closure 判据下更优。§14.5 的边界照旧。

### 14.7 实现：一处小改 + 一个必须先定的表示决策

**(a) 窗层**（约 10 行，`closure.py`）：取 `pswf_window(c)` 的节点值 → 乘 $(1-x^2)^\ell$ →
重新归一化 $\int_0^1\chi=1$ → 包进 `NodalWindow(ell=ℓ)`。
`self_check(strict=True)` 应当**全过**（$\chi(1)=0$ 由因子保证、归一化由构造保证、
$\chi'(0)=0$ 由偶性保证）——而裸 `pswf` 那个是 FAIL 的，正好构成对照。

**(b) ⚠ 表示决策（阻塞项）：$U_{\ell,c}$ 不是多项式 ⇒ 现有下游全部接不上。**
Lepton 串（实现 #1）、插件的 closure 系数表（#5）、解析 $dU/dr$（#4）
**都假设 $\chi$ 是偶次多项式**。三个选项：

| 选项 | 做法 | 代价 |
|---|---|---|
| **B1（推荐）** | 把 $\psi_0^c$ 用**偶次多项式拟合**（$x^2$ 的 Chebyshev/Legendre，拟合到 ~1e-14），PCF 表示成 $(1-x^2)^\ell\times P_{\rm fit}(x^2)$ | **下游一个字节不用改**：仍是偶多项式 ⇒ Lepton 串、系数表、解析导数、偶性、端点条件、$q^{-(\ell+1)}$ 衰减全部保持 |
| B2 | 把 $U$ 和 $dU/dr$ 做成 `Continuous1DFunction` 表（像 `density_kernel` 那样） | 引入插值误差进**力**（DEC-002 已记录 spline 在支撑边界的伪影），且插件 kernel 要改 |
| B3 | 只在插件路径支持 | 失去 CustomGB 这个 oracle，验收无从做（DEC-005 的 oracle 就是它） |

#### 14.7b ⚠ 更正：PCF 上 GPU 比本节原先写的小得多（2026-09-07）

原文（以及我给用户的说法）是「插件只实现 ZMM ⇒ PCF 必须走 `customgb`，慢 3.1–3.6×」。
**查过源码，这个结论错了。** 实际结构：

| 层 | 现状 | 对 PCF 的含义 |
|---|---|---|
| CUDA kernel（`platforms/common/kernels/localCWLD.cc`） | `closure(r)=1/r+(c_0+x^2(c_2+x^2(c_4+x^2c_6)))/r_c`，**Horner 求值 4 个偶次系数** | **数学完全通用，一行不用改** |
| kernel 参数传递（`CommonLocalCWLDKernels.cpp:464`） | `addReal(pairKernel, c0)` … `c6` —— **运行时参数**，不是源码字面量 | **不需要重编译逻辑** |
| 系数来源（同文件 `:49`） | `zmmCoefficients(int order, …)`，`switch` 只认 1/2/3，其余抛错 | **唯一的闸门就在这里** |
| 公共 API（`LocalCWLDForce.h:116`） | 整数 `zmm_order` | 需要一个「直接给 4 个系数」的入口 |

⇒ **启用 PCF on GPU = 加一条系数入口（API setter + 绕过那个 switch）。
没有新 kernel 数学、没有新 pass、没有新 oracle。**

**前提：PCF 必须能用 4 个系数表示 ⇒ `deg=2`。已验（ℓ=1, c=1，对 `deg=10` 参照）**：

| deg | #系数 | `self_check(strict)` | max rel Δ$U$ | max rel Δ$dU/dr$ | 装得进 c0..c6？ |
|---|---|---|---|---|---|
| **2** | **4** | **PASS** | **7.0e-06** | **8.2e-06** | ✅ |
| 3 | 5 | PASS | 1.9e-08 | 2.3e-08 | ❌ 需加 c8（kernel 要多一项 Horner） |
| 4 | 6 | PASS | 5.2e-11 | 6.2e-11 | ❌ |

**8e-6 在现有容差里绰绰有余**：DEC-002 接受的 tabulated-vs-analytic $K(r)$ 吻合就是 ~1e-6；
DEC-005 的逐原子力验收阈值是 1.0，实测 plugin-vs-CustomGB 的 p99 是 **3.3e-3**
⇒ **降次代价比生产里已接受的引擎间差异小 ~400 倍。**

**验收协议不用新设计**：CustomGB 的 Lepton 串与插件系数**出自同一个 window 对象**，
所以 DEC-005 那套 oracle（生产 CustomGBForce，真实 1AAY，GPU 对 GPU，逐粒子分级报
`dens→Q→energy→force`）原样适用，只是喂 PCF 窗而不是 ZMM 窗。
冻结 fixture 也不受影响 —— 只要 `zmm_order` 那条默认路径不动。

⇒ **`customgb`（41 ns/day）不再是唯一选择：走插件是 212 ns/day，同一个矩阵便宜 5 倍。**
⚠ 但这仍是一次**对 C++ 插件的真实改动**，而插件当前状态是「已完成、无待办」。
要做就按 DEC-005 的纪律：**先定验收阈值，再动代码**。

**B1 之后 PCF 与 `pswfz2` 落在同一个表示类里**（都是 $(1-x^2)^\ell\times$ 偶多项式），
差别只在系数是**拟合 prolate** 还是**解约束最优问题**——
于是 §12.1 那个「直接优化系数 $a_{2m}$」的参数化把两者都包住了。
**这也说明：PCF 的价值在「定义与叙事」，不在「表示能力」。**

### 14.8 kill switch（全部纯 numpy，秒级，零 MD）

| 关 | 判据 | 不过怎么办 |
|---|---|---|
| C0 | `self_check(strict=True)` 全过；$c\to0$ 时**逐位**退化到 ZMM 冻结系数 | family 定义或归一化写错 |
| C1 | B1 的多项式拟合残差 $\le$ 1e-12（否则动到 fixture 的 1e-12 断言） | 提高拟合次数，或改走 B2 |
| C2 | 带外能量对照表：PCF$(\ell,c)$ vs `pswfz2` vs ZMM$\ell$ —— 确认 §14.5 的预期（PCF 不如约束最优） | 若 PCF 反而更好，说明 `pswfz2` 的变分解有问题，回查 |
| C3 | `lips-force-audit`：扫 $c$，看局域力误差随 $c$ 的走势 | ✅ **2026-09-07 已跑：确实单调劣化** ⇒ §14.6 成立，PCF 的收益只在 §13 的 hybrid 下 |

**C0/C1/C2 = PASS（2026-09-07，`python src/lips/engine/closure.py`）**：
C0 $c=10^{-6}$ 时对 ZMM 冻结系数差 $\le1.8\times10^{-13}$；
C1 B1 偶多项式拟合残差 $4.0\times10^{-15}$（deg=14）⇒ **下游一个字节没改**；
C2 pswfz2 全表最优（0.0739），PCF 最好只到 0.163（`pcf1_c4`）⇒ §14.5 的预期成立。
**C3 仍待做**（要 `lips-force-audit`，得在真体系上跑）。
实现落在 `closure.py::pcf_window`，注册名 `pcf{ell}_c{c}`（`pswfz2` 等旧名一律保留）。

### 14.9 参考

* ESP（用户提供）：*Accelerating molecular dynamics simulations using fast Ewald summation
  with prolates*, Nat. Commun. (2026), `s41467-026-73232-8`。
  按用户的阅读，该文直接用零阶 PSWF 作 splitting kernel，$\chi(x)=\psi_0^c(x)/C_0$，
  出发点就是 PSWF 的频域能量集中最优性，用以替代传统 Gaussian Ewald splitting kernel。
  **⚠ 本条未独立核对原文。**
* **与 ESP 的关键差别（这也是 $(1-x^2)^\ell$ 因子的正当性所在）**：
  ESP **保留 reciprocal half**，所以它没有 $U(r_c)=U'(r_c)=\dots=0$ 那套纯截断约束，
  裸用 $\psi_0^c$ 没问题；**本项目要纯截断**，所以必须自己补端点正则性。
  乘 $(1-x^2)^\ell$ 恰好是把 **PSWF 的 spectral shape** 与 **ZMM 的 endpoint regularity**
  分开——这就是 §14.2 那个职责分离的来源。
