"""Stage 1 · Kill switch —— 在写任何生产代码之前判定窗族路线是否值得做。

设计来源：PLAN_PSWF_Closure_Family.md §3.5。纯 numpy，不跑 MD、不碰轨迹。

理论（文档 §3 修订版，本脚本第一段就把它数值验证一遍）
--------------------------------------------------------------------------
split  1/r = L(r) + S(r),  L=(1-A(x))/r,  S=A(x)/r,  x=r/rc

    L̂(k) = (4π rc/k)∫₀¹(1-A(x))sin(qx)dx,  q=k·rc
    分部积分 ⇒  Ŝ(k) = 4π/k² - L̂(k) = (4π/k²)·ĉ(q),  ĉ(q)=∫₀¹χ(x)cos(qx)dx

即 Fourier 空间的 split 就是 4π/k² 乘窗的余弦变换。ĉ(0)=∫χ=1 恒成立
⇒ Ŝ 在 k→0 必然发散成 4π/k²，任何 χ 改不了；截断成立靠体系屏蔽。

被丢项的均方力误差（k⁴ 与 Ŝ² 里的 k⁻⁴ 精确抵消）：

    E_F[χ] ∝ ∫₀^∞ ĉ(k·rc)² S_ZZ(k) dk

**这就是 PSWF 的能量集中问题本身**，只是权重从锐截断 |q|>c 换成物理的 S_ZZ。

取 Debye 型 S_ZZ(k)=k²/(k²+κ²)（Stillinger-Lovett 小 k 极限 + 大 k 趋 1），
令 s=κ·rc，实空间闭式：

    Ẽ_F[χ;s] = ∫₀¹χ² - (s/2)∬χ(x)χ(y)[e^{-s|x-y|}+e^{-s(x+y)}]dxdy

正是「恒等算子减平滑核」的 PSWF 结构。极限：s→0 给 ∫χ²（无屏蔽，无救），
s→∞ 给 0（强屏蔽，截断无误差）。两个极限本脚本都验。

优化问题（文档 §3.4 修订版）
--------------------------------------------------------------------------
可行空间取  χ(x) = (1-x²)^ell · Σ_{n=0..N} c_n P̃_n(x)   （P̃=移位 Legendre）
  * (1-x²)^ell 因子让 ell 个端点条件 χ(1)=..=χ^(ell-1)(1)=0 **自动满足**
  * 只剩归一化 ∫₀¹χ=1 一条线性约束
  * N=0 精确退化为 ZMM_ell  ⇒  最优解必然 ≤ ZMM，kill switch 因此是干净的

min cᵀBc s.t. aᵀc=1  ⇒  c* = B⁻¹a/(aᵀB⁻¹a),  min = 1/(aᵀB⁻¹a)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from lips.engine.closure import zmm_window, pswf_window

RC_NM = 1.2                      # 生产 cutoff

# ---------------------------------------------------------------------------
# 求积工具
# ---------------------------------------------------------------------------
# x 求积必须能分辨 cos(qx) 的振荡：n 点 GL 大约能撑到 q ~ 2n。
# 取 n=4000 ⇒ q<=400 安全（早期版本用 600 点算到 q=2e4，纯混叠噪声，
# 结果尾段占 99%、E_F 几乎不随 s 变——那是假象）。
_XG, _WG = np.polynomial.legendre.leggauss(4000)
X_UNIT = 0.5 * (_XG + 1.0)       # [0,1] 上的 GL 节点
W_UNIT = 0.5 * _WG

# ĉ(q) 的衰减指数由**端点零点阶 ell 定**，不是所有窗都一样（2026-09-07 更正）：
#   ĉ ~ q^-(ell+1)  ⇒  ĉ² S_ZZ ~ q^-2(ell+1)+2 = q^-2ell
#   ell=1 (zmm1): ĉ~q^-2, 被积 ~q^-4      <- 原注释写死的 q^-3 只对 ell=2
#   ell=2 (zmm2/pswfz2): ĉ~q^-3, 被积 ~q^-6
#   ell=3 (zmm3): ĉ~q^-4, 被积 ~q^-8
# 原注释「ĉ(q) ~ q^-3（因 χ'(0)=χ(1)=χ'(1)=0）」是 ell=2 专属的，而 zmm1 现在
# 恰好是最值得关注的窗（它同时赢静态力审计与 B0 的纯 closure 列），所以要更正。
# 实测 q>400 尾段占比（Debye s=5；head 走自适应求积、tail 走闭式递推）：
#   zmm1 1.7e-7   zmm2 1.1e-11   zmm3 1.6e-15   pswfz2 5.7e-11
# ⇒ zmm1 的尾巴比 zmm2 胖 4 个量级，但仍有 7 个量级余量，**q<=400 的截断照旧安全**。
# 最后一段专门用来显示尾部贡献确实可忽略。
Q_PANELS = [(0.0, 10.0), (10.0, 40.0), (40.0, 120.0), (120.0, 400.0)]


def _q_nodes(n_per_panel=300):
    xg, wg = np.polynomial.legendre.leggauss(n_per_panel)
    qs, ws, tags = [], [], []
    for i, (a, b) in enumerate(Q_PANELS):
        qs.append(0.5 * (b - a) * xg + 0.5 * (b + a))
        ws.append(0.5 * (b - a) * wg)
        tags.append(np.full(n_per_panel, i))
    return np.concatenate(qs), np.concatenate(ws), np.concatenate(tags)


Q_NODES, Q_WEIGHTS, Q_TAGS = _q_nodes()


def chat_of_values(chi_vals, q=None):
    """ĉ(q)=∫₀¹χcos(qx)dx，χ 以 X_UNIT 上的值给出。"""
    q = Q_NODES if q is None else np.atleast_1d(q)
    return (W_UNIT * chi_vals) @ np.cos(np.outer(q, X_UNIT)).T


# ---------------------------------------------------------------------------
# S_ZZ 模型
# ---------------------------------------------------------------------------
def szz_debye(k, kappa):
    """Debye/线性屏蔽：小 k 满足 Stillinger-Lovett k²/κ²，大 k 趋 1。"""
    return k**2 / (k**2 + kappa**2)


def szz_debye_peaked(k, kappa, k0=20.0, sigma=6.0, amp=1.5):
    """加一个分子结构峰（水的 O-O 近邻 ~0.3nm ⇒ k0~20 nm⁻¹），用于稳健性检验。"""
    return szz_debye(k, kappa) * (1.0 + amp * np.exp(-0.5 * ((k - k0) / sigma) ** 2))


# ---------------------------------------------------------------------------
# E_F：两条独立路线
# ---------------------------------------------------------------------------
def EF_kspace(chi_vals, s, rc=RC_NM, szz=szz_debye, **kw):
    """主路线：Ê = ∫₀^∞ ĉ(q)² S_ZZ(q/rc) dq（去掉公共 1/rc，只用于比值）。"""
    kappa = s / rc
    c = chat_of_values(chi_vals)
    return float(Q_WEIGHTS @ (c**2 * szz(Q_NODES / rc, kappa, **kw)))


def EF_realspace_analytic(chi_vals, s, n_grid=None):
    """交叉验证路线：Ẽ_F = ∫χ² - (s/2)∬χχ[e^{-s|x-y|}+e^{-s(x+y)}]。

    与 EF_kspace 的关系：Ẽ_F = (2/π)·Ê。

    实现是 O(n) 而非 O(n²)：
      * e^{-s(x+y)} 项是秩一，直接 (∫χe^{-sx})²
      * e^{-s|x-y|} 项化成一次因果卷积
            ∬ = 2∫₀¹χ(x)g(x)dx,  g(x)=∫₀ˣχ(y)e^{-s(x-y)}dy
        g 用逐区间精确积分（χ 取分段线性）递推，对任意 s 数值稳定；
        e^{+sy} 那种写法在大 s 下会溢出，不能用。
    早期版本用稠密核 + 固定段数：s=1000 时既要 1.7 TiB 内存，段数不够时
    还会算出负的 Ẽ_F（该量恒非负）。
    """
    if n_grid is None:
        n_grid = int(min(4_000_000, max(20_000, 400 * s)))
    x = np.linspace(0.0, 1.0, n_grid)
    h = x[1] - x[0]
    f = np.interp(x, X_UNIT, chi_vals)

    E = np.exp(-s * h)
    # 逐区间精确：∫₀^h [f1 + (f0-f1)u/h] e^{-su} du
    c1 = (1.0 - E) / s                                  # 配 f1
    c0 = (1.0 - E * (1.0 + s * h)) / (s * s * h)        # 配 (f0-f1)
    step = f[1:] * c1 + (f[:-1] - f[1:]) * c0

    # g[i] = E*g[i-1] + step[i-1] 是一阶 IIR，用 lfilter 向量化
    # （纯 Python 循环在 n~4e5 时要跑几分钟）。
    u = np.concatenate([[0.0], step])
    g = lfilter([1.0], [1.0, -E], u)

    T1 = 2.0 * np.trapezoid(f * g, x)
    T2 = float(np.trapezoid(f * np.exp(-s * x), x)) ** 2
    l2 = float(np.trapezoid(f * f, x))
    return l2 - 0.5 * s * (T1 + T2)


# ---------------------------------------------------------------------------
# 任意 S_ZZ（含实测表）的通用接口
# ---------------------------------------------------------------------------
def make_szz_from_table(k_tab, szz_tab, sl_fit_npts=4):
    """把实测 S_ZZ(k) 表变成可调用对象，并处理两端外推。

    低 k（k < k_min，盒子给不出的区域）：按 Stillinger-Lovett 用 S_ZZ=k²/κ² 外推，
      κ 由最小 sl_fit_npts 个实测点拟合（S_ZZ/k² 的均值）。
    高 k（k > k_max）：取无关联极限 S_ZZ=1。
    返回 (callable, info_dict)。
    """
    k_tab = np.asarray(k_tab, float)
    szz_tab = np.asarray(szz_tab, float)
    order = np.argsort(k_tab)
    k_tab, szz_tab = k_tab[order], szz_tab[order]
    ok = np.isfinite(k_tab) & np.isfinite(szz_tab) & (k_tab > 0)
    k_tab, szz_tab = k_tab[ok], szz_tab[ok]
    n = min(sl_fit_npts, k_tab.size)
    inv_kappa2 = float(np.mean(szz_tab[:n] / k_tab[:n] ** 2))
    kappa = float(1.0 / np.sqrt(inv_kappa2)) if inv_kappa2 > 0 else np.inf
    kmin, kmax = float(k_tab[0]), float(k_tab[-1])

    def szz(k):
        k = np.asarray(k, float)
        out = np.interp(k, k_tab, szz_tab)
        out = np.where(k < kmin, inv_kappa2 * k**2, out)
        out = np.where(k > kmax, 1.0, out)
        return out

    return szz, {"kappa_nm^-1": kappa, "k_min": kmin, "k_max": kmax,
                 "s_equiv=kappa*rc": kappa * RC_NM}


def ef_weight(szz_callable, rc=RC_NM):
    """把 S_ZZ 折进 q 求积权重：w = S_ZZ(q/rc)·W_q。之后 E_F = Σ w·ĉ²。"""
    return szz_callable(Q_NODES / rc) * Q_WEIGHTS


def EF_from_weight(chi_vals, w):
    return float(w @ chat_of_values(chi_vals) ** 2)


def optimal_window_from_weight(ell, N, w):
    """与 optimal_window 同，但权重直接给定（支持任意/实测 S_ZZ）。"""
    phi = basis_values(ell, N)
    Cq = np.array([chat_of_values(p) for p in phi])
    B = 0.5 * ((Cq * w) @ Cq.T + ((Cq * w) @ Cq.T).T)
    a = phi @ W_UNIT
    cond = float(np.linalg.cond(B))
    Binv_a = (np.linalg.solve(B, a) if cond < 1e13
              else np.linalg.pinv(B, rcond=1e-13) @ a)
    denom = float(a @ Binv_a)
    if denom <= 0:
        return np.inf, None, cond, None
    coef = Binv_a / denom
    return 1.0 / denom, coef @ phi, cond, coef


def ef_extrapolation_share(szz_callable, info, rc=RC_NM):
    """E_F 里来自两端外推区的份额（应该很小，否则结论依赖外推假设）。"""
    z2 = zmm_window(2).chi(X_UNIT)
    c2 = chat_of_values(z2) ** 2
    k = Q_NODES / rc
    w = szz_callable(k) * Q_WEIGHTS
    tot = float(w @ c2)
    lo = float((w * (k < info["k_min"])) @ c2)
    hi = float((w * (k > info["k_max"])) @ c2)
    return {"low_k_share": lo / tot, "high_k_share": hi / tot}


# ---------------------------------------------------------------------------
# 最优窗
# ---------------------------------------------------------------------------
def basis_values(ell, N):
    """φ_n(x) = (1-x²)^ell · P̃_n(x)，返回 (N+1, len(X_UNIT)) 的节点值矩阵。"""
    from numpy.polynomial import legendre as L
    phi = np.empty((N + 1, X_UNIT.size))
    env = (1.0 - X_UNIT**2) ** ell
    for n in range(N + 1):
        coef = np.zeros(n + 1)
        coef[n] = 1.0
        phi[n] = env * L.legval(2.0 * X_UNIT - 1.0, coef)
    return phi


def optimal_window(ell, N, s, rc=RC_NM, szz=szz_debye, **kw):
    """在 φ 基上解 min cᵀBc s.t. aᵀc=1。返回 (EF_min, chi_vals, cond, coef)。"""
    phi = basis_values(ell, N)
    kappa = s / rc
    wgt = szz(Q_NODES / rc, kappa, **kw) * Q_WEIGHTS
    Cq = np.array([chat_of_values(p) for p in phi])            # (N+1, nq)
    B = (Cq * wgt) @ Cq.T
    B = 0.5 * (B + B.T)
    a = phi @ W_UNIT
    cond = float(np.linalg.cond(B))
    Binv_a = np.linalg.solve(B, a) if cond < 1e13 else np.linalg.pinv(B, rcond=1e-13) @ a
    denom = float(a @ Binv_a)
    if denom <= 0:
        return np.inf, None, cond, None
    coef = Binv_a / denom
    return 1.0 / denom, coef @ phi, cond, coef


# ---------------------------------------------------------------------------
def main():
    np.set_printoptions(precision=6, suppress=False)
    print("=" * 78)
    print("Stage 1 Kill Switch —— 窗族 closure 路线是否值得做")
    print(f"rc = {RC_NM} nm；S_ZZ 主模型 = Debye k²/(k²+κ²)，s ≡ κ·rc")
    print("=" * 78)

    windows = {f"zmm{e}": zmm_window(e) for e in (1, 2, 3)}
    chi_of = {n: w.chi(X_UNIT) for n, w in windows.items()}

    # ---- 验证 1：ĉ(0)=1（Ŝ 小 k 发散不可修，文档 §3.2）--------------------
    print("\n[验证 1] ĉ(0) = ∫₀¹χ = 1 对每个窗都成立 ⇒ Ŝ(k)→4π/k² 不可修")
    for n, chi in chi_of.items():
        print(f"    {n}: ĉ(0) = {float(chat_of_values(chi, np.array([0.0]))[0]):.15f}")

    # ---- 验证 2：两条独立路线算 E_F 必须一致（Ẽ_F = (2/π)Ê）--------------
    print("\n[验证 2] k 空间 vs 实空间闭式，二者应满足 Ẽ_F = (2/π)·Ê")
    print(f"    {'window':8s} {'s':>6s} {'(2/pi)*Ê':>14s} {'Ẽ_F(实空间)':>16s} {'rel.diff':>11s}")
    worst = 0.0
    for n, chi in chi_of.items():
        for s in (1.0, 5.0, 20.0):
            e_k = 2.0 / np.pi * EF_kspace(chi, s)
            e_r = EF_realspace_analytic(chi, s)
            rel = abs(e_k - e_r) / abs(e_r)
            worst = max(worst, rel)
            print(f"    {n:8s} {s:6.1f} {e_k:14.8e} {e_r:16.8e} {rel:11.2e}")
    print(f"    → 最大相对偏差 {worst:.2e}  "
          f"{'OK（两条路线互证）' if worst < 2e-3 else '!! 不一致，推导或实现有问题'}")

    # ---- 验证 3：物理极限 -------------------------------------------------
    print("\n[验证 3] 极限行为：s→0 应给 ∫₀¹χ²（无屏蔽无救），s→∞ 应 →0（强屏蔽）")
    chi = chi_of["zmm2"]
    l2 = float(W_UNIT @ chi**2)
    for s in (1e-3, 1e-2, 1.0, 100.0, 1000.0):
        print(f"    s={s:<8g} Ẽ_F={EF_realspace_analytic(chi, s):12.6e}   (∫χ²={l2:.6f})")

    # ---- 尾部收敛性 ------------------------------------------------------
    tail = float(Q_WEIGHTS[Q_TAGS == len(Q_PANELS) - 1]
                 @ (chat_of_values(chi)[Q_TAGS == len(Q_PANELS) - 1] ** 2
                    * szz_debye(Q_NODES[Q_TAGS == len(Q_PANELS) - 1] / RC_NM, 5.0 / RC_NM)))
    total = EF_kspace(chi, 5.0)
    print(f"\n[验证 4] q∈[{Q_PANELS[-1][0]:g},{Q_PANELS[-1][1]:g}] 尾段占比 = "
          f"{tail/total:.3e}（应远小于 1，否则 q 截断不够或 x 求积混叠）")

    # ---- 主表：各窗 E_F 与最优窗 -----------------------------------------
    print("\n" + "=" * 78)
    print("主结果：均方力误差 E_F（对 zmm2 归一），以及同 ell 下的最优窗")
    print("gain_rms = sqrt(E_F(zmm2)/E_F(opt)) = RMS 力误差能降低的倍数")
    print("=" * 78)

    pswf_cache = {}
    header = (f"{'s=κ·rc':>7s} {'zmm1':>9s} {'zmm2':>9s} {'zmm3':>9s} "
              f"{'pswf*':>9s} {'opt_ell2':>10s} {'gain_rms':>9s} {'N*':>3s}")
    print(header)
    print("-" * len(header))
    rows = []
    for s in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0):
        ref = EF_kspace(chi_of["zmm2"], s)
        vals = {n: EF_kspace(c, s) / ref for n, c in chi_of.items()}

        # pswf：扫带宽 c，取该 s 下最好的（对 PSWF 最有利的读法）
        best_p, best_c = np.inf, None
        for cband in (2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0):
            if cband not in pswf_cache:
                pswf_cache[cband] = pswf_window(cband).chi(X_UNIT)
            v = EF_kspace(pswf_cache[cband], s) / ref
            if v < best_p:
                best_p, best_c = v, cband

        # 最优窗：扫 N，取收敛后的值
        best_opt, best_N = np.inf, None
        for N in range(0, 15):
            ef, _, cond, _ = optimal_window(2, N, s)
            if cond > 1e13:
                break
            v = ef / ref
            if v < best_opt - 1e-12:
                best_opt, best_N = v, N
        rows.append((s, vals, best_p, best_c, best_opt, best_N))
        print(f"{s:7.1f} {vals['zmm1']:9.4f} {vals['zmm2']:9.4f} {vals['zmm3']:9.4f} "
              f"{best_p:9.4f} {best_opt:10.4f} {np.sqrt(1.0/best_opt):9.3f} {best_N:3d}")
    print(f"\n  pswf* = 对每个 s 扫带宽 c∈[2,32] 取最优（对 PSWF 最有利的读法）")

    # N=0 必须精确等于 ZMM ---------------------------------------------------
    print("\n[验证 5] 最优化基里 N=0 必须精确复现 ZMM_ell（可行空间包含基线）")
    for ell in (1, 2, 3):
        for s in (1.0, 5.0):
            ef0, chi0, _, _ = optimal_window(ell, 0, s)
            ref = EF_kspace(chi_of[f"zmm{ell}"], s)
            print(f"    ell={ell} s={s}: opt(N=0)/zmm{ell} = {ef0/ref:.12f}")

    # ---- 稳健性：带分子峰的 S_ZZ ------------------------------------------
    print("\n" + "=" * 78)
    print("稳健性：S_ZZ 加分子结构峰（k0=20 nm⁻¹）后排序是否变")
    print("=" * 78)
    print(f"{'s':>6s} {'zmm1':>9s} {'zmm3':>9s} {'pswf*':>9s} {'opt_ell2':>10s} {'gain_rms':>9s}")
    for s in (1.0, 3.0, 8.0, 20.0):
        ref = EF_kspace(chi_of["zmm2"], s, szz=szz_debye_peaked)
        v1 = EF_kspace(chi_of["zmm1"], s, szz=szz_debye_peaked) / ref
        v3 = EF_kspace(chi_of["zmm3"], s, szz=szz_debye_peaked) / ref
        bp = min(EF_kspace(pswf_cache[c], s, szz=szz_debye_peaked) / ref
                 for c in pswf_cache)
        bo = np.inf
        for N in range(0, 15):
            ef, _, cond, _ = optimal_window(2, N, s, szz=szz_debye_peaked)
            if cond > 1e13:
                break
            bo = min(bo, ef / ref)
        print(f"{s:6.1f} {v1:9.4f} {v3:9.4f} {bp:9.4f} {bo:10.4f} {np.sqrt(1.0/bo):9.3f}")

    # ---- 最优窗的形状健康度 ----------------------------------------------
    print("\n" + "=" * 78)
    print("最优窗形状健康度（s=5, ell=2）：χ 是否病态振荡")
    print("=" * 78)
    ref = EF_kspace(chi_of["zmm2"], 5.0)
    print(f"{'N':>3s} {'E_F/zmm2':>10s} {'max|χ|':>9s} {'∫χ²':>9s} {'χ(0)':>9s} {'cond(B)':>10s}")
    z = chi_of["zmm2"]
    print(f"{'zmm2':>3s} {1.0:10.4f} {np.max(np.abs(z)):9.4f} "
          f"{float(W_UNIT@z**2):9.4f} {z[0]:9.4f} {'-':>10s}")
    for N in (0, 1, 2, 3, 4, 6, 8, 10, 12):
        ef, cv, cond, _ = optimal_window(2, N, 5.0)
        if cv is None or cond > 1e13:
            print(f"{N:3d}   (cond(B)={cond:.2e} 过大，停止)")
            break
        print(f"{N:3d} {ef/ref:10.4f} {np.max(np.abs(cv)):9.4f} "
              f"{float(W_UNIT@cv**2):9.4f} "
              f"{float(np.interp(0.0, X_UNIT, cv)):9.4f} {cond:10.2e}")

    # ---- 判定 -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("KILL SWITCH 判定")
    print("=" * 78)
    print("门槛（文档 §3.5）：最优窗须把 RMS 力误差降低 >=2 倍（E_F 比 <=0.25）")
    print("才值得进 Stage 2；否则整条路线就地终止。\n")
    for s, vals, bp, bc, bo, bN in rows:
        g = np.sqrt(1.0 / bo)
        verdict = "PASS" if bo <= 0.25 else ("边缘" if bo <= 0.5 else "FAIL")
        print(f"    s={s:5.1f}  最优/zmm2={bo:7.4f}  gain_rms={g:6.3f}x  "
              f"pswf最好={bp:7.4f}(c={bc})  -> {verdict}")


if __name__ == "__main__":
    main()
