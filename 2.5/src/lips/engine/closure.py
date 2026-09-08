"""窗函数 closure 层 —— L-IPS pair closure 的统一表示。

设计来源：PLAN_PSWF_Closure_Family.md。本模块是 Stage 0 的核心，同时被
Stage 1 的 kill switch 脚本复用。纯 numpy，**不依赖 OpenMM**。

统一正规形式（文档 §1）：

    U(r) = (1 - A(x)) / r,   A(x) = ∫₀ˣ χ(t)dt,   x = r/rc

推论：
  * A(1)=1  ⟺  U(rc)=0                          （能量在 cutoff 归零）
  * U'(rc) = -χ(1)/rc²  ⟺  力连续需要 χ(1)=0
  * 一般地 U^(m)(rc)=0, m=0..ell  ⟺  ∫₀¹χ=1 且 χ(1)=...=χ^(ell-1)(1)=0
    即 χ 在 x=1 处有 ell 阶零点。Sakuraba ZMM 的 χ_ell=c_ell(1-x²)^ell 正是
    满足这组条件的最低次多项式。

Fourier（Stage 1 用，文档 §3 修订版）：

    Ŝ(k) = (4π/k²)·ĉ(q),   ĉ(q) = ∫₀¹ χ(x)cos(qx)dx,   q = k·rc

窗族命名：zmm1/zmm2/zmm3（现有基线，绝不改动）、pswf（消融对照）、
optwin{ell}（在物理判据下最优化的窗；最终方法名 D1 待定）。
"""

from __future__ import annotations

import functools
import re

import numpy as np
from numpy.polynomial import Polynomial

__all__ = [
    "ClosureWindow", "ZMMWindow", "PolyWindow", "NodalWindow",
    "zmm_window", "pswf_window", "pswfz2_window", "pcf_window", "FrozenChiWindow",
    "make_window", "WINDOW_REGISTRY",
    "FROZEN_ZMM_POLY_COEFFS", "FROZEN_ZMM_LEPTON", "FROZEN_PSWFZ2_CHI_EVEN",
]

# --- 从 test_lips_vs_pmeV2.6.py / evaluator.py 抄来的冻结值，用作回归断言 ------
# U = 1/r + Σ_m b_m r^{2m} / rc^{2m+1}
FROZEN_ZMM_POLY_COEFFS = {
    1: (-1.5, 0.5),
    2: (-15 / 8, 5 / 4, -3 / 8),
    3: (-35 / 16, 35 / 16, -21 / 16, 5 / 16),
}

# test_lips_vs_pmeV2.6.py::zmm_uclosure 的字符串，逐字符照抄（rc 占位符为 {rc}）
FROZEN_ZMM_LEPTON = {
    1: "(1.0/r - 1.5/{rc} + r^2/(2.0*{rc}^3))",
    2: ("(1.0/r - 15.0/(8.0*{rc}) + 5.0*r^2/(4.0*{rc}^3)"
        " - 3.0*r^4/(8.0*{rc}^5))"),
    3: ("(1.0/r - 35.0/(16.0*{rc}) + 35.0*r^2/(16.0*{rc}^3)"
        " - 21.0*r^4/(16.0*{rc}^5) + 5.0*r^6/(16.0*{rc}^7))"),
}


# ---------------------------------------------------------------------------
# 带限最优窗（PSWF 判据 + ZMM 端点约束），2026-09-01 冻结
# ---------------------------------------------------------------------------
# 由 stage1 的约束变分问题解出：
#   目标  min ∫_{q>c} ĉ(q)² dq         <- PSWF 的能量集中判据
#   约束  χ = (1-x²)² · P(2x²-1)，P 为 4 次 Legendre   <- ell=2 端点条件自动满足
#         ∫₀¹χ = 1                                      <- U(rc)=0
#   带宽  c = 2π，由 cutoff 定死（q=k·rc，k=2π/rc ⇒ q=2π），不是自由参数
# 基取「x² 的 Legendre」而非移位 Legendre：后者含 x 奇次 ⇒ χ'(0)≠0 ⇒ 偶延拓在
# 原点有折点 ⇒ ĉ 只按 q⁻² 衰减（偶基是 q⁻³），既伤带限又让 U 出现 r 的奇次项。
#
# 实测对比（对 zmm2 归一）：带外能量 q>2π 为 0.0729（好 13.7 倍），
# S_ZZ 加权力误差 1.164（差 16.4%）。单精度相对误差 3.1e-8，与 zmm2 的 3.09e-8 同级。
FROZEN_PSWFZ2_CHI_EVEN = (
    2.0423602378199353, -5.989054829756399, 9.466260108590408,
    -16.898657073637075, 27.77293804110903, -25.023832573806075,
    8.629986089680179,
)


@functools.lru_cache(maxsize=32)
def _gl_unit(n: int):
    """[0,1] 上的 n 点 Gauss-Legendre 节点与权重（leggauss 不便宜，缓存住）。"""
    xg, wg = np.polynomial.legendre.leggauss(n)
    return 0.5 * (xg + 1.0), 0.5 * wg


class ClosureWindow:
    """窗函数 closure 的公共接口。子类至少要实现 chi() 和 A()。"""

    name = "<abstract>"
    ell = 0            # χ 在 x=1 处零点阶数（= U 在 rc 处的光滑阶数）
    is_polynomial = False

    # ---- 必须实现 -----------------------------------------------------------
    def chi(self, x):
        raise NotImplementedError

    def A(self, x):
        raise NotImplementedError

    # ---- 由上面两个导出 -----------------------------------------------------
    def U(self, r, rc):
        """closure 能量 U(r) = (1 - A(r/rc))/r。r 可以是数组。"""
        r = np.asarray(r, dtype=float)
        return (1.0 - self.A(r / rc)) / r

    def dUdr(self, r, rc):
        """解析导数 dU/dr = -(1-A(x))/r² - χ(x)/(rc·r)。"""
        r = np.asarray(r, dtype=float)
        x = r / rc
        return -(1.0 - self.A(x)) / r**2 - self.chi(x) / (rc * r)

    def chat(self, q):
        """ĉ(q) = ∫₀¹ χ(x)cos(qx)dx，Gauss-Legendre，**节点数随 max|q| 走**。

        ⚠ 2026-09-07 查出的坑：n 点 GL 只撑到 q≈2n（kill_switch.py 开头那条同源
        告警）。原来这里写死 400 点，q≳250 起就是**纯混叠的 O(1) 假值** ——
        ∫_{2π}^{2000}ĉ²dq 因此虚高三个量级（2.38 vs 真值 2.86e-3），而且对外层
        求积节点数完全不敏感，看起来"收敛"了。判据是「节点/振荡周期」，
        不是「外层网格够不够密」。
        """
        q = np.atleast_1d(np.asarray(q, dtype=float))
        qmax = float(np.max(np.abs(q))) if q.size else 0.0
        n = int(min(20480, max(512, 512 * np.ceil(8.0 * qmax / 512))))
        # 量化到 512 的倍数：leggauss 不便宜，不量化的话 _gl_unit 的缓存全是 miss
        x, w = _gl_unit(n)
        wchi = w * self.chi(x)
        out = np.empty(q.shape[0])
        for s in range(0, q.shape[0], 512):           # 分块，别开 Nq×n 的大矩阵
            out[s:s + 512] = np.cos(np.outer(q[s:s + 512], x)) @ wchi
        return out

    def poly_coeffs(self):
        """U = 1/r + Σ_m b_m r^{2m}/rc^{2m+1} 的 (b_0, b_1, ...)。仅多项式窗有。"""
        raise NotImplementedError(f"{self.name} 不是多项式窗，没有系数表")

    def lepton(self, rc="rc"):
        """CustomGBForce 用的 Lepton 表达式字符串。仅多项式窗有。"""
        raise NotImplementedError(f"{self.name} 不是多项式窗，无法写成 Lepton 串")

    # ---- 自检（文档 §4.1 的 5 项）--------------------------------------------
    def self_check(self, rc=1.2, strict=True, verbose=False):
        """返回 (ok, report_dict)。strict=True 时端点条件不满足即判 fail。"""
        rep, fails = {}, []

        # 这里的 600 **可以**是字面量：被积的是 χ 与 χ·x^{2m}，没有 cos/sin 振荡，
        # 所以与 chat / CHECK-1 那类「节点数必须随 max|q| 走」的约束无关。
        # 别把这个数当范例复制到有振荡的积分里去。
        xg, wg = np.polynomial.legendre.leggauss(600)
        x = 0.5 * (xg + 1.0)
        w = 0.5 * wg
        chi = self.chi(x)

        # 1. 归一化 ∫₀¹χ = 1  →  U(rc)=0
        norm = float(w @ chi)
        rep["norm_int_chi"] = norm
        if abs(norm - 1.0) > 1e-10:
            fails.append(f"归一化 ∫χ={norm:.12g} != 1（会导致 U(rc)!=0）")

        # 2. 端点条件 χ(1)=χ'(1)=...=0（力连续性）
        h = 1e-5
        chi_1 = float(np.atleast_1d(self.chi(np.array([1.0])))[0])
        dchi_1 = float((np.atleast_1d(self.chi(np.array([1.0])))[0]
                        - np.atleast_1d(self.chi(np.array([1.0 - h])))[0]) / h)
        rep["chi_at_1"] = chi_1
        rep["dchi_at_1"] = dchi_1
        rep["force_jump_at_rc"] = -chi_1 / rc**2      # dU/dr(rc)
        if abs(chi_1) > 1e-9:
            msg = (f"χ(1)={chi_1:.6g} != 0 → 力在 cutoff 跳变 "
                   f"dU/dr(rc)={-chi_1/rc**2:.6g}，NVE 会漂")
            (fails if strict else rep.setdefault("warnings", [])).append(msg)

        # 3. 矩（备查，不判定）
        rep["moments"] = {f"m{m}": float(w @ (chi * x**(2 * m)))
                          for m in range(4)}

        # 4. dUdr vs 中心差分。注意 U'(rc)=-χ(1)/rc² 对 ell>=1 趋于 0，
        #    纯相对误差在 rc 附近没有意义，故用混合绝对/相对判据。
        rr = np.linspace(0.15 * rc, 0.95 * rc, 41)
        hh = 1e-6 * rc
        fd = (self.U(rr + hh, rc) - self.U(rr - hh, rc)) / (2 * hh)
        an = self.dUdr(rr, rc)
        scale = np.maximum(np.abs(an), 1.0 / rc**2)
        err = float(np.max(np.abs(fd - an) / scale))
        rep["dUdr_vs_fd_max_mixed"] = err
        if err > 1e-6:
            fails.append(f"dUdr 与中心差分不一致，max mixed err={err:.3e}")

        # 5. r→0 前导奇异性保持：r·U = 1-A(x)，而 A(x)≈χ(0)x，
        #    所以 |rU-1| 本来就是 O(x)；判据必须随 x 缩放，否则是假失败。
        x_small = np.array([1e-6, 1e-5, 1e-4])
        r_small = x_small * rc
        rU = r_small * self.U(r_small, rc)
        chi0 = float(np.atleast_1d(self.chi(np.array([0.0])))[0])
        rep["chi_at_0"] = chi0
        rep["rU_minus_1_over_x"] = [float(v) for v in (rU - 1.0) / x_small]
        # (rU-1)/x 应收敛到 -χ(0)
        if np.max(np.abs((rU - 1.0) / x_small + chi0)) > 1e-3 * max(abs(chi0), 1.0):
            fails.append(
                f"r→0 前导行为不对：(rU-1)/x={rep['rU_minus_1_over_x']}，应趋于 -χ(0)={-chi0}")

        # 6. U(rc)=0 直接验（冗余但便宜）
        rep["U_at_rc"] = float(np.atleast_1d(self.U(np.array([rc]), rc))[0])
        if abs(rep["U_at_rc"]) > 1e-12:
            fails.append(f"U(rc)={rep['U_at_rc']:.3e} != 0")

        rep["failures"] = fails
        if verbose:
            print(f"[self_check] {self.name}: {'PASS' if not fails else 'FAIL'}")
            for k, v in rep.items():
                if k != "failures":
                    print(f"    {k} = {v}")
            for f in fails:
                print(f"    !! {f}")
        return (not fails), rep


class _PolyMixin:
    """χ 是多项式时，A / poly_coeffs / lepton / chat 全部可解析。"""

    is_polynomial = True
    _chi_poly: Polynomial          # χ(x)
    _A_poly: Polynomial            # A(x) = ∫₀ˣ χ

    def _set_chi_poly(self, chi_poly: Polynomial):
        self._chi_poly = chi_poly
        self._A_poly = chi_poly.integ(lbnd=0.0)

    def chi(self, x):
        return self._chi_poly(np.asarray(x, dtype=float))

    def A(self, x):
        return self._A_poly(np.asarray(x, dtype=float))

    def poly_coeffs(self):
        """A(x)=Σ a_m x^{2m+1} ⇒ U = 1/r - Σ a_m r^{2m}/rc^{2m+1}，返回 (-a_m)。"""
        a = self._A_poly.coef
        # A 必须只有奇次项（χ 只有偶次项）才对应 U 的 r^{2m}/rc^{2m+1} 形式
        even_part = np.abs(a[0::2])
        if even_part.size and np.max(even_part) > 1e-12:
            raise ValueError(f"{self.name}: A(x) 含偶次项，无法写成标准系数表")
        return tuple(-float(v) for v in a[1::2])

    def lepton(self, rc="rc"):
        terms = ["1.0/r"]
        for m, b in enumerate(self.poly_coeffs()):
            if abs(b) < 1e-15:
                continue
            sign = "-" if b < 0 else "+"
            mag = abs(b)
            if m == 0:
                terms.append(f"{sign} {mag!r}/{rc}")
            else:
                terms.append(f"{sign} {mag!r}*r^{2*m}/{rc}^{2*m+1}")
        return "(" + " ".join(terms) + ")"

    def chat(self, q):
        """多项式窗也走基类的自适应 GL —— 决定节点数的是 cos(qx) 的振荡，
        不是 χ 的次数（见基类 chat 的告警）。"""
        return ClosureWindow.chat(self, q)


class ZMMWindow(_PolyMixin, ClosureWindow):
    """Sakuraba ZMM：χ_ell(x) = c_ell(1-x²)^ell，c_ell=(2ell+1)!!/(2ell)!!。

    **这是现有生产基线。构造出来后立刻与 test_lips_vs_pmeV2.6.py 的冻结系数
    逐位比对，不一致直接抛异常。**
    """

    def __init__(self, ell: int):
        if ell not in (1, 2, 3):
            raise ValueError(f"ZMM ell 只支持 1/2/3，收到 {ell}")
        self.ell = ell
        self.name = f"zmm{ell}"
        c = _double_fact_ratio(ell)
        one_minus_x2 = Polynomial([1.0, 0.0, -1.0])
        self._set_chi_poly(c * one_minus_x2 ** ell)
        self._c_ell = c
        self._assert_frozen()

    def _assert_frozen(self):
        got = np.asarray(self.poly_coeffs(), dtype=float)
        want = np.asarray(FROZEN_ZMM_POLY_COEFFS[self.ell], dtype=float)
        if got.shape != want.shape or not np.allclose(got, want, rtol=0, atol=1e-15):
            raise AssertionError(
                f"zmm{self.ell} 系数与 v2.6 冻结值不符：窗层={got} 冻结={want}")
        # c_ell 应等于常数项的分子（文档 §1 的结论）
        if abs(self._c_ell - (-want[0])) > 1e-15:
            raise AssertionError(
                f"zmm{self.ell}: c_ell={self._c_ell} != -b_0={-want[0]}")

    def lepton(self, rc="rc"):
        """**返回 v2.6 的冻结字符串**，不重新生成 —— 见文档 §4.3 纪律。"""
        return FROZEN_ZMM_LEPTON[self.ell].format(rc=rc)

    def lepton_regenerated(self, rc="rc"):
        """由系数重新生成的串，仅用于等价性测试，不进生产。"""
        return _PolyMixin.lepton(self, rc)


class PolyWindow(_PolyMixin, ClosureWindow):
    """χ(x) = (1-x²)^ell · P(x)，P 为多项式。

    (1-x²)^ell 因子让 ell 个端点条件（χ(1)=..=χ^(ell-1)(1)=0）**自动满足**，
    于是只剩归一化 ∫₀¹χ=1 一条线性约束——这正是 Stage 1 优化问题的可行空间，
    且 P=const 时精确退化为 ZMM。
    """

    def __init__(self, ell: int, p_poly: Polynomial, name: str | None = None):
        self.ell = int(ell)
        self.name = name or f"poly_ell{ell}_deg{p_poly.degree()}"
        one_minus_x2 = Polynomial([1.0, 0.0, -1.0])
        self._set_chi_poly((one_minus_x2 ** self.ell) * p_poly)
        self.p_poly = p_poly


class NodalWindow(ClosureWindow):
    """χ 只以求积节点上的值给出（PSWF 等数值窗）。没有 lepton/coeffs。"""

    def __init__(self, x_nodes, w_nodes, chi_nodes, ell=0, name="nodal"):
        self.name = name
        self.ell = ell
        self._x = np.asarray(x_nodes, dtype=float)
        self._w = np.asarray(w_nodes, dtype=float)
        self._chi = np.asarray(chi_nodes, dtype=float)
        # A(x) 用累积求积；chi(x) 用单调 3 次插值
        order = np.argsort(self._x)
        xs = self._x[order]
        cs = self._chi[order]
        self._xs, self._cs = xs, cs
        # A 在节点上的值（对 χ 的插值做精细再求积，避免 GL 累加的锯齿）
        fine = np.linspace(0.0, 1.0, 20001)
        chi_fine = np.interp(fine, xs, cs)
        A_fine = np.concatenate([[0.0], np.cumsum(
            0.5 * (chi_fine[1:] + chi_fine[:-1]) * np.diff(fine))])
        self._fine, self._A_fine = fine, A_fine

    def chi(self, x):
        return np.interp(np.asarray(x, dtype=float), self._xs, self._cs)

    def A(self, x):
        xa = np.asarray(x, dtype=float)
        return np.interp(np.clip(xa, 0.0, 1.0), self._fine, self._A_fine)


# --------------------------------------------------------------------------
def _double_fact_ratio(ell: int) -> float:
    """c_ell = (2ell+1)!!/(2ell)!!，由 ∫₀¹(1-t²)^ell dt = (2ell)!!/(2ell+1)!! 定。"""
    num = den = 1.0
    for k in range(1, 2 * ell + 2, 2):
        num *= k
    for k in range(2, 2 * ell + 1, 2):
        den *= k
    return num / den


class FrozenChiWindow(_PolyMixin, ClosureWindow):
    """χ 由冻结的偶次单项式系数给出（x^0, x^2, ... 的系数）。"""

    def __init__(self, chi_even_coeffs, ell, name):
        self.ell = int(ell)
        self.name = name
        full = np.zeros(2 * len(chi_even_coeffs) - 1)
        full[0::2] = np.asarray(chi_even_coeffs, dtype=float)
        self._set_chi_poly(Polynomial(full))


def pswfz2_window() -> FrozenChiWindow:
    """带限最优窗，ell=2 端点约束，带宽 c=2π。见 FROZEN_PSWFZ2_CHI_EVEN。"""
    return FrozenChiWindow(FROZEN_PSWFZ2_CHI_EVEN, ell=2, name="pswfz2")


def pcf_window(ell: int, c: float, deg: int = 10, n_quad: int = 400) -> PolyWindow:
    """Prolate Closure Family（文档 §14）：χ_{ell,c} = (1-x²)^ell ψ₀^c(x) / ∫。

    表示走 §14.7 的 **B1**：把 ψ₀^c 用 t=2x²-1 的 Legendre 拟合成 x 的偶次多项式，
    于是 χ 仍落在 (1-x²)^ell × 偶多项式 这个类里 ⇒ Lepton 串、插件系数表、解析
    dU/dr、偶性、端点条件、q^-(ell+1) 衰减**全部照旧**，下游一个字节不用改。

    职责分离（§14.2）：ell 定 x=1 的零点阶 ⇒ ĉ 的**渐近包络阶** O(q^-(ell+1))；
    c 只改带内形状与前因子。**必须说包络阶、不能说逐点**：sin/cos 因子会让首项
    在孤立 q 上归零，那附近的逐点斜率没有意义。
    c→0 时 ψ₀^c→const ⇒ 精确退化为 ZMM_ell（§14.3），所以 ZMM 是本族的极限成员。

    ⚠ PCF **不是**约束下的最优窗（§14.5）：乘 (1-x²)^ell 破坏了 ψ₀^c 的最优性，
    同 ell 下带外集中度必然不如 `pswfz2`。别把它说成「最优」。

    返回的窗上挂了 `fit_resid`（B1 拟合的相对残差，C1 关要 ≤1e-12）与 `band_c`。

    **c=0 直接返回冻结的 ZMM 对象本身**（不是在极小 c 上做数值极限）：
    这样 `pcf{ell}_c0` 能在命令行里当窗名用（`lips-force-audit --kernels pcf1_c0,...`），
    而 ZMM 的冻结系数断言与**逐比特相同的 Lepton 串**一个字节都不动，
    同时完全绕开 c→0 时集中算子近秩 1 的条件数问题。
    注意返回对象的 `.name` 仍是 `zmm{ell}` —— 故意的，免得别名冒充成另一个窗。
    """
    if c == 0:
        return zmm_window(int(ell))
    src = pswf_window(c, n_quad=n_quad)
    x, psi = src._x, src._chi
    t = 2.0 * x**2 - 1.0
    leg = np.polynomial.legendre.legvander(t, deg)
    coef = np.linalg.lstsq(leg, psi, rcond=None)[0]

    # Legendre(t) → x 的多项式：t = 2x²-1 只含偶次 ⇒ 复合后仍只含偶次
    t_poly = Polynomial([-1.0, 0.0, 2.0])
    mono = np.polynomial.legendre.leg2poly(coef)
    p_x = sum((a * t_poly**k for k, a in enumerate(mono)), Polynomial([0.0]))

    # 残差在**最终多项式**上量（leg2poly 的相消损失也一并体现，不藏账）
    resid = float(np.max(np.abs(p_x(x) - psi)) / np.max(np.abs(psi)))

    one_minus_x2 = Polynomial([1.0, 0.0, -1.0])
    norm = ((one_minus_x2 ** int(ell)) * p_x).integ(lbnd=0.0)(1.0)
    win = PolyWindow(int(ell), p_x / norm, name=f"pcf{ell}_c{c:g}")
    win.fit_resid = resid
    win.band_c = float(c)
    return win


def zmm_window(ell: int) -> ZMMWindow:
    return ZMMWindow(ell)


def pswf_window(c: float, n_quad: int = 400, ell: int = 0) -> NodalWindow:
    """余弦变换意义下的 PSWF 窗：ĉ(q) 在 |q|<c 内能量集中最大的 χ。

    算子核 K_c(x,y) = ∫₀^c cos(qx)cos(qy)dq
                    = ½[ sin(c(x-y))/(x-y) + sin(c(x+y))/(x+y) ]
    取最大特征向量，再归一化到 ∫₀¹χ=1。

    注意：prolate 函数在端点不为零，所以本窗 χ(1)≠0 —— 这正是文档 §8 风险 1，
    self_check(strict=True) 会判 FAIL。它只作消融对照用。
    """
    xg, wg = np.polynomial.legendre.leggauss(n_quad)
    x = 0.5 * (xg + 1.0)
    w = 0.5 * wg

    dx = x[:, None] - x[None, :]
    sx = x[:, None] + x[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        k1 = np.where(np.abs(dx) < 1e-14, c, np.sin(c * dx) / np.where(dx == 0, 1, dx))
        k2 = np.sin(c * sx) / sx
    K = 0.5 * (k1 + k2)

    sw = np.sqrt(w)
    Ksym = sw[:, None] * K * sw[None, :]
    Ksym = 0.5 * (Ksym + Ksym.T)
    evals, evecs = np.linalg.eigh(Ksym)
    v = evecs[:, -1] / sw
    if (w @ v) < 0:
        v = -v
    v = v / (w @ v)                      # 归一化 ∫₀¹χ=1
    win = NodalWindow(x, w, v, ell=0, name=f"pswf_c{c:g}")
    # 集中度 = ∫₀^c ĉ²dq / ∫₀^∞ ĉ²dq，分母由 Parseval = (π/2)∫₀¹χ²
    win.concentration = float(2.0 * evals[-1] / np.pi)
    win.band_c = c
    return win


WINDOW_REGISTRY = {
    "zmm1": lambda: zmm_window(1),
    "zmm2": lambda: zmm_window(2),
    "zmm3": lambda: zmm_window(3),
    "pswfz2": pswfz2_window,          # 旧名保留为别名，绝不重命名
    # PCF 的退化极限：χ_{ell,c→0} = χ_ell^ZMM 是**精确等式**（集中核退化成常数
    # ⇒ ψ₀^c → const），所以这三个键直接解析到冻结的 ZMM 对象。
    # 目的是让「族内最优在 (ell=1, c→0)」这类说法能被命令行直接复现，
    # 而不是只停留在文字里。zmm1/2/3 仍是规范名（§14.4）。
    "pcf1_c0": lambda: zmm_window(1),
    "pcf2_c0": lambda: zmm_window(2),
    "pcf3_c0": lambda: zmm_window(3),
}


def make_window(name: str):
    if name in WINDOW_REGISTRY:
        return WINDOW_REGISTRY[name]()
    if name.startswith("pswf_c"):
        return pswf_window(float(name[len("pswf_c"):]))
    m = re.fullmatch(r"pcf(\d+)_c([0-9.eE+-]+)", name)
    if m:
        return pcf_window(int(m.group(1)), float(m.group(2)))
    raise ValueError(f"未知窗名 {name!r}；已注册：{sorted(WINDOW_REGISTRY)}")


if __name__ == "__main__":
    print("=" * 72)
    print("窗层自检 —— ZMM 三档必须全 PASS，且系数与 v2.6 冻结值逐位一致")
    print("=" * 72)
    for ell in (1, 2, 3):
        wnd = zmm_window(ell)
        ok, rep = wnd.self_check(verbose=True)
        print(f"    c_ell            = {wnd._c_ell}")
        print(f"    poly_coeffs      = {wnd.poly_coeffs()}")
        print(f"    lepton (frozen)  = {wnd.lepton('rc')}")
        print(f"    lepton (regen)   = {wnd.lepton_regenerated('rc')}")
        print(f"  => {'PASS' if ok else 'FAIL'}\n")

    print("=" * 72)
    print("带限最优窗 pswfz2（新增）—— 必须同样通过全部自检")
    print("=" * 72)
    w = pswfz2_window()
    ok, _ = w.self_check(verbose=True)
    print(f"    poly_coeffs = {w.poly_coeffs()}")
    print(f"    lepton      = {w.lepton('rc')}")
    print(f"  => {'PASS' if ok else 'FAIL'}")

    # ---- §14.8 PCF kill switch C0/C1/C2（纯 numpy，秒级，零 MD）-------------
    print("\n" + "=" * 72)
    print("§14.8 C0/C1 —— PCF 的退化极限与 B1 多项式表示")
    print("=" * 72)
    print(f"{'ell':>4s} {'c':>8s} {'deg':>4s} {'fit_resid':>11s} "
          f"{'self_check':>10s} {'max|coef-ZMM|':>14s}")
    for ell in (1, 2, 3):
        for c, deg in ((1e-6, 4), (2.0, 8), (2 * np.pi, 10), (2 * np.pi, 14),
                       (12.0, 14), (12.0, 20)):
            wnd = pcf_window(ell, c, deg=deg)
            ok, _ = wnd.self_check(strict=True)
            got = np.asarray(wnd.poly_coeffs(), dtype=float)
            want = np.asarray(FROZEN_ZMM_POLY_COEFFS[ell], dtype=float)
            n = min(got.size, want.size)
            dz = max(float(np.max(np.abs(got[:n] - want[:n]))),
                     float(np.max(np.abs(got[n:]))) if got.size > n else 0.0)
            print(f"{ell:4d} {c:8.4g} {deg:4d} {wnd.fit_resid:11.3e} "
                  f"{'PASS' if ok else 'FAIL':>10s} {dz:14.3e}")
    print("  C0: c→0 那几行的 max|coef-ZMM| 必须 ~1e-15（family 含 ZMM 为极限成员）")
    print("  C1: fit_resid <= 1e-12 才不动到 fixture 的 1e-12 断言；否则提高 deg")

    print("\n" + "=" * 72)
    print("§14.8 C2 —— 带外能量 ∫_{q>2π} ĉ²dq（对 zmm2 归一，越小越集中）")
    print("        §14.5 预期：PCF 必然不如约束最优的 pswfz2。实测确认。")
    print("        §14.2 预期：ell 定渐近**包络**阶、c 不改该阶。实测确认（包络斜率在下）。")
    print("=" * 72)
    _QLO, _QHI = 2 * np.pi, 100.0        # 实测 hi=50 起就收敛到 4 位；ĉ²~q^-6
    # 外层节点数从 _QHI 推：ĉ² 的振荡周期是 π，要 >>10 点/周期
    qg, qw = np.polynomial.legendre.leggauss(int(max(2000, 40 * _QHI / np.pi)))
    q_out = 0.5 * (_QHI - _QLO) * qg + 0.5 * (_QHI + _QLO)
    w_out = 0.5 * (_QHI - _QLO) * qw

    def _out_of_band(wnd):
        return float(w_out @ wnd.chat(q_out) ** 2)

    base = _out_of_band(zmm_window(2))
    print(f"{'c':>7s} {'pcf1':>9s} {'pcf2':>9s} {'pcf3':>9s}")
    for c in (1e-6, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 2 * np.pi):
        vals = [_out_of_band(pcf_window(ell, c, deg=16)) / base for ell in (1, 2, 3)]
        print(f"{c:7.3g} " + " ".join(f"{v:9.4f}" for v in vals))
    print(f"{'zmm':>7s} " + " ".join(
        f"{_out_of_band(zmm_window(ell)) / base:9.4f}" for ell in (1, 2, 3)))
    print(f"{'pswfz2':>7s} {_out_of_band(pswfz2_window()) / base:9.4f}"
          f"   <- 约束最优，PCF 全表都不如它")
    print("  读法：c→0 那一行必须逐位等于 zmm 行（C0 的另一种查法）。")
    print("  c 的最优值随 ell 移动（ell=1 约 c≈4，ell=2 约 c≈2.5，ell=3 是 c→0），")
    print("  且 ell=3 时 c 单调劣化 —— 即 §14.6「优化器把 c 拧回 0」在 ell=3 成立。")

    print("\n" + "=" * 72)
    print("§14.2 职责分离验证：ĉ 包络斜率 dlog|ĉ|/dlogq（q~200-400）")
    print("=" * 72)
    for ell in (1, 2, 3):
        for c in (1e-6, 2.0, 2 * np.pi):
            wnd = pcf_window(ell, c, deg=16)
            env = [max(np.max(np.abs(wnd.chat(np.linspace(q0, 1.15 * q0, 400)))), 1e-300)
                   for q0 in (200.0, 400.0)]
            slope = np.log(env[1] / env[0]) / np.log(2.0)
            print(f"  ell={ell} c={c:6.3g}: 斜率={slope:+.3f}   理论 -(ell+1)={-(ell + 1)}")
