"""窗层的回归护栏 —— 覆盖 PCF 的 C0/C1、§13.1 的 split 恒等式、chat 的高-q 混叠。

判据与数值都来自 docs/reports/REPORT_HYBRID_LOWK_B0_2026-09-07.md。纯 numpy，秒级。
"""
import numpy as np
import pytest

from lips.engine.closure import (FROZEN_ZMM_POLY_COEFFS, make_window,
                                 pcf_window, zmm_window)


# 节点数从下面 test_split_identity 用到的最大 k 推（q=k·rc，n 点 GL 只撑到 q≈2n）
_K_MAX, _RC = 50.0, 1.2
_XG, _WG = np.polynomial.legendre.leggauss(int(max(2000, 8 * _K_MAX * _RC)))
_X = 0.5 * (_XG + 1.0)


@pytest.mark.parametrize("ell", [1, 2, 3])
def test_c0_pcf_degenerates_to_zmm(ell):
    """C0：c→0 时 PCF 必须回到 ZMM 的冻结系数（ZMM 是 family 的极限成员）。"""
    # deg=4 是 t=2x²-1 的次数 ⇒ χ 比 ZMM 多出几个高次项，它们必须 ~0
    got = np.asarray(pcf_window(ell, 1e-6, deg=4).poly_coeffs())
    want = np.asarray(FROZEN_ZMM_POLY_COEFFS[ell])
    n = want.size
    assert got.size >= n
    assert np.max(np.abs(got[:n] - want)) < 1e-11
    assert np.max(np.abs(got[n:]), initial=0.0) < 1e-11


@pytest.mark.parametrize("name", ["zmm1", "zmm2", "zmm3", "pswfz2", "pcf2_c2.5"])
def test_endpoint_and_normalisation(name):
    ok, rep = make_window(name).self_check(strict=True)
    assert ok, rep["failures"]


def test_c1_even_poly_fit_beats_fixture_tolerance():
    """C1：B1 表示的拟合残差必须远小于 fixture 的 1e-12 断言。"""
    for c in (1.0, 2.0, 2 * np.pi, 12.0):
        assert pcf_window(2, c, deg=16).fit_resid < 1e-12


def test_split_identity_lhat_equals_one_minus_chat():
    """§13.1：L̂(k) = (4π/k²)(1-ĉ(k·rc))。低-k 修正权重 w(k)=ĉ 就靠这条。"""
    rc = _RC
    x, w = _X, 0.5 * _WG * rc
    for name in ("zmm1", "zmm2", "zmm3", "pswfz2"):
        wnd = make_window(name)
        one_minus_A = 1.0 - wnd.A(x)
        for k in (0.2, 1.0, 2 * np.pi / rc, 20.0, _K_MAX):
            lhat = (4 * np.pi / k) * float(w @ (one_minus_A * np.sin(k * rc * x)))
            pred = (4 * np.pi / k**2) * (1.0 - float(wnd.chat(k * rc)[0]))
            assert abs(lhat - pred) / abs(pred) < 1e-9, (name, k)


def _chat_exact(wnd, q):
    """ĉ(q) 的闭式：χ=Σa_m x^{2m} ⇒ 递推 ∫₀¹x^n cos/sin(qx)dx。

    递推方向是「除以 q」，大 q 上稳定；小 q 会相消，所以它只当**高-q 的独立
    oracle**，不拿来替 chat 的自适应求积（两个不同方法对上，比调节点数更有说服力）。
    """
    a = wnd._chi_poly.coef
    I = np.sin(q) / q                     # I_0
    J = (1.0 - np.cos(q)) / q             # J_0
    tot = a[0] * I
    for n in range(1, len(a)):
        I, J = np.sin(q) / q - (n / q) * J, -np.cos(q) / q + (n / q) * I
        tot += a[n] * I
    return tot


@pytest.mark.parametrize("name", ["zmm2", "zmm3", "pswfz2"])
def test_chat_no_high_q_aliasing(name):
    """chat 曾写死 400 点 GL —— n 点 GL 只撑到 q≈2n，q≳250 起是纯混叠的 O(1) 假值
    （∫_{2π}^{2000}ĉ²dq 因此虚高 830×，还对外层网格不敏感、看起来"收敛"）。
    """
    wnd = make_window(name)
    for q in (300.0, 1000.0):
        got = float(wnd.chat(q)[0])
        want = _chat_exact(wnd, q)
        # 绝对判据：ĉ 在这里已是 1e-6 量级，而求积是对 O(1) 的 χ 做相消求和，
        # 所以能指望的是**绝对**精度 ~1e-13，不是相对精度。混叠时假值是 O(1e-2)，
        # 1e-11 这道门照样拦得住。
        assert abs(got - want) < 1e-11, (name, q, got, want)


@pytest.mark.parametrize("ell", [1, 2, 3])
def test_pcf_c0_alias_is_the_frozen_zmm(ell):
    """`pcf{ell}_c0` 必须解析到**冻结的** ZMM 对象，不能是 c→0 的数值近似。

    别名一旦悄悄漂成"另算一遍的窗"，冻结系数断言与逐比特 Lepton 串就白设了。
    两条路（注册表键 / pcf_window(c=0)）都要落到同一个东西上。
    """
    zmm = make_window(f"zmm{ell}")
    for alias in (make_window(f"pcf{ell}_c0"), pcf_window(ell, 0)):
        assert type(alias) is type(zmm)
        assert alias.poly_coeffs() == zmm.poly_coeffs()      # 逐比特
        assert alias.lepton("rc") == zmm.lepton("rc")        # 冻结字符串原样
        assert alias.name == f"zmm{ell}"                     # 别名不冒充新窗
