#!/usr/bin/env python
"""路线 B 的 B0 关 —— local closure + 少量 low-k reciprocal 修正（纯 numpy，零 MD）。

设计来源：docs/plans/PLAN_PSWF_Closure_Family.md §13。B0 是那条路线**最便宜、
也必须最先过**的一关（§13.6）：单帧、纯 numpy、对精确 Ewald 的逐原子力误差，
看它随 k_c 是否单调下降、并在 k_c=2π/rc 处显著优于纯 closure。

    U_total = U_local + U_low-k,
    U_low-k = (1/2V) Σ_{0<|k|<=k_c} (4π/k²)·w(k)·|ρ_q(k)|²,   w(k)=ĉ(k·rc)

§13.1 的关键结论是 **w(k) 不是自由参数**：split 1/r = L + S 里
L̂ = (4π/k²)(1-ĉ)，所以每个模式上缺掉的恰好是 Ŝ = (4π/k²)ĉ。本脚本 CHECK-1
就把这个恒等式用直接径向求积验一遍——形式或系数推错，那里就炸。

判定尺度沿用 `lips-force-audit` 的 rel_rms = sqrt(<|ΔF|²>/<|F_ref|²>)。

**⚠ 范围：本脚本在 PLAN §2 的非目标之外。** §2 写死了「不在本轮引入 FFT /
reciprocal space」，终点必须保持 U=Σ_{i<j, r<rc} Q_iQ_jK(r_ij)。本脚本验证的
low-k 修正引入了 reciprocal 求和 ⇒ 属于**需要用户明确表态的范围扩展**，
不是默认继续。B0 的数是可信的，但别把它当已获批的路线往下建。

**⚠ H2 · response consistency —— 本脚本一个字都没验。** 倒空间能量里每一处 Q_i
依赖都得进 λ_i=∂E/∂Q_i，通道有三条：(1) pair ∂|ρ_k|²/∂Q_i=2Re[e^{-ik·r_i}ρ_k*]；
(2) **self** ∂(Q_i²)/∂Q_i=2Q_i≠0 —— 固定电荷下 self 项是常数、对力零贡献，
PME 的习惯就是忽略它，**CWLD 下这个反射是错的**；(3) **background** 丢 k=0
蕴含的中性化背景带 (ΣQ_i)² 结构，∂/∂Q_i≠0，同时把 H2 与净电荷那条耦合起来。
失效模式：**能量对、力不是它的梯度** ⇒ 只是 NVE 慢漂，而该通道在 fp32 + 200 ps
下测不出来 ⇒ 下游抓不到。本脚本用固定电荷 + 严格中性合成体系 ⇒ **三条通道全未激活**。

**范围声明（别越界读）**：本脚本用的是合成的抖动格点体系，不是 1AAY/水。
它判的是「§13 的形式与系数对不对、收益随 k_c 怎么走」，**不是**生产体系上的
绝对精度。真体系上的数要靠 `lips-force-audit`（B2）与实测 S_ZZ（B1）。
排除对（§13.5-2）与 self 项（§13.5-3）在这里不适用：合成体系没有 bonded
exclusion，而 self 项在**固定电荷下**与坐标无关 ⇒ 对力零贡献（只有能量需要它）。
⚠ 「self 项反正是常数」这句话**只在固定电荷下成立** —— 上面 H2 的通道 (2) 就是
这个反射在 CWLD 下失效。本 docstring 早前把它写成了无条件的，已更正。

用法：
    lips-hybrid-lowk --checks              # 三个恒等式/收敛检查，约 1 分钟
    lips-hybrid-lowk                       # 检查 + k_c 扫描主表
    lips-hybrid-lowk --n-side 12 --windows zmm1,zmm2,pswfz2
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from lips.engine import closure as cw

ONE_4PI_EPS0 = 138.93545764498226      # kJ/mol·nm/e²
RC_DEFAULT = 1.2


def log(m):
    print(m, flush=True)


def _rms(F):
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", F, F))))


# ---------------------------------------------------------------------------
# k 网格与通用倒空间求和
# ---------------------------------------------------------------------------
def kvecs(L: float, kmax: float):
    """立方盒里 0<|k|<=kmax 的全部格矢（含 ±k 两半，不做半空间折叠）。"""
    n = int(np.floor(kmax * L / (2.0 * np.pi) + 1e-12))
    idx = np.arange(-n, n + 1)
    g = np.stack(np.meshgrid(idx, idx, idx, indexing="ij"), -1).reshape(-1, 3)
    k = (2.0 * np.pi / L) * g
    k2 = np.einsum("ij,ij->i", k, k)
    keep = (k2 > 0) & (k2 <= kmax**2 * (1 + 1e-12))
    return k[keep], k2[keep]


def recip_forces(pos, q, V, k, vhat, chunk=2048):
    """F_i = (q_i/V) Σ_k v̂(k)·k·Im[ρ*(k)e^{ik·r_i}]，ρ(k)=Σ_j q_j e^{ik·r_j}。

    推导：U=(1/2V)Σ_k v̂|ρ|²，∂|ρ|²/∂r_i = -2q_i k Im[ρ* e^{ik·r_i}]，F=-∂U/∂r。
    """
    F = np.zeros_like(pos)
    for s in range(0, len(k), chunk):
        kk, vh = k[s:s + chunk], vhat[s:s + chunk]
        E = np.exp(1j * (kk @ pos.T))                       # (Kc, N)
        rho = E @ q
        coef = vh[:, None] * (np.conj(rho)[:, None] * E).imag
        F += (coef.T @ kk) * q[:, None]
    return F / V


# ---------------------------------------------------------------------------
# 参照：精确 Ewald（纯 numpy）
# ---------------------------------------------------------------------------
def ewald_forces(pos, q, L, alpha, k_sigma=10.0):
    """标准 Ewald。alpha 需满足 erfc(alpha·L·√3/2)≈0（于是最小镜像即够）。

    倒空间截到 k=k_sigma·alpha ⇒ exp(-k²/4α²)=exp(-k_sigma²/4)，10 给 1.4e-11。
    """
    from scipy.special import erfc
    d = pos[:, None, :] - pos[None, :, :]
    d -= L * np.round(d / L)
    r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
    np.fill_diagonal(r, np.inf)
    # -dU/dr / r for U = q_i q_j erfc(ar)/r
    g = erfc(alpha * r) / r**2 + (2 * alpha / np.sqrt(np.pi)) * np.exp(-(alpha * r) ** 2) / r
    pref = (q[:, None] * q[None, :]) * g / r
    F = np.einsum("ij,ijk->ik", pref, d)

    k, k2 = kvecs(L, k_sigma * alpha)
    vhat = 4.0 * np.pi / k2 * np.exp(-k2 / (4.0 * alpha**2))
    return F + recip_forces(pos, q, L**3, k, vhat)


# ---------------------------------------------------------------------------
# 被测：local closure（+ low-k 修正）
# ---------------------------------------------------------------------------
def closure_forces(pos, q, L, rc, win):
    """U(r)=(1-A(r/rc))/r 的最小镜像 pair 力。rc<=L/2 时紧支撑 ⇒ 与全周期和等价。"""
    d = pos[:, None, :] - pos[None, :, :]
    d -= L * np.round(d / L)
    r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
    np.fill_diagonal(r, np.inf)
    inside = r < rc
    rsafe = np.where(inside, r, 0.5 * rc)
    dU = np.where(inside, win.dUdr(rsafe, rc), 0.0)
    pref = -(q[:, None] * q[None, :]) * dU / np.where(inside, r, 1.0)
    return np.einsum("ij,ijk->ik", pref, d)


def scan(pos, q, L, rc, wins, kc_grid, Fref, chunk=2048):
    """扫 k_c，返回 (nmodes, {name: [rel_rms per k_c]})。

    各 k_c 的模式集是**嵌套前缀**（按 |k| 排序后），所以整张表只需对最大的 k 集
    遍历一次、在每个 k_c 处快照。窗之间共用同一批 e^{ik·r}，只有 v̂ 不同。
    """
    kc_grid = sorted(kc_grid)
    k, k2 = kvecs(L, max(kc_grid))
    order = np.argsort(k2, kind="stable")
    k, k2 = k[order], k2[order]
    kmag = np.sqrt(k2)
    V = L**3
    F = {n: closure_forces(pos, q, L, rc, w) for n, w in wins.items()}
    vh = {n: (4.0 * np.pi / k2) * w.chat(kmag * rc) for n, w in wins.items()}
    r2 = np.mean(np.einsum("ij,ij->i", Fref, Fref))

    out = {n: [] for n in wins}
    nmodes, cut = [], 0
    for kc in kc_grid:
        end = int(np.searchsorted(kmag, kc, side="right"))
        for s in range(cut, end, chunk):
            e = min(s + chunk, end)
            E = np.exp(1j * (k[s:e] @ pos.T))
            rho = E @ q
            im = (np.conj(rho)[:, None] * E).imag
            for n in wins:
                F[n] += ((vh[n][s:e, None] * im).T @ k[s:e]) * q[:, None] / V
        cut = end
        nmodes.append(end // 2)                 # 独立模式数（ρ(-k)=ρ*(k)）
        for n in wins:
            d = F[n] - Fref
            out[n].append(float(np.sqrt(np.mean(np.einsum("ij,ij->i", d, d)) / r2)))
    return nmodes, out


# ---------------------------------------------------------------------------
# 合成体系：抖动的交替电荷格点（中性、有最小间距、有屏蔽）
# ---------------------------------------------------------------------------
def make_system(n_side, L, jitter=0.25, seed=0):
    rng = np.random.default_rng(seed)
    a = L / n_side
    g = np.stack(np.meshgrid(*(np.arange(n_side),) * 3, indexing="ij"), -1).reshape(-1, 3)
    pos = np.mod((g + 0.5) * a + rng.normal(0.0, jitter * a, size=(len(g), 3)), L)
    q = np.ones(len(pos))
    q[rng.permutation(len(pos))[: len(pos) // 2]] = -1.0
    q -= q.mean()                                   # 严格中性（§13.5-4）
    return pos, q


# ---------------------------------------------------------------------------
# CHECK-1：§13.1 的 split 恒等式  L̂(k) = (4π/k²)(1-ĉ(k·rc))
# ---------------------------------------------------------------------------
def check_split_identity(rc=RC_DEFAULT, names=("zmm1", "zmm2", "zmm3", "pswfz2")):
    """L̂(k)=(4π/k)∫₀^rc (1-A(r/rc))sin(kr)dr，与 (4π/k²)(1-ĉ) 逐点比。"""
    log("=" * 78)
    log("CHECK-1  §13.1 的 split 恒等式：L̂(k) 直接径向求积  vs  (4π/k²)(1-ĉ(k·rc))")
    log("=" * 78)
    kk = np.array([0.2, 0.5, 1.0, 2.0, 2 * np.pi / rc, 10.0, 20.0, 50.0])
    # 节点数**从 kk 推**，不写死：被积的振荡参数是 q=k·rc，n 点 GL 只撑到 q≈2n。
    # 写死会在有人往 kk 里加大 k 时静默混叠——这个坑本仓库已经踩过两次
    # （kill_switch 的 600 点算到 q=2e4；ClosureWindow.chat 的 400 点）。
    n_x = int(max(2000, 512 * np.ceil(8.0 * kk.max() * rc / 512)))
    xg, wg = np.polynomial.legendre.leggauss(n_x)
    x, w = 0.5 * (xg + 1.0), 0.5 * wg * rc          # dr = rc·dx
    log(f"{'window':>10s} " + " ".join(f"{k:9.3f}" for k in kk))
    ok = True
    for nm in names:
        wnd = cw.make_window(nm)
        one_minus_A = 1.0 - wnd.A(x)
        rel = []
        for k in kk:
            lhat = (4.0 * np.pi / k) * float(w @ (one_minus_A * np.sin(k * rc * x)))
            pred = (4.0 * np.pi / k**2) * (1.0 - float(np.atleast_1d(wnd.chat(k * rc))[0]))
            rel.append(abs(lhat - pred) / abs(pred))
        ok &= max(rel) < 1e-9
        log(f"{nm:>10s} " + " ".join(f"{v:9.2e}" for v in rel))
    log(f"  -> {'PASS' if ok else 'FAIL'}（相对差全部 <1e-9 才算形式与系数都对）\n")
    return ok


# ---------------------------------------------------------------------------
# CHECK-2/3：参照可信 + hybrid 在 k_c→大 时收敛到它
# ---------------------------------------------------------------------------
def check_reference(pos, q, L, alphas=(1.2, 1.8, 2.4)):
    log("=" * 78)
    log("CHECK-2  参照可信度：Ewald 力对 alpha 的无关性（同一帧，三个 alpha）")
    log("=" * 78)
    Fs = [ewald_forces(pos, q, L, a) for a in alphas]
    ok = True
    for a, F in zip(alphas[1:], Fs[1:]):
        rel = _rms(F - Fs[0]) / _rms(Fs[0])
        ok &= rel < 1e-8
        log(f"  alpha={alphas[0]} vs {a}:  rel_rms = {rel:.3e}")
    log(f"  -> {'PASS' if ok else 'FAIL'}（<1e-8）\n")
    return ok, Fs[0]


def check_convergence(rc=RC_DEFAULT, seed=0):
    """小盒子专用（L=2.6 nm）：k_c 推到 60 nm⁻¹ 看 hybrid → 精确 Ewald。

    单独用小盒是为了成本：模式数 ∝ (k_c·L)³，生产尺寸的 L 推不到这么高的 k_c。
    """
    L, rc = 2.6, rc
    pos, q = make_system(5, L, seed=seed)
    Fref = ewald_forces(pos, q, L, 2.5)
    log("=" * 78)
    log(f"CHECK-3  k_c→大 时 hybrid 必须收敛到精确 Ewald（小盒 L={L} nm, N={len(pos)}）")
    log("=" * 78)
    grid = [0.0, 2 * np.pi / rc, 10.0, 20.0, 40.0, 60.0]
    wins = {"zmm2": cw.make_window("zmm2")}
    nmodes, res = scan(pos, q, L, rc, wins, grid, Fref)
    mono = all(b < a for a, b in zip(res["zmm2"], res["zmm2"][1:]))
    for kc, nm, v in zip(sorted(grid), nmodes, res["zmm2"]):
        log(f"  k_c={kc:6.2f} nm⁻¹  独立模式={nm:8d}   rel_rms={v:.4e}")
    log(f"  -> {'PASS' if mono else 'FAIL'}（必须单调下降；k_c=0 那行就是纯 closure）\n")
    return mono


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="B0：local closure + low-k reciprocal（纯 numpy）")
    ap.add_argument("--n-side", type=int, default=10, help="每边格点数，N=n_side³")
    ap.add_argument("--box", type=float, default=6.8, help="立方盒边长 nm（默认对齐生产 L）")
    ap.add_argument("--rc", type=float, default=RC_DEFAULT)
    ap.add_argument("--jitter", type=float, default=0.25, help="格点抖动，单位为格距")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.2, help="参照 Ewald 的 alpha")
    ap.add_argument("--windows", default="zmm1,zmm2,zmm3,pswfz2,pcf1_c4,pcf2_c2.5")
    ap.add_argument("--kc-max", type=float, default=15.0)
    ap.add_argument("--checks", action="store_true", help="只跑恒等式/收敛检查")
    args = ap.parse_args()

    L, rc = args.box, args.rc
    if rc > 0.5 * L:
        raise SystemExit(f"rc={rc} > L/2={0.5*L}：紧支撑窗的最小镜像和不再等价于全周期和")

    ok = check_split_identity(rc)
    pos, q = make_system(args.n_side, L, args.jitter, args.seed)
    log(f"[sys] N={len(pos)}  L={L} nm  rho={len(pos)/L**3:.2f} /nm³  "
        f"Σq={q.sum():.2e}  k_min={2*np.pi/L:.4f} nm⁻¹  k_c(2π/rc)={2*np.pi/rc:.4f}")
    ok_ref, Fref = check_reference(pos, q, L, (args.alpha, 1.8, 2.4))
    ok &= ok_ref
    ok &= check_convergence(rc, args.seed)
    if args.checks:
        raise SystemExit(0 if ok else 1)

    log(f"[ref] 精确 Ewald: rms|F| = {ONE_4PI_EPS0*_rms(Fref):.4g} kJ/mol/nm")
    wins = {n.strip(): cw.make_window(n.strip()) for n in args.windows.split(",")}
    grid = sorted({0.0, 2 * np.pi / rc, *np.geomspace(0.9, args.kc_max, 22)})
    t0 = time.time()
    nmodes, res = scan(pos, q, L, rc, wins, grid, Fref)
    log(f"[scan] {len(grid)} 个 k_c × {len(wins)} 个窗，一次 k 遍历，{time.time()-t0:.1f}s")

    show = [0.0, 2 * np.pi / rc]
    show += [g for g in grid if g not in show][::3]
    show = sorted(set(show))
    idx = {g: i for i, g in enumerate(grid)}
    log("\n" + "=" * 78)
    log("B0 主表：rel_rms 力误差（对精确 Ewald），行=窗，列=k_c[nm⁻¹]")
    log("        k_c=0 就是纯 closure；k_c=2π/rc=%.3f 是 §13 的标称带宽" % (2 * np.pi / rc))
    log("=" * 78)
    log(f"{'window':>12s} " + " ".join(f"{g:9.3g}" for g in show))
    log(f"{'独立模式数':>12s} " + " ".join(f"{nmodes[idx[g]]:9d}" for g in show))
    for n in wins:
        log(f"{n:>12s} " + " ".join(f"{res[n][idx[g]]:9.2e}" for g in show))
    log("\n  纯 closure → k_c=2π/rc 的改善倍率：")
    for n in wins:
        e0, e1 = res[n][idx[0.0]], res[n][idx[2 * np.pi / rc]]
        log(f"    {n:>12s}  {e0:.3e} -> {e1:.3e}   {e0/e1:6.1f}x")

    # §13.3 的可量化断言：同一目标残差下，好窗需要的模式数少多少倍
    log("\n" + "=" * 78)
    log("§13.3 的断言：同样的力误差，更集中的窗需要更少的 reciprocal 模式")
    log("        （原文按 q_c^-(2ell+1) 估 pswfz2 只需 zmm2 的 1/4.8。这里直接量。）")
    log("=" * 78)
    base = "zmm2" if "zmm2" in wins else next(iter(wins))
    names = list(wins)
    log(f"{'target rel':>11s} " + " ".join(f"{n:>15s}" for n in names))
    m = np.asarray(nmodes)
    for tgt in (3e-2, 1e-2, 3e-3, 1e-3):
        need = {}
        for n in names:
            hit = np.nonzero(np.asarray(res[n]) <= tgt)[0]
            need[n] = int(m[hit[0]]) if len(hit) else None
        ref = need[base]
        cells = []
        for n in names:
            v = need[n]
            if v is None:
                cells.append(f"{'>k_c max':>15s}")
            elif ref:
                cells.append(f"{v:9d}({ref/max(v,1):4.2f}x)")
            else:
                cells.append(f"{v:15d}")
        log(f"{tgt:11.0e} " + " ".join(cells))
    log(f"  括号内 = 相对 {base} 的模式数节省倍率（>1 表示需要的模式更少）")
    log("  ⚠ 模式数落在 k_c 网格上，倍率的分辨率受网格限制；只读量级，不读小数。")

    # 主表里 pswfz2 会在 k_c≳10 处**平台化**。机制在 ĉ 的包络上，不在实现里：
    # 约束最优解是「min ∫_{q>2π}ĉ²」，那个积分由紧挨着 2π 的头几个振荡主导，
    # 于是解会把带边压下去、把**尾巴换胖**。下表就是拿来看这件事的。
    log("\n" + "=" * 78)
    log("诊断：ĉ(q) 的包络（q 附近 ±7% 取 max|ĉ|）—— 解释主表里的平台化")
    log("=" * 78)
    names = list(wins)
    log(f"{'q':>7s} {'k=q/rc':>8s} " + " ".join(f"{n:>11s}" for n in names))
    for q0 in (2 * np.pi, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 30.0, 50.0):
        qs = np.linspace(0.93 * q0, 1.07 * q0, 300)
        row = [float(np.max(np.abs(wins[n].chat(qs)))) for n in names]
        log(f"{q0:7.2f} {q0/rc:8.2f} " + " ".join(f"{v:11.3e}" for v in row))
    log("  读法：某个窗在小 q 上赢、在大 q 上输 ⇒ 它只在标称带宽附近有优势，")
    log("        k_c 再往上推就会被自己的尾巴限住（这正是主表 pswfz2 那一行的形状）。")


if __name__ == "__main__":
    main()
