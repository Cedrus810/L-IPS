#!/usr/bin/env python
"""跨 run 的派生量汇总 —— 把「当场算完直接报」变成可复现的产物。

存在的理由：2026-09-07 这一天里，报告 §7.3-R / §9.3-R2/R3/R4 / PLAN §14.6c 的
派生数字（dt 标度比、A/B 拆解、seed sem、σ 检验、对 zmm2 归一、O(c²) 系数、兑换率）
全是用一次性的 `python -c` 算出来直接写进 markdown 的 —— 与 §9.2 那两个丢掉来源的
Δq 数字（0.75% / 11.5%）**是同一种失败**。本脚本让每一个数字都能用一条命令重算。

只读 `force_audit` / `nve_drift` 已经写在盘上的 CSV，**不跑 MD、不碰轨迹**。

用法：
    python -m lips.analysis.cross_run            # 两张表都出，写盘
    python -m lips.analysis.cross_run --no-write # 只打印
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from lips import paths


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------
def _nve_files():
    return sorted(glob.glob(paths.data("closure_nve_drift_dt*.csv")))


def nve_summary():
    """按 (precision, kernel 集合签名, dt) 聚合，并给出 dt 配对的标度比与 A/B 拆解。"""
    rows = []
    for f in _nve_files():
        m = re.search(r"closure_nve_drift_dt([0-9.]+)fs_([a-z]+)_(\d+k[0-9a-f]+)\.csv$",
                      os.path.basename(f))
        if not m:
            continue
        dt_fs, prec, sig = float(m.group(1)), m.group(2), m.group(3)
        d = pd.read_csv(f)
        for k, g in d.groupby("kernel"):
            v = g["drift_kT_per_ns_per_dof"].to_numpy(float)
            n = len(v)
            rows.append(dict(
                precision=prec, sig=sig, dt_fs=dt_fs, kernel=k, n_seed=n,
                mean=v.mean(),
                sem=(v.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                max_nonlinearity=g["nonlinearity"].max(),
                # 单条 run 的拟合 se —— 与 seed sem 对比用（教训 #1）
                fit_se_typ=g["drift_se_kT_per_ns_per_dof"].median()
                if "drift_se_kT_per_ns_per_dof" in g else np.nan,
                src=os.path.basename(f)))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    per = pd.DataFrame(rows).sort_values(["precision", "sig", "kernel", "dt_fs"])

    # dt 配对：同 (precision, sig, kernel) 下取两个最小的 dt 做标度比
    pair = []
    for (prec, sig, k), g in per.groupby(["precision", "sig", "kernel"]):
        g = g.sort_values("dt_fs")
        if len(g) < 2:
            continue
        for i in range(len(g) - 1):
            hi, lo = g.iloc[i], g.iloc[i + 1]          # hi = 大 dt
            ratio = lo["mean"] / hi["mean"]
            # drift = A/dt + B dt^2 的两点解
            M = np.array([[1 / hi.dt_fs, hi.dt_fs ** 2], [1 / lo.dt_fs, lo.dt_fs ** 2]])
            try:
                A, B = np.linalg.solve(M, [hi["mean"], lo["mean"]])
            except np.linalg.LinAlgError:
                A = B = np.nan
            pair.append(dict(
                precision=prec, sig=sig, kernel=k,
                dt_hi_fs=hi.dt_fs, dt_lo_fs=lo.dt_fs,
                drift_hi=hi["mean"], drift_lo=lo["mean"], ratio=ratio,
                dt2_expected=(lo.dt_fs / hi.dt_fs) ** 2,
                per_step_expected=hi.dt_fs / lo.dt_fs,
                A_roundoff=A, B_integrator=B,
                roundoff_frac_at_dt_hi=(A / hi.dt_fs) / hi["mean"] if hi["mean"] else np.nan))
    return per, pd.DataFrame(pair)


def nve_contrast(per, precision="mixed", ref="zmm2"):
    """同 (precision, sig, dt) 下各 kernel 对 ref 的比值 + σ 检验。"""
    out = []
    for (prec, sig, dt), g in per.groupby(["precision", "sig", "dt_fs"]):
        if prec != precision or ref not in set(g.kernel):
            continue
        r = g[g.kernel == ref].iloc[0]
        for _, x in g.iterrows():
            diff = abs(r["mean"]) - abs(x["mean"])          # >0 表示 x 更好
            sd = float(np.hypot(x["sem"], r["sem"]))
            out.append(dict(
                sig=sig, dt_fs=dt, kernel=x.kernel, n_seed=int(x.n_seed),
                ratio_to_ref=abs(x["mean"]) / abs(r["mean"]),
                diff_abs=diff, sem_combined=sd,
                sigma=diff / sd if sd else np.nan,
                verdict=("ref" if x.kernel == ref else
                         ("噪声" if abs(diff) < (x["sem"] + r["sem"]) else "信号")),
                seed_sem_over_fit_se=x["sem"] / x["fit_se_typ"]
                if x["fit_se_typ"] else np.nan))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
def closure_summary(ref="zmm2"):
    """force_audit 的 CSV 归一 + PCF 的 O(c²) 拟合。

    ref 不在某个 run 里时（例如只扫 pcf1_c* 的那次），用**同轨迹同精度**的其它 run
    补基线 —— 这正是那次输出没有归一表的原因。
    """
    frames = []
    for f in sorted(glob.glob(paths.data("closure_force_audit_*.csv"))):
        d = pd.read_csv(f)
        d["src"] = os.path.basename(f)
        frames.append(d)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    alld = pd.concat(frames, ignore_index=True)
    if "kernel" not in alld:
        return alld, pd.DataFrame()

    metrics = [c for c in ("rel_rms", "rel_water_O", "rel_water_H",
                           "rel_protein_heavy", "dE_mean") if c in alld]
    # 基线**按 src 各自算** —— 不同体系/轨迹的绝对误差差一个量级，跨 src 共用基线
    # 会把 1CKK 的行normalize到 1AAY 的 zmm2 上，出来的比值毫无意义。
    # 只有该 src 自己没有 ref 行时（例如只扫 pcf1_c* 的那次），才借用同轨迹的其它 run。
    norm = alld[["src", "kernel"] + metrics].copy()
    for c in metrics:
        norm[c + "_over_" + ref] = np.nan
    for src, g in alld.groupby("src"):
        own = g[g.kernel == ref]
        if len(own):
            base, base_src = own.iloc[0][metrics], src
        else:
            # 借基线：只接受**同一条轨迹 tag** 的 run，避免跨体系
            tag = re.sub(r"_\d+k[0-9a-f]+\.csv$", "", src)
            cand = alld[alld.src.str.startswith(tag) & (alld.kernel == ref)]
            if not len(cand):
                continue
            base, base_src = cand.iloc[0][metrics], cand.iloc[0]["src"]
        idx = norm.src == src
        for c in metrics:
            norm.loc[idx, c + "_over_" + ref] = norm.loc[idx, c] / base[c]
        norm.loc[idx, "ref_from"] = base_src

    # PCF 的 O(c²)：χ_{l,c}=χ_ZMM+O(c²) ⇒ 任何光滑判据在 c=0 一阶导为零
    pcf = []
    m = alld.kernel.str.match(r"^pcf(\d+)_c([0-9.]+)$", na=False)
    for (ell,), g in alld[m].assign(
            ell=alld[m].kernel.str.extract(r"^pcf(\d+)_c")[0],
            c=alld[m].kernel.str.extract(r"_c([0-9.]+)$")[0].astype(float)
    ).groupby(["ell"]):
        g = g.sort_values("c")
        if 0.0 not in set(g.c):
            continue
        z = g[g.c == 0.0].iloc[0]
        for _, x in g[g.c > 0].iterrows():
            pcf.append(dict(ell=int(ell), c=x.c,
                            rel_rms=x.rel_rms,
                            frac_worse=x.rel_rms / z.rel_rms - 1.0,
                            over_c2=(x.rel_rms / z.rel_rms - 1.0) / x.c ** 2))
    return norm, pd.DataFrame(pcf)


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="zmm2", help="归一/对照的基线 kernel")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的产物")
    a = ap.parse_args(argv)

    per, pair = nve_summary()
    if len(per):
        log("=" * 78); log("NVE：逐 (precision, kernel 集合, dt) 聚合"); log("=" * 78)
        log(per.drop(columns=["src"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        log("")
        log("=" * 78); log("NVE：dt 配对标度 + A/dt + B dt² 拆解"); log("=" * 78)
        log(pair.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        log("\n判据：dt 减半 ⇒ ratio→0.25（真积分误差）或 →2.0（每步舍入）；符号必须不变。")
        con = nve_contrast(per, ref=a.ref)
        if len(con):
            log("")
            log("=" * 78); log(f"NVE：对 {a.ref} 的比值与 σ 检验"); log("=" * 78)
            log(con.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
            log("\n⚠ seed_sem_over_fit_se 是「seed 间 sem / 单条 run 的拟合 se」——"
                "\n   它 >1 的倍数就是「只看拟合 se 会把不确定度低估多少倍」（教训 #1）。")

    norm, pcf = closure_summary(ref=a.ref)
    if len(norm):
        log("")
        log("=" * 78); log(f"force_audit：对 {a.ref} 归一（<1 更好）"); log("=" * 78)
        cols = ["src", "kernel"] + [c for c in norm.columns
                                    if c.endswith("_over_" + a.ref)] + ["ref_from"]
        cols = [c for c in cols if c in norm.columns]
        sh = norm[cols].copy()
        sh["src"] = sh["src"].str.replace("closure_force_audit_", "", regex=False)
        sh["ref_from"] = sh.get("ref_from", pd.Series(dtype=object)).astype(str)\
            .str.replace("closure_force_audit_", "", regex=False)
        sh["借基线"] = np.where(sh["src"] != sh["ref_from"], "←借", "")
        log(sh.drop(columns=["ref_from"]).to_string(index=False,
                                                    float_format=lambda v: f"{v:.3f}"))
    if len(pcf):
        log("")
        log("=" * 78); log("PCF：劣化 / c²（c→0 收敛到常数 ⇒ c=0 是驻点）"); log("=" * 78)
        log(pcf.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    if not a.no_write:
        for name, df in (("cross_run_nve_per_config.csv", per),
                         ("cross_run_nve_dt_pairs.csv", pair),
                         ("cross_run_nve_contrast.csv", nve_contrast(per, ref=a.ref)),
                         ("cross_run_closure_normalized.csv", norm),
                         ("cross_run_pcf_c2_scaling.csv", pcf)):
            if len(df):
                fn = paths.result_new(name, force=a.force)
                df.to_csv(fn, index=False)
                log(f"-> {os.path.basename(fn)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
