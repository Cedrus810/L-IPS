#!/usr/bin/env python
"""从已有轨迹实测电荷结构因子 S_ZZ(k)，并用它重跑 closure 判据。

**不跑任何 MD**：只读已有的 dcd（默认目录由 L_IPS_DATA_DIR 指定）。

体系是参数，不是硬编码：用 `--system`（见 SYSTEM_BUILDERS）或
`--topology/--system-xml` 指定任意预建体系。默认仍是 1CKK，因为现存的那批
LONG5ns 轨迹是 1CKK 的——数据文件名不改，脚本对它的假设改成了 `--dcd-pattern`。

背景：`STAGE1_CLOSURE_KILLSWITCH_REPORT.md` 的全部数值建立在解析 S_ZZ 模型上，
而该模型把电荷当无结构屏蔽流体，看不见"电荷绑在中性刚性基团上"——这正是
Sakuraba ZMM 逐阶消去排除区多极矩所针对的东西。所以判据把 ell 排序排反了
（我说 ell=1 最好，论文在同一工作点 rc=1.2nm/α=0 推荐 ell=2/3）。

本脚本两阶段：
  阶段 A  实测 S_ZZ(k) = <|ρ̂(k)|²> / Σ_j q_j²,  ρ̂(k)=Σ_j q_j e^{-ik·r_j}
          k 取盒子倒格矢（逐帧按该帧盒子重算，NPT 安全），按 |k| 分壳平均。
          **默认在 GPU 上跑（torch, float32）**，实测比 numpy 快约 48×
          （31358 原子、2268 个 k：5.5 s/帧 -> 0.115 s/帧），S_ZZ 相对差 1.8e-6，
          分箱计数逐位相同。`--device numpy` 可切回原来的 CPU float64 路径。
  阶段 B  用实测 S_ZZ 重算 E_F 表，做**判据自身的证伪测试**：
            排序变成 ell=2/3 优于 ell=1  -> 判据修好了，可以继续 kill switch 判定
            仍然 ell=1 赢                -> E_F 的公式或记账本身有错，先修判据

用法（节点上一条命令全跑）：

    cd /home/ruigengji/L-IPS/2.5
    L_IPS_DATA_DIR=/home/ruigengji/L-IPS/2.4 \\
      python measure_szz.py 2>&1 | tee logs/szz_$(date +%Y%m%d_%H%M%S).log

可断点续跑：已经写出 szz_*.csv 的轨迹会自动跳过（--force 可强制重算）。
只想看阶段 B（S_ZZ 已经测过）：加 --skip-measure。
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from lips import paths
from lips.analysis import dcd_integrity
from lips import systems as ls

from lips.paths import PROJECT_ROOT, DATA_DIR as _DATA_DIR
SCRIPT_DIR = str(PROJECT_ROOT)   # 兼容旧写法；新代码请直接用 lips.paths
DATA_DIR = str(_DATA_DIR)

VERDICT_CSV = "szz_closure_verdict.csv"

# 每个体系的轨迹命名约定。数据文件名一律不改，这里只记录脚本该怎么找它们。
DCD_PATTERNS = {
    "1aay": "{label}_traj.dcd",
    "1aay-amoeba": "{label}_traj.dcd",
    "1ckk": "{label}_1ckk.dcd",       # 旧数据在 L_IPS_DATA_DIR 指的 2.4 里
}
DEFAULT_DCD_PATTERN = "{label}_traj.dcd"

# 1CKK 时代那 12 条的显式清单，仅在 --system 1ckk 时作为默认。
LEGACY_1CKK_LABELS = [
    f"LONG5ns_{m}_seed{s}"
    for m in ("PME", "CWLD_aq0_ca_w1", "CWLD_aq0_ca_w2", "CWLD_aq0p5_ca_w2")
    for s in (0, 1, 2)
]


def resolve_dcd_pattern(args):
    if args.dcd_pattern:
        return args.dcd_pattern
    if args.topology or args.system_xml:
        return DEFAULT_DCD_PATTERN
    return DCD_PATTERNS.get(args.system, DEFAULT_DCD_PATTERN)


def discover_labels(pattern):
    """按命名模板在 DATA_DIR 里发现现有轨迹，返回 label 列表。

    比硬编码清单稳：新跑一批轨迹不用回来改脚本，也不会因为清单里某条已经删了
    而一路 [skip] 到底还让人以为跑过了。
    """
    prefix, _, suffix = pattern.partition("{label}")
    hits = sorted(glob.glob(os.path.join(DATA_DIR, f"{prefix}*{suffix}")))
    labels = []
    for path in hits:
        name = os.path.basename(path)
        if suffix and not name.endswith(suffix):
            continue
        labels.append(name[len(prefix):len(name) - len(suffix)] if suffix
                      else name[len(prefix):])
    return labels


def safe_name(label):
    """标签可以带子目录（如 0706/LIPS_fixedQ_baseline），文件名里把 / 换掉。"""
    return label.replace("/", "__").replace(os.sep, "__")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 阶段 A：实测 S_ZZ(k)
# ---------------------------------------------------------------------------
def build_kvectors(box_nm, kmax, max_per_shell, dk, kfine=6.0, cap_lowk=256):
    """挑一组整数三元组 n（k = n·B），按 |k| 分壳、每壳限量。

    整数三元组一次选定、全程复用；k 本身逐帧按该帧盒子重算，所以 NPT 下
    每条轨迹的壳归属会随盒子微动——这由逐帧分箱处理，不是近似。
    """
    L = np.array([box_nm[0, 0], box_nm[1, 1], box_nm[2, 2]], float)
    n_max = int(np.ceil(kmax * L.max() / (2 * np.pi))) + 1
    rng = np.arange(-n_max, n_max + 1)
    n1, n2, n3 = np.meshgrid(rng, rng, rng, indexing="ij")
    n = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=1)
    n = n[np.any(n != 0, axis=1)]                      # 去掉 k=0

    recip = 2 * np.pi * np.linalg.inv(box_nm).T
    kmag = np.linalg.norm(n @ recip, axis=1)
    n = n[kmag <= kmax]
    kmag = kmag[kmag <= kmax]

    # 每壳限量，确定性抽样（等间隔取，不用随机数，保证可复现）
    keep = []
    nb = int(np.ceil(kmax / dk))
    for b in range(nb):
        idx = np.where((kmag >= b * dk) & (kmag < (b + 1) * dk))[0]
        if idx.size == 0:
            continue
        # 分段配额：E_F 的权重几乎全在低 k（实测 ~90% 在 k<5），
        # 而高 k 壳的格点数 ∝ k² 最多——不分段的话算力全花在贡献 0.7% 的区间。
        # 但 kmax 仍必须给到 ~40，因为 S_ZZ 要到 k≈27 才升到 1，
        # 高 k 的 "S_ZZ=1" 外推在更小的 kmax 上站不住。
        cap = cap_lowk if (b + 0.5) * dk < kfine else max_per_shell
        if idx.size > cap:
            idx = idx[np.linspace(0, idx.size - 1, cap).astype(int)]
        keep.append(idx)
    keep = np.concatenate(keep)
    return n[keep].astype(np.float64)


def resolve_device(requested):
    """把 --device 解析成 ('cuda'|'cpu', torch 模块或 None)。

    'auto' 有 CUDA 就用，没有就静默回落 CPU；显式 'cuda' 拿不到则报错退出，
    不静默降级——跟 test.zsh 里"平台固定 CUDA，不可用就退出"同一个口径。
    """
    if requested == "numpy":
        return "numpy", None
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise SystemExit("--device cuda 需要 torch，但 import torch 失败")
        return "numpy", None
    if requested in ("auto", "cuda"):
        if torch.cuda.is_available():
            return "cuda", torch
        if requested == "cuda":
            raise SystemExit("--device cuda 但 torch.cuda.is_available() 为 False")
        return "numpy", None
    return "cpu", torch


def accumulate_frames_torch(torch, device, xyz, uv, n_int, charges, edges, nb, blk,
                            progress, log_fn):
    """阶段 A 的每帧累加，GPU/torch 版。返回 (acc, acc2, cnt, ksum) 的 numpy 数组。

    数值精度：**重活全部 float32**——相位、cos/sin、以及 q·cos(ph) 这两个
    (N × blk) 规模的乘加。这不是妥协：dcd 里存的坐标本来就是 float32，把它
    升到 float64 只是给已经没有的位数配更宽的容器。

    只有那四个**长度 nb（约 80）的直方图累加器**用 float64。它们不是算力，
    是记账：`var = acc2/cnt - mean²` 是标准的灾难性相消式子，而 acc2 会累到
    很大的量级。80 个 double 在 GPU 上不花任何时间，却把方差的可信度买回来。
    """
    n_frames = xyz.shape[0]
    dev = torch.device(device)
    pos_all = torch.as_tensor(xyz, dtype=torch.float32)
    q = torch.as_tensor(charges, dtype=torch.float32, device=dev)
    n_int_t = torch.as_tensor(n_int, dtype=torch.float32, device=dev)
    edges_t = torch.as_tensor(edges, dtype=torch.float32, device=dev)
    uv_t = torch.as_tensor(uv, dtype=torch.float32)

    acc = torch.zeros(nb, dtype=torch.float64, device=dev)
    acc2 = torch.zeros(nb, dtype=torch.float64, device=dev)
    cnt = torch.zeros(nb, dtype=torch.float64, device=dev)
    ksum = torch.zeros(nb, dtype=torch.float64, device=dev)

    two_pi = float(2 * np.pi)
    t0 = time.time()
    for f in range(n_frames):
        box = uv_t[f].to(dev)
        # 逐帧按该帧盒子重算倒格矢，跟 numpy 路径一致（NPT 下盒子会动）。
        recip = two_pi * torch.linalg.inv(box).T
        kv = n_int_t @ recip                             # (nk, 3)
        kmag = torch.linalg.vector_norm(kv, dim=1)
        pos = pos_all[f].to(dev, non_blocking=True)      # (N, 3)

        rho2 = torch.empty(kv.shape[0], dtype=torch.float32, device=dev)
        for a in range(0, kv.shape[0], blk):
            kb = kv[a:a + blk]
            ph = pos @ kb.T                              # (N, blk)
            re = q @ torch.cos(ph)
            # sin 就地写回 ph，省掉一份 (N, blk) 的峰值显存。
            torch.sin_(ph)
            im = q @ ph
            rho2[a:a + blk] = re * re + im * im

        # torch.bucketize 的 right=False 等价于 np.digitize 的 right=False。
        b = torch.clamp(torch.bucketize(kmag, edges_t, right=False) - 1, 0, nb - 1)
        acc.index_add_(0, b, rho2.double())
        acc2.index_add_(0, b, (rho2.double()) ** 2)
        cnt.index_add_(0, b, torch.ones_like(kmag, dtype=torch.float64))
        ksum.index_add_(0, b, kmag.double())

        if progress and (f + 1) % progress == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            el = time.time() - t0
            log_fn(f"       frame {f+1}/{n_frames}  {el:.0f}s "
                   f"(eta {el/(f+1)*(n_frames-f-1):.0f}s)")
    if device == "cuda":
        torch.cuda.synchronize()
    return (acc.cpu().numpy(), acc2.cpu().numpy(),
            cnt.cpu().numpy(), ksum.cpu().numpy())


def measure_one(dcd_path, md_top, charges, args):
    import mdtraj as md

    t0 = time.time()
    traj = md.load(dcd_path, top=md_top, stride=args.stride)
    uv = traj.unitcell_vectors
    if uv is None:
        raise SystemExit(f"{dcd_path} 没有盒子信息，无法定义倒格矢")
    xyz = traj.xyz.astype(np.float64)
    n_frames = xyz.shape[0]
    log(f"       {n_frames} frames (stride={args.stride}), "
        f"box0 = {np.diag(uv[0]).round(3)} nm, load {time.time()-t0:.1f}s")

    n_int = build_kvectors(uv[0].astype(np.float64), args.kmax,
                           args.max_per_shell, args.dk,
                           kfine=args.kfine, cap_lowk=args.cap_lowk)
    log(f"       {n_int.shape[0]} k-vectors, |k| <= {args.kmax} nm^-1 "
        f"(k<{args.kfine} 每壳<={args.cap_lowk}, 以上每壳<={args.max_per_shell})")

    nb = int(np.ceil(args.kmax / args.dk))
    edges = np.arange(nb + 1) * args.dk
    acc = np.zeros(nb)          # Σ |ρ̂|²
    acc2 = np.zeros(nb)         # Σ |ρ̂|⁴（给标准误）
    cnt = np.zeros(nb)
    ksum = np.zeros(nb)
    q2tot = float(np.sum(charges ** 2))

    blk = args.kblock
    device, torch = resolve_device(args.device)
    t0 = time.time()
    if device in ("cuda", "cpu"):
        name = (torch.cuda.get_device_name(0) if device == "cuda" else "torch CPU")
        log(f"       engine: torch/{device} float32 ({name}), kblock={blk}")
        acc, acc2, cnt, ksum = accumulate_frames_torch(
            torch, device, xyz, uv, n_int, charges, edges, nb, blk, args.progress, log)
    else:
        log(f"       engine: numpy CPU float64, kblock={blk}")
        for f in range(n_frames):
            recip = 2 * np.pi * np.linalg.inv(uv[f].astype(np.float64)).T
            kv = n_int @ recip                                   # (nk,3)
            kmag = np.linalg.norm(kv, axis=1)
            pos = xyz[f]
            rho2 = np.empty(kv.shape[0])
            for a in range(0, kv.shape[0], blk):
                ph = pos @ kv[a:a + blk].T                       # (N, blk)
                re = charges @ np.cos(ph)
                im = charges @ np.sin(ph)
                rho2[a:a + blk] = re * re + im * im
            b = np.clip(np.digitize(kmag, edges) - 1, 0, nb - 1)
            np.add.at(acc, b, rho2)
            np.add.at(acc2, b, rho2 ** 2)
            np.add.at(cnt, b, 1.0)
            np.add.at(ksum, b, kmag)
            if args.progress and (f + 1) % args.progress == 0:
                el = time.time() - t0
                log(f"       frame {f+1}/{n_frames}  {el:.0f}s "
                    f"(eta {el/(f+1)*(n_frames-f-1):.0f}s)")
    log(f"       accumulate {time.time()-t0:.1f}s "
        f"({n_frames/(time.time()-t0):.1f} frames/s)")

    good = cnt > 0
    mean = acc[good] / cnt[good]
    var = np.maximum(acc2[good] / cnt[good] - mean ** 2, 0.0)
    return pd.DataFrame({
        "k_nm^-1": ksum[good] / cnt[good],
        "S_ZZ": mean / q2tot,
        "S_ZZ_stderr": np.sqrt(var / cnt[good]) / q2tot,
        "n_samples": cnt[good].astype(int),
    })


def stage_a(args):
    import mdtraj as md

    topology, sys_pme = ls.load_reference_system_from_args(args)
    md_top = md.Topology.from_openmm(topology)


    charges = ls.base_charges(sys_pme)
    log(f"[build] {len(charges)} atoms, Σq = {charges.sum():+.3e} e, "
        f"Σq² = {np.sum(charges**2):.2f} e²")
    log("[note] 用的是 qbase（力场静电荷）。CWLD 轨迹里 Q 会随环境浮动，"
        "但跨方法比较必须用同一个电荷定义，且 Δq 有 clamp。")

    pattern = resolve_dcd_pattern(args)
    if args.labels:
        labels = args.labels
    elif args.system == "1ckk" and not (args.topology or args.system_xml):
        labels = LEGACY_1CKK_LABELS
    else:
        labels = discover_labels(pattern)
        if not labels:
            raise SystemExit(
                f"在 {DATA_DIR} 里没发现匹配 {pattern!r} 的轨迹。"
                f"用 --labels 显式指定，或用 L_IPS_DATA_DIR 指向数据目录。")
        log(f"[scan] 在 {DATA_DIR} 发现 {len(labels)} 条匹配 {pattern!r} 的轨迹")
        if not args.skip_integrity_check:
            outliers = dcd_integrity.flag_frame_count_outliers(
                [os.path.join(DATA_DIR, pattern.format(label=l)) for l in labels])
            for path, why in outliers.items():
                log(f"[scan] ⚠ {os.path.basename(path)}: {why}")
    for i, label in enumerate(labels, 1):
        out = paths.result(f"szz_{safe_name(label)}.csv")
        dcd = os.path.join(DATA_DIR, pattern.format(label=label))
        log(f"\n[{i}/{len(labels)}] {label}")
        if not os.path.exists(dcd):
            log(f"       [skip] 找不到 {dcd}")
            continue
        if os.path.exists(out) and not args.force:
            log(f"       [skip] 已有 {os.path.basename(out)}（--force 可重算）")
            continue
        # 闸门一：完整性。坏 dcd 有两种坏法，mdtraj 对两种都只警告不报错：
        #   帧交错（两个进程并发写同一路径，头声称 2500 实际 3379/4258）
        #   截断（mv 活文件，只剩 156 帧；文件本身自洽，只靠自洽性抓不到）
        # 两种都会静默污染结论，所以在读之前先过 dcd_integrity。
        if not args.skip_integrity_check:
            try:
                ok, dinfo = dcd_integrity.check_dcd(dcd, strict=False)
            except Exception as exc:
                log(f"       [skip] 完整性校验读不动: {exc}")
                continue
            if not ok:
                log(f"       [skip] ⚠ dcd 完整性校验未通过，拒绝使用：")
                for problem in dinfo["problems"]:
                    log(f"                - {problem}")
                log(f"                （--skip-integrity-check 可强行绕过，但基本不该用）")
                continue

        # 闸门二：原子数。自动发现会捞到别的体系的轨迹（例如 ZN_water_* 和
        # ZN_1aay_* 同一个命名模板）。原子数对不上就说明这条轨迹不属于当前参考
        # 体系，用它的 qbase 会整体错位——必须挡掉，不能算出一条看着正常的假曲线。
        try:
            probe = md.open(dcd)
            n_traj_atoms = probe.read(1)[0].shape[1]
            probe.close()
        except Exception as exc:
            log(f"       [skip] 读不动 {os.path.basename(dcd)}: {exc}")
            continue
        if n_traj_atoms != md_top.n_atoms:
            log(f"       [skip] 原子数不符：轨迹 {n_traj_atoms} vs 参考体系 "
                f"{md_top.n_atoms}")
            snapshot = os.path.join(DATA_DIR, f"{label}_top.pdb")
            if os.path.exists(snapshot):
                log(f"                这条轨迹自带拓扑快照 {os.path.basename(snapshot)}，"
                    f"但它的 System(qbase) 没有留存，无法算 S_ZZ")
            elif n_traj_atoms == 32818 and md_top.n_atoms == 32794:
                log(f"                这正是 2026-09-03 那次 1AAY/amber/solvated.pdb "
                    f"被从 32818 覆盖成 32794 的后果：老轨迹没有匹配的 System 了。"
                    f"不是这个脚本的问题，也**不能**用现在的 qbase 硬算——会整体错位。")
            else:
                log(f"                多半是别的体系的轨迹（如 ZN_water_*）碰巧匹配了"
                    f"文件名模板")
            continue
        df = measure_one(dcd, md_top, charges, args)
        df.to_csv(out, index=False)
        log(f"       -> {os.path.basename(out)}")


# ---------------------------------------------------------------------------
# 阶段 B：用实测 S_ZZ 重跑判据 + 证伪测试
# ---------------------------------------------------------------------------
def stage_b(args):
    from lips.engine import closure as cw
    from lips.analysis import kill_switch as ks

    # 注意排除本脚本自己的输出，否则第二次运行会把 verdict 表当成 S_ZZ 表读
    files = sorted(f for f in glob.glob(paths.result("szz_*.csv"))
                   if os.path.basename(f) != VERDICT_CSV)
    if not files:
        log("[stage B] 没有 szz_*.csv，跳过")
        return
    log("\n" + "=" * 78)
    log("阶段 B：用实测 S_ZZ 重算 E_F —— 判据自身的证伪测试")
    log("=" * 78)
    log("已知真值：Sakuraba ZMM 论文在同一工作点(rc=1.2nm, α=0)推荐 ell=2/3。")
    log("若下表 zmm2/zmm3 仍差于 zmm1，则 E_F 的公式或记账有错，不只是模型问题。\n")

    chi = {f"zmm{e}": cw.zmm_window(e).chi(ks.X_UNIT) for e in (1, 2, 3)}
    rows = []
    hdr = (f"{'trajectory':38s} {'kappa':>7s} {'zmm1':>8s} {'zmm2':>8s} "
           f"{'zmm3':>8s} {'opt2':>8s} {'外推占比':>9s} {'判据':>6s}")
    log(hdr)
    log("-" * len(hdr))
    for fp in files:
        label = os.path.basename(fp)[4:-4]
        df = pd.read_csv(fp)
        szz, info = ks.make_szz_from_table(df["k_nm^-1"].values, df["S_ZZ"].values)
        w = ks.ef_weight(szz)
        share = ks.ef_extrapolation_share(szz, info)
        ref = ks.EF_from_weight(chi["zmm2"], w)
        v = {n: ks.EF_from_weight(c, w) / ref for n, c in chi.items()}
        best = np.inf
        for N in range(0, 15):
            ef, _, cond, _ = ks.optimal_window_from_weight(2, N, w)
            if cond > 1e13:
                break
            best = min(best, ef / ref)
        order = " < ".join(n for n, _ in sorted(v.items(), key=lambda kv: kv[1]))
        # 有效性判定：用被测 closure 自己生成的轨迹去验该 closure 是**循环论证**。
        # 只有全 Ewald(PME) 轨迹才是独立参照。CWLD/L-IPS 轨迹仍然测、仍然报，
        # 但不计入证伪测试。
        is_ref = args.reference_pattern in label
        ok = ("PASS" if v["zmm2"] < v["zmm1"] else "FAIL") if is_ref else "n/a(非参照)"
        tot_share = share["low_k_share"] + share["high_k_share"]
        log(f"{label:38s} {info['kappa_nm^-1']:7.2f} {v['zmm1']:8.4f} "
            f"{v['zmm2']:8.4f} {v['zmm3']:8.4f} {best:8.4f} "
            f"{tot_share:9.2%} {ok:>6s}")
        rows.append(dict(trajectory=label, kappa_nm_inv=info["kappa_nm^-1"],
                         k_min=info["k_min"], k_max=info["k_max"],
                         EF_zmm1=v["zmm1"], EF_zmm2=v["zmm2"], EF_zmm3=v["zmm3"],
                         EF_opt_ell2=best,
                         gain_rms_opt=float(np.sqrt(1.0 / best)),
                         extrap_low_k_share=share["low_k_share"],
                         extrap_high_k_share=share["high_k_share"],
                         ell_order_best_first=order,
                         is_reference=is_ref,
                         criterion_matches_sakuraba=(ok == "PASS") if is_ref else None))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(SCRIPT_DIR, VERDICT_CSV), index=False)

    # E_F 权重的 k 分布 —— 决定 kmax 够不够、低 k 外推重不重要
    df0 = pd.read_csv(files[0])
    szz0, info0 = ks.make_szz_from_table(df0["k_nm^-1"].values, df0["S_ZZ"].values)
    kk = ks.Q_NODES / ks.RC_NM
    c2 = ks.chat_of_values(chi["zmm2"]) ** 2
    w0 = szz0(kk) * ks.Q_WEIGHTS
    tot = float(w0 @ c2)
    log(f"\nE_F 被积函数按 k 段的占比（zmm2, {os.path.basename(files[0])}）：")
    log("  这决定了算力该花在哪——占比高的区间必须有足够样本，占比低的可以砍")
    for lo, hi in [(0.0, info0["k_min"]), (info0["k_min"], 5.0), (5.0, 15.0),
                   (15.0, 40.0), (40.0, np.inf)]:
        m = (kk >= lo) & (kk < hi)
        tag = " <- 纯外推" if hi == info0["k_min"] else ""
        log(f"    k∈[{lo:>6.3f},{hi:>6.1f}) : {float((w0 * m) @ c2)/tot:7.2%}{tag}")

    log("\n" + "=" * 78)
    ref = out[out["is_reference"]]
    nonref = out[~out["is_reference"]]
    n_pass = int(ref["criterion_matches_sakuraba"].sum()) if len(ref) else 0
    log(f"证伪测试（只用参照轨迹，pattern='{args.reference_pattern}'）："
        f"{n_pass}/{len(ref)} 条给出 zmm2 优于 zmm1")
    if len(nonref):
        log(f"  另有 {len(nonref)} 条由被测 closure 自己生成的轨迹，不计入"
            f"（循环论证）；其 kappa = "
            f"{nonref['kappa_nm_inv'].min():.1f}-{nonref['kappa_nm_inv'].max():.1f}"
            f" vs 参照 {ref['kappa_nm_inv'].min():.1f}-{ref['kappa_nm_inv'].max():.1f}"
            if len(ref) else "")
    if len(ref) and n_pass == len(ref):
        marg = float((ref["EF_zmm1"] - ref["EF_zmm2"]).mean())
        log(f"  -> zmm2 优于 zmm1，与论文偏好 ell>=2 一致。但 margin 只有 {marg:+.3f}"
            f"（E_F 相对量），**不是一个决定性的确认**。")
        log(f"     排序（最好在前）：{ref['ell_order_best_first'].mode()[0]}")
        if float(ref["EF_zmm3"].mean()) > 1.0:
            log("     ⚠ zmm3 在本判据下明显差于 zmm2。论文注释记的是"
                "「ell=2/3 是甜点」，")
            log("       其中 ZOct 那句讲的是相对 SPME 的**提速**(16-44%)，不必然是精度。")
            log("       所以这未必是矛盾——要确认得回去查原文对 ell=3 精度的实际主张。")
        g = float(ref["gain_rms_opt"].mean())
        log(f"     全类天花板 gain_rms = {g:.3f}x (门槛 2x) -> "
            f"{'仍 FAIL' if g < 2 else 'PASS'}")
        log(f"     另注：zmm2 的 E_F 已在理论最优的 "
            f"{1/float(ref['EF_opt_ell2'].mean()):.2f} 倍以内"
            f"（RMS 力误差意义上 {1/np.sqrt(float(ref['EF_opt_ell2'].mean())):.2f} 倍），"
            f"即现行默认已接近最优。")
    elif len(ref) == 0:
        log("  -> 没有参照轨迹（标签里不含 "
            f"'{args.reference_pattern}'），无法做证伪测试。")
    elif n_pass == 0:
        log("  -> **判据没修好**。实测 S_ZZ 也给 ell=1 赢，说明问题不在 S_ZZ 模型，")
        log("     而在 E_F 的推导或记账（最可能：未纳入 ZMM 的中和/自能约定，")
        log("     v2.6 的 U_pair 还带 -qbase_i*qbase_j*(1/r-1/rc) 抵消原生 RF）。")
        log("     **先修判据，别用本表任何数字下结论。**")
    else:
        log("  -> 结果不一致，按方法/seed 分组看上表，先查是不是某条轨迹有问题。")
    log(f"\n-> {VERDICT_CSV}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ls.add_system_arguments(ap)
    ap.add_argument("--dcd-pattern", default=None,
                    help="轨迹文件名模板，{label} 会被替换。默认按 --system 选："
                         "1aay -> '{label}_traj.dcd'，1ckk -> '{label}_1ckk.dcd'"
                         "（数据文件本身不改名）")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="轨迹标签（不含 --dcd-pattern 的后缀）。默认 12 条 LONG5ns")
    ap.add_argument("--stride", type=int, default=5, help="帧步长（默认 5 -> ~100 帧/条）")
    ap.add_argument("--kmax", type=float, default=40.0, help="最大 |k|, nm^-1")
    ap.add_argument("--dk", type=float, default=0.5, help="|k| 壳宽, nm^-1")
    ap.add_argument("--max-per-shell", type=int, default=16,
                    help="k>=--kfine 时每壳最多取几个 k（那段只值 ~0.7%% 的 E_F）")
    ap.add_argument("--kfine", type=float, default=6.0,
                    help="低 k 加密区的上界, nm^-1（E_F 权重 ~90%% 在 k<5）")
    ap.add_argument("--cap-lowk", type=int, default=256,
                    help="k<--kfine 时每壳最多取几个 k")
    ap.add_argument("--kblock", type=int, default=256, help="k 分块大小（控内存/显存）")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "cpu", "numpy"],
                    help="阶段 A 的计算后端：auto=有 CUDA 就用（默认）；"
                         "cuda=强制 GPU，拿不到就退出不静默降级；"
                         "cpu=torch CPU；numpy=原来的 numpy float64 路径")
    ap.add_argument("--progress", type=int, default=20, help="每几帧打一次进度；0 关闭")
    ap.add_argument("--force", action="store_true", help="重算已有的 szz_*.csv")
    ap.add_argument("--skip-integrity-check", action="store_true",
                    help="跳过 dcd 完整性校验。默认开着校验：坏 dcd（帧交错/截断）"
                         "mdtraj 只警告不报错，会静默污染结论")
    ap.add_argument("--skip-measure", action="store_true", help="只跑阶段 B")
    ap.add_argument("--reference-pattern", default="PME",
                    help="标签含此串的轨迹才算独立参照（默认 PME）。"
                         "用被测 closure 自己生成的轨迹验该 closure 是循环论证。")
    args = ap.parse_args()

    log(f"[info] data dir  : {DATA_DIR}")
    log(f"[info] output dir: {SCRIPT_DIR}")
    if not args.skip_measure:
        if not os.path.isdir(DATA_DIR):
            raise SystemExit(f"L_IPS_DATA_DIR 不存在: {DATA_DIR}")
        stage_a(args)
    stage_b(args)


if __name__ == "__main__":
    main()
