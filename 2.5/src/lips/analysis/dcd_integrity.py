"""DCD 完整性校验 —— 两道，缺一不可。

2026-09-02 两次真实损坏，坏法不同，所以需要两道不同的检查：

  ① **并发交错**：两个进程写同一路径，帧被交错。
     现象：头里 NSET 仍是 2500，而文件大小对应 3379/4258 帧。
     -> 靠**自洽性**抓：头声称的帧数 vs 文件大小推算的帧数。

  ② **对活文件 mv**：同文件系统 mv 只改目录项、inode 不变，写句柄跟着走，
     文件被从头重写。现象：头和大小**都**变成 156 帧 —— **自洽,但被截断**。
     -> 自洽性抓不到，只能靠**期望帧数**抓（由 prod_ns / report_ps 推出）。

两种情况 mdtraj 都只打 warning、`md.load()` 照样返回数据，分析脚本会静默给出
垃圾结论。所以两道都必须是硬失败。

原子数**从 DCD 自己读**，不由调用方传 —— 传错了帧长就算错，会把好文件误判成坏的
（第一版就是这么错的：传了 32794，而 PME 那批其实是另一个原子数的拓扑）。
"""
from __future__ import annotations
import os, struct


def dcd_header(path):
    """返回 (nset, natoms)。NSET 在偏移 8；NATOM 在标题块之后的第三个记录里。"""
    with open(path, "rb") as f:
        blk = f.read(4)
        if len(blk) < 4:
            raise RuntimeError(f"{path}: 文件太短，读不到头")
        f.seek(8)
        nset = struct.unpack("<i", f.read(4))[0]
        # 跳过第一个记录（4 + 84 + 4），再跳标题块，然后读 NATOM
        f.seek(0)
        n1 = struct.unpack("<i", f.read(4))[0]
        f.seek(4 + n1 + 4)
        n2 = struct.unpack("<i", f.read(4))[0]
        f.seek(4 + n1 + 4 + 4 + n2 + 4)
        n3 = struct.unpack("<i", f.read(4))[0]
        natoms = struct.unpack("<i", f.read(4))[0]
        header_len = 4 + n1 + 4 + 4 + n2 + 4 + 4 + n3 + 4
    return nset, natoms, header_len


def check_dcd(path, expected_frames=None, strict=True):
    """两道校验。expected_frames 给了就一并查截断。

    ⚠ 第二个参数是**期望帧数**，不是原子数。第一版的签名是 (path, n_atoms)，
      重写后含义变了而调用点没改，于是"期望 32818 帧"，把好文件全拦下。
      下面加了个范围保护：帧数不可能大到原子数那个量级。
    """
    size = os.path.getsize(path)
    nset, natoms, hdr = dcd_header(path)
    if expected_frames is not None and expected_frames == natoms:
        raise TypeError(
            f"check_dcd 的第二个参数是**期望帧数**，不是原子数。"
            f"收到 {expected_frames}，正好等于该文件的原子数 —— 多半传错了。")
    problems = []
    n_from_size = None
    for cell in (True, False):
        fs = (56 if cell else 0) + 3 * (8 + 4 * natoms)
        if fs > 0 and (size - hdr) % fs == 0:
            n_from_size = (size - hdr) // fs
            frame_size, has_cell = fs, cell
            break
    if n_from_size is None:
        problems.append(f"文件大小 {size} 减去头 {hdr} 后不是帧长的整数倍"
                        f"（natoms={natoms}）")
    elif n_from_size != nset:
        problems.append(f"**自洽性失败**：头声称 {nset} 帧，按大小推算 {n_from_size} 帧"
                        f" —— 典型是两个进程写过同一路径（帧交错）")
    if expected_frames is not None and n_from_size is not None \
            and n_from_size != expected_frames:
        problems.append(f"**帧数不符**：实际 {n_from_size} 帧，期望 {expected_frames} 帧"
                        f" —— 典型是被截断（例如对正在被写的文件做过 mv）")
    info = dict(path=os.path.basename(path), size=size, natoms=natoms,
                header_len=hdr, n_header=nset, n_from_size=n_from_size,
                expected=expected_frames, problems=problems)
    if problems and strict:
        raise RuntimeError(f"DCD 完整性校验失败: {os.path.basename(path)}\n  "
                           + "\n  ".join(problems) +
                           "\n  这类文件 mdtraj 只警告不报错，会静默污染结论 —— 拒绝使用。")
    return (not problems), info


def flag_frame_count_outliers(paths, tolerance=0.5):
    """在一批同协议轨迹里挑出帧数离群的那些，返回 {path: 说明}。

    **为什么需要这个**：`check_dcd` 的自洽性检查抓不到截断——被 `mv` 截断的文件
    头尾自洽（实测 salvage/ 那条：头声称 156 帧、按体积推算也是 156 帧，连同源
    `_state.csv` 也是 156 行，三处一致）。截断只能靠"期望帧数"抓，而那个信息不在
    文件里。

    这里用的是同批比较：一条 156 帧的轨迹夹在一堆 2500 帧的同批轨迹里，本身就是
    信号。这是**启发式，不是证明**——同批里长度本来就不同（比如故意跑短的对照）
    会误报，所以只报告、不拦截。真要卡死请给 check_dcd 传 expected_frames。
    """
    counts = {}
    for path in paths:
        try:
            _, info = check_dcd(path, strict=False)
        except Exception:
            continue
        counts[path] = info["n_header"]
    if len(counts) < 3:
        return {}
    ordered = sorted(counts.values())
    median = ordered[len(ordered) // 2]
    if median <= 0:
        return {}
    return {path: (f"{n} 帧，同批中位数 {median} 帧"
                   f"（差 {abs(n - median) / median:.0%}）—— 可能被截断，"
                   f"要确认请用 check_dcd(..., expected_frames=...)")
            for path, n in counts.items()
            if abs(n - median) / median > tolerance}


def main():
    """命令行：逐个校验 dcd，任何一个不通过就非零退出。

    这个模块原本只是个被 import 的库，没有 CLI；但"轨迹是不是完整的"是每次
    分析前最该先问的问题，值得能单独跑。
    """
    import argparse
    import glob as _glob

    from lips.paths import DATA_DIR

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*",
                    help="dcd 路径或 glob；不给则扫 DATA_DIR 下所有 *.dcd")
    ap.add_argument("--expect-frames", type=int, default=None,
                    help="期望帧数；给了就一并核对")
    ap.add_argument("--strict", action="store_true",
                    help="发现问题即抛异常（默认只汇报）")
    args = ap.parse_args()

    targets = []
    for pattern in (args.paths or [str(DATA_DIR / "*.dcd")]):
        targets.extend(sorted(_glob.glob(pattern)) or [pattern])

    bad = 0
    for path in targets:
        try:
            ok, info = check_dcd(path, expected_frames=args.expect_frames,
                                 strict=args.strict)
        except Exception as exc:
            print(f"✗ {os.path.basename(path)}: {exc}")
            bad += 1
            continue
        mark = "✓" if ok else "✗"
        print(f"{mark} {info['path']}  natoms={info['natoms']} "
              f"frames(header)={info['n_header']} frames(size)={info['n_from_size']}")
        for problem in info["problems"]:
            print(f"    - {problem}")
        bad += 0 if ok else 1

    print(f"\n{len(targets) - bad}/{len(targets)} 通过")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
