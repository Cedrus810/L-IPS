"""项目路径的单一来源。

**为什么需要这个**：打包之前每个脚本都写
`SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))`，然后把数据和产物都
当成"跟脚本同一个目录"。文件一旦挪进 `src/lips/`，那个假设立刻全错。路径解析
只能有一份实现，而且必须跟模块自己在哪无关。

三个目录，都可以用环境变量覆盖：

    PROJECT_ROOT  仓库根（含 pyproject.toml 的那一层）
    DATA_DIR      轨迹/原始数据所在（L_IPS_DATA_DIR），默认 PROJECT_ROOT
    RESULTS_DIR   分析产物写到哪（L_IPS_RESULTS_DIR），默认 PROJECT_ROOT

2026-09-05：轨迹和产物**已经**挪进 `<repo>/data/`（118 个文件、17 GB，
清单在 `data/MOVED_FROM_ROOT_2026-09-05.txt`，回滚就是照着清单挪回去）。
挪之前确认过没有进程在写、两小时内无改动，且所有读取都走这里的 DATA_DIR。
默认值随之改成 `PROJECT_ROOT/"data"`；显式设了 `L_IPS_DATA_DIR` 的调用方
（比如指向 2.4 旧数据的 szz）不受影响。

`data/` 是**平的**，输入和产物混在一起。分子目录会让 szz 这类"读自己写的 csv、
同时扫 dcd"的工具需要两个根，收益不抵成本。上传的新东西先进 `data/inbox/`，
那个子目录不会被任何 glob 扫到。
"""
from __future__ import annotations

import os
from pathlib import Path

_MARKERS = ("pyproject.toml", "openmm-localcwld")


def _find_project_root() -> Path:
    if (env := os.environ.get("L_IPS_PROJECT_ROOT")):
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if all((candidate / m).exists() for m in _MARKERS):
            return candidate
    # 装成 wheel 之后源码树可能不在旁边；退回当前目录，并让调用方能看出来。
    return Path.cwd().resolve()


PROJECT_ROOT: Path = _find_project_root()
_DEFAULT_DATA = PROJECT_ROOT / "data"
DATA_DIR: Path = Path(os.environ.get("L_IPS_DATA_DIR", _DEFAULT_DATA)).resolve()
RESULTS_DIR: Path = Path(os.environ.get("L_IPS_RESULTS_DIR", _DEFAULT_DATA)).resolve()
INBOX_DIR: Path = DATA_DIR / "inbox"      # 上传落点；分析的 glob 不扫这里
SYSTEMS_DIR: Path = PROJECT_ROOT          # 预建体系（1AAY/ 等）就在根下


def run_sig(*bits, n: int = 6) -> str:
    """把「影响物理的开关」压成一个短签名，供产物文件名用。

    为什么需要：约束 #6 要求产物名带上"是什么跑出来的"，但开关一多名字就长到
    不可用（9 个 kernel 名摊开是 80+ 字符）。所以可变的那部分压成 n 位签名，
    **完整取值仍在产物内容里**（如 CSV 的 kernel 列），因此可反查。
    """
    import hashlib
    norm = "|".join(
        ",".join(sorted(x.strip() for x in str(b).split(",") if x.strip()))
        for b in bits)
    return hashlib.sha1(norm.encode()).hexdigest()[:n]


def result_new(*parts, force: bool = False) -> str:
    """同 result()，但**产物已存在就拒跑**（除非 force=True）。

    起因是两次真实事故，都属于"名字看着正常、内容被静默换掉"：
      * 2026-09-05：一次 0.5 ns 计时跑与 5 ns 生产轨迹同名，--force 把 984 MB
        覆盖成 61 MB，无备份不可恢复。
      * nve_drift 的产物名里**没有 dt**，而 dt 对照正是它唯一的判据 —— 跑完
        --dt 0.002 再跑 --dt 0.001，第二次会把第一次悄悄吃掉。

    ⚠ **在开跑之前调用它**，不要等算完再写：这两个脚本一跑就是几小时，
    在末尾才发现"产物已存在"等于白烧机时。
    """
    from datetime import datetime
    p = Path(result(*parts))
    if p.exists():
        st = p.stat()
        desc = (f"{st.st_size:,} B, "
                f"{datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}")
        if not force:
            raise FileExistsError(
                f"产物已存在，拒绝覆盖：{p}\n"
                f"  旧产物：{desc}\n"
                f"  要覆盖请显式加 --force；或把旧产出搬走；"
                f"或换一个 L_IPS_RESULTS_DIR。")
        print(f"  ⚠ --force：即将覆盖 {p.name}（{desc}）", flush=True)
    return str(p)


def data(*parts) -> str:
    """DATA_DIR 下的路径，字符串形式（openmm/mdtraj 的 API 收字符串）。"""
    return str(DATA_DIR.joinpath(*map(str, parts)))


def result(*parts) -> str:
    """RESULTS_DIR 下的路径；父目录不存在时自动建。"""
    p = RESULTS_DIR.joinpath(*map(str, parts))
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def system_dir(*parts) -> str:
    """预建体系目录下的路径，例如 system_dir('1AAY', 'amber')。"""
    return str(SYSTEMS_DIR.joinpath(*map(str, parts)))


def describe() -> str:
    return (f"PROJECT_ROOT={PROJECT_ROOT}\n"
            f"DATA_DIR    ={DATA_DIR}\n"
            f"RESULTS_DIR ={RESULTS_DIR}")


def main():
    print(describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
