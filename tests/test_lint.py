"""静态检查闸门：把"代码整洁"变成会失败的测试，而不是一份没人看的报告。

为什么值得单独一个测试：这一轮打包重构里，pyflakes 抓到的 46 条问题里有一条是
**真 bug** —— `analysis/szz.py` 里 `NonbondedForce` 未定义（迁移时删了 import 却留着
用法），`lips-szz` 一跑到那行就 NameError。它躺在那儿没人发现，因为没有任何东西在看。

分两级：
* `undefined name` —— **永远失败，没有例外**。这类必然是运行时崩溃。
* 未使用的 import —— 失败，但允许一份显式豁免名单（见下），因为有三个文件目前
  处在"刻意不改动"的约束下。豁免是**按文件+问题**精确列出的，不是整个文件放行；
  这些文件里出现新的问题照样会失败。

刻意不检查 `f-string is missing placeholders`：纯风格，改它要动几十处日志行，
换不来任何正确性。

**2026-09-07 补的第三级（由 `2-5-3a` 查出的漏洞）**：原来的两个测试都是
「按消息子串筛选」，于是任何**不属于这两类**的 pyflakes 消息都被静默丢弃 ——
包括语法错误。`src/lips/analysis/dens_parity.py` 曾有一处 IndentationError
（模块根本 import 不了，那还是 §9.1 的 P0 脚本），三个测试**全绿**：
语法错误行 `path:101:1: unexpected indent` 能正常切成 4 段，却匹配不上任何一个筛选器，
而 pyflakes 遇语法错误的退出码是 1、正好在允许的 `(0, 1)` 里 ⇒
**一个语法错误的文件与一个干净文件不可区分**。它还顺带掩盖了同一文件里一个
本闸门专门要抓的 `undefined name 'paths'`（缩进修好后 pyflakes 立刻报了两次）。

所以现在改成**白名单式**：消息类别必须落在 `ENFORCED_KINDS`（各有自己的测试）
或 `IGNORED_KINDS`（显式豁免、写明理由）里，**其余一律失败**。
这样语法错误、以及将来 pyflakes 新增的任何消息类别，都会撞上 `test_no_unrecognised_kinds`
而不是被无声吃掉。另外 `test_all_modules_parse` 完全不依赖 pyflakes ——
pyflakes 缺席时整个模块会 skip，那时它是唯一还在看「文件能不能 parse」的东西。
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "lips"

# 精确豁免：(相对路径, 问题描述片段)。
#
# 这三个文件此刻不能动，原因各不相同，都记在这里而不是靠口头约定：
#   zn_coordination.py / zn_job.py —— 1AAY 科学线未收口，跑分析的会话随时可能
#       补跑扫描，改动会让"这批数据跑在什么代码上"更难说清。
#   engine/v26.py —— 2791 行单体，同一套 CWLD 数学在仓库里有多份独立实现，
#       动它有让它们静默失同步的风险；在拆分方案定下来之前只做零改动。
#
# 解除条件写在每一行后面。清掉之后请把对应条目删掉，不要留着当permanent 豁免。
KNOWN = {
    ("engine/v26.py", "'glob' imported but unused"),                    # v26 解冻后清
    ("analysis/zn_coordination.py", "'sys' imported but unused"),       # 科学线收口后清
    ("analysis/zn_coordination.py", "DATA_DIR as _DATA_DIR' imported but unused"),
    ("run/zn_job.py", "DATA_DIR as _DATA_DIR' imported but unused"),
    ("run/zn_job.py", "'openmm as mm' imported but unused"),
    ("run/zn_job.py", "'openmm.unit' imported but unused"),
}

# 各自有专门测试在盯的类别。不在这里、也不在 IGNORED_KINDS 里的消息 ⇒ 直接失败。
ENFORCED_KINDS = (
    "undefined name",           # test_no_undefined_names
    "imported but unused",      # test_no_unused_imports_outside_allowlist
)

# 显式豁免的类别，每条都要有理由。豁免的是**类别**，不是某个文件。
IGNORED_KINDS = (
    # 纯风格；改它要动几十处日志行，换不来正确性。
    "f-string is missing placeholders",
    # 9 处，其中 4 处在 engine/v26.py —— 那个文件在拆分方案定下来前是零改动约束
    # （见 KNOWN 的说明），所以现在没法强制这一类。等 v26 解冻后再决定要不要提级。
    "is assigned to but never used",
)


def _pyflakes():
    """返回 [(相对路径, 行号, 消息)]。pyflakes 缺席就跳过整个模块。"""
    try:
        proc = subprocess.run([sys.executable, "-m", "pyflakes", str(PACKAGE)],
                              capture_output=True, text=True, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"pyflakes 不可用: {exc}")
    if proc.returncode not in (0, 1):
        # 注意：语法错误是 returncode 1 + stderr 有内容，**不算**工具挂了，
        # 不能因为 stderr 非空就 skip，否则又把语法错误变成"跳过"。
        pytest.skip(f"pyflakes 运行失败: {proc.stderr.strip()[:200]}")

    # **stdout 与 stderr 都要读**：pyflakes 把语法错误写到 **stderr**（2026-09-07 实测），
    # 只读 stdout 的话语法错误连消息筛选器都进不去 —— 这才是
    # `dens_parity.py` 那个 IndentationError 能全绿通过的真正机制。
    # （「按类别筛选、语法错误匹配不上」只是第二层；不补这一行，
    #   test_no_unrecognised_kinds 一样看不见它。）
    out = []
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        parts = line.split(":", 3)
        if len(parts) < 4:
            continue
        path, lineno, _col, message = parts[0], parts[1], parts[2], parts[3].strip()
        # 语法错误时 pyflakes 还会回显源码行与 `^` 标记行。那两行通常切不出 4 段，
        # 但源码里只要有 3 个冒号（比如一个 dict 字面量）就会切出来 ⇒ 必须靠
        # 「path 确实是 PACKAGE 下的文件、lineno 确实是数字」把它们挡掉，
        # 否则 relative_to 会抛 ValueError，测试变成 error 而不是 fail。
        if not lineno.isdigit():
            continue
        try:
            rel = str(Path(path).resolve().relative_to(PACKAGE))
        except ValueError:
            continue
        if any(k in message for k in IGNORED_KINDS):
            continue
        out.append((rel, lineno, message))
    return out


def _exempt(rel, message):
    return any(rel == f and frag in message for f, frag in KNOWN)


def test_no_undefined_names():
    """未定义的名字一律不放行——它就是还没触发的 NameError。"""
    bad = [(r, n, m) for r, n, m in _pyflakes() if "undefined name" in m]
    assert not bad, "存在未定义的名字（运行到就崩）:\n" + "\n".join(
        f"  {r}:{n}  {m}" for r, n, m in bad)


def test_no_unused_imports_outside_allowlist():
    """未使用的 import：豁免名单之外一律失败。"""
    bad = [(r, n, m) for r, n, m in _pyflakes()
           if "imported but unused" in m and not _exempt(r, m)]
    assert not bad, (
        "新增了未使用的 import：\n"
        + "\n".join(f"  {r}:{n}  {m}" for r, n, m in bad)
        + "\n删掉它们；确实必须保留的话，在 tests/test_lint.py 的 KNOWN 里"
          "写明文件、问题和解除条件。")


def test_allowlist_has_no_stale_entries():
    """豁免名单不许留过期条目——否则它会慢慢变成一张永久免死金牌。"""
    findings = _pyflakes()
    stale = [entry for entry in KNOWN
             if not any(entry[0] == r and entry[1] in m for r, _n, m in findings)]
    assert not stale, (
        "这些豁免已经不需要了，请从 KNOWN 里删掉：\n"
        + "\n".join(f"  {f}  ({frag})" for f, frag in stale))


def test_all_modules_parse():
    """每个 .py 都必须能 parse —— import 不了的模块等于没写。

    **不依赖 pyflakes**，所以 pyflakes 缺席、`_pyflakes()` 整个 skip 掉的时候，
    这一条仍然在看。`dens_parity.py` 那次 IndentationError 就是被这一层漏掉的：
    当时没有任何东西直接检查"能不能 parse"。
    """
    bad = []
    for f in sorted(PACKAGE.rglob("*.py")):
        try:
            ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError as exc:
            bad.append(f"  {f.relative_to(PACKAGE)}:{exc.lineno}  {exc.msg}")
    assert not bad, "这些文件根本 parse 不了（import 就炸）:\n" + "\n".join(bad)


def test_no_unrecognised_kinds():
    """未归类的 pyflakes 消息一律失败 —— 「没见过」不等于「可以忽略」。

    这是补上「语法错误与干净文件不可区分」那个漏洞的通用做法：不去枚举语法错误的
    各种措辞，而是反过来要求每条消息都落在 ENFORCED_KINDS 或 IGNORED_KINDS 里。
    将来 pyflakes 新增消息类别时，也会在这里显形而不是被静默丢弃。
    """
    bad = [(r, n, m) for r, n, m in _pyflakes()
           if not any(k in m for k in ENFORCED_KINDS)]
    assert not bad, (
        "pyflakes 报了未归类的问题（语法错误就长这样）:\n"
        + "\n".join(f"  {r}:{n}  {m}" for r, n, m in bad)
        + "\n修掉它；确实要放行的话，把**类别**加进 tests/test_lint.py 的 "
          "IGNORED_KINDS 并写明理由。")
