"""守住"体系加载只有一个入口"这条线。

这不是形式主义。打包之前 force_audit / nve_drift / run_zn_job 各写一份"怎么
拿到体系"，各自演化，结果 `normalize_ion_resnames()` 只在其中一份里被调用——
漏掉它会让金属的 dens_source 静默变 0，CWLD 机制整个关掉且**不报错**，实测
废掉过一整批轨迹。下面的测试让这种回归在 CI 里当场失败，而不是三周后在结果里。
"""
import inspect

import pytest

from lips import systems as ls

CONSUMERS = [
    "lips.analysis.szz",
    "lips.analysis.force_audit",
    "lips.analysis.nve_drift",
    "lips.analysis.deltaq_probe",
    "lips.analysis.block_time",
    "lips.run.zn_job",
]

# 只允许出现在 lips.systems / lips.engine.v26 / lips.build.* 里的加载动作
FORBIDDEN = ("XmlSerializer.deserialize", "hid_resids", "normalize_ion_resnames(")


@pytest.mark.parametrize("name", CONSUMERS)
def test_consumer_uses_shared_loader(name):
    import importlib
    src = inspect.getsource(importlib.import_module(name))
    assert "load_reference_system" in src, (
        f"{name} 没有走 lips.systems.load_reference_system —— "
        f"体系加载不允许有第二份实现")


@pytest.mark.parametrize("name", CONSUMERS)
def test_consumer_does_not_reimplement_loading(name):
    import importlib
    src = inspect.getsource(importlib.import_module(name))
    # 注释里提到这些名字是允许的（解释历史），实际调用不允许。
    code = "\n".join(l.split("#")[0] for l in src.split("\n"))
    hits = [f for f in FORBIDDEN if f in code]
    assert not hits, (
        f"{name} 自己实现了体系加载步骤 {hits}；这些只能在 lips.systems 里出现")


def test_loader_normalizes_ions_by_default():
    """默认必须规范化离子名，否则金属不会被认成 density source。"""
    sig = inspect.signature(ls.load_reference_system)
    assert sig.parameters["normalize_ions"].default is True


def test_divalent_metal_species_single_source():
    """二价金属清单只有一份，force_audit 不得另起炉灶。"""
    from lips.analysis import force_audit
    assert force_audit.DIVALENT_METAL_SPECIES is ls.DIVALENT_METAL_SPECIES


def test_default_system_is_not_1ckk():
    """1CKK 已不是当前研究体系；默认值不该再指向它。"""
    assert ls.DEFAULT_SYSTEM != "1ckk"
    assert "1ckk" in ls.SYSTEM_BUILDERS, "但要保留，用于复算旧数据"
