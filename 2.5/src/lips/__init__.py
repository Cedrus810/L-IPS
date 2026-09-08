"""L-IPS：CWLD 局部环境电荷响应 vs PME / AMOEBA。

分层（打包前这些全是仓库根下平铺的脚本，同一份逻辑被抄了三遍，改一处漏两处）：

    lips.paths      项目路径的单一来源
    lips.systems    **体系加载的单一入口** —— 读体系 / 还原 HID-HIE /
                    规范化离子残基名 / 取 qbase。所有脚本必须走这里。
    lips.engine     CWLD 数学与 OpenMM 力场装配（closure, v26）
    lips.build      体系构建（zinc_finger=1AAY, zn_water）
    lips.analysis   轨迹分析（szz, zn_coordination, force_audit, ...）
    lips.run        作业驱动（zn_job）

`lips.systems` 是这个包存在的主要理由。在它之前，
`force_audit` / `nve_drift` / `run_zn_job` 各写一份"怎么拿到体系"，于是
`normalize_ion_resnames()` 只在其中一份里被调用——漏掉它会让金属的
`dens_source` 静默变 0，CWLD 机制整个关掉且不报错，实测废掉过一整批轨迹。
这种步骤只能有一份实现。
"""

__version__ = "2.6.0"
