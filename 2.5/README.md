# L-IPS v2.6 — CWLD 局部环境电荷响应

用 CWLD（局部密度驱动的隐式电荷响应）对比 PME 固定电荷与 AMOEBA 极化力场。
当前研究体系是 **1AAY Cys₂His₂ 锌指**；1CKK（钙调蛋白）是上一阶段的体系，
只保留复算能力。

主结果见 `docs/reports/REPORT_1AAY_ZN.md`。

## 快速开始

```bash
mamba activate openmm_dev            # OpenMM 8.5.2.dev + CUDA，细节见「环境」
pip install -e .                     # 已装则跳过
lips-paths                           # 确认路径解析
pytest tests/ -q                     # 21 passed
```

CWLD 生产路径现在走原生插件。没构建过先：

```bash
cd openmm-localcwld && cmake --build build-cuda -j && (cd build-cuda && ctest)
```

## 目录

```
src/lips/                包本体（唯一的代码入口）
  paths.py               PROJECT_ROOT / DATA_DIR / RESULTS_DIR / INBOX_DIR 的单一来源
  systems.py             ★ 体系加载的单一入口（见「必须守住的约束」）
  engine/
    v26.py               CWLD 引擎：closure 数学 + 元数据 + 力场装配 + MD 驱动
    closure.py           closure 窗函数族（zmm / pswf）
  build/                 体系构建：zinc_finger(1AAY) / zn_water
  analysis/              szz / zn_coordination / force_audit / nve_drift /
                         deltaq_probe / block_time / kill_switch / dcd_integrity
  run/zn_job.py          作业驱动（唯一会真正跑 MD 的入口）
tests/                   回归 + 静态检查闸门（含 lint 门：未定义的名字直接判失败）
openmm-localcwld/        LocalCWLDForce 原生插件（独立 CMake 工程，ctest 7/7）
docs/{reports,plans}/    报告与计划
scripts/                 zsh 运行脚本
1AAY/                    预建体系（输入，不是产物）

data/                    轨迹与产物，**平的一层**
  inbox/                 ← 新上传的东西放这里
logs/                    作业日志（含历史 training.o*）
archive/                 旧版本与手工备份
results_cwld_corrupt/    已隔离的坏数据，不要进任何分析
salvage/                 同上
```

`data/` 保持平的一层是故意的：`szz` 这类「读自己写的 csv、同时扫 dcd」的工具需要
单一根，分子目录得给它两个根，收益不抵成本。`inbox/` 是唯一例外 —— 分析的 glob
不会扫它，丢进去的文件不会被误当成一批轨迹。

## 命令

每个分析入口都是一个命令，不再是"跑某个散落的脚本"：

| 命令 | 作用 |
|---|---|
| `lips-paths` | 打印路径解析结果 |
| `lips-run-zn` | 跑 Zn 判据作业（**会跑 MD**） |
| `lips-szz` | 电荷结构因子 S_ZZ(k)，阶段 A 默认走 GPU/float32 |
| `lips-zn-coord` | Zn 配位数/几何的逐帧分析 |
| `lips-force-audit` | 静态力审计：只换 pair kernel，不跑 MD |
| `lips-nve-drift` | NVE 能量漂移 |
| `lips-deltaq` | Δq 定点探针 |
| `lips-block-time` | 分块收敛检查 |
| `lips-dcd-check` | 轨迹完整性校验 |
| `lips-build-1aay` / `lips-build-znwat` | 建体系 |

体系是参数：`--system 1aay`（默认）/ `1aay-amoeba` / `1ckk`，或
`--topology X.pdb --system-xml Y.xml` 指任意预建体系。

数据目录用 `L_IPS_DATA_DIR` 覆盖（复算 1CKK 旧数据时指到 `../2.4`）。

## CWLD 引擎：两个实现，别混

`setup_cwld_lips_system(..., engine=...)`：

| `engine` | 实现 | 用在哪 |
|---|---|---|
| `"plugin"`（默认） | `LocalCWLDForce` 原生插件 | **新跑的一切** |
| `"customgb"` | `mm.CustomGBForce` | 只用于补跑 2026-09-05 之前的既有矩阵 |

插件快 **3.1–3.6x**（1AAY 32794 原子、2 fs：54.78 → 184.33 ns/day；CWLD 力从
4.83 趟普通 rc 遍历降到 1.51 趟）。一个 29×5 ns 的矩阵从 48.4 GPU-h 降到 14.4（⚠ 48.4 是硬编码的 2080 Ti 基线，换硬件只有加速比可比、小时数不可比）。
相对 PME（3090 同机同 harness 实测）：CWLD 从 PME 的 **8.2%** 提到 **42.6%**
（PME 497.2 / 插件 212.0 / 旧引擎 41 ns/day）。
来龙去脉见 `openmm-localcwld/docs/reports/LCWLD-161-kernel-level-attribution.md`。

⚠ **两者数值等价但不是逐比特相同**（逐原子力 p99 3.3e-3，DEC-005 阈值 1.0），
所以**同一个统计量里不能混** —— 2.4 的 5 ns 矩阵就是被引擎不一致混掉过一次。
产物里因此带 `cwld_engine` 字段（`lips.engine.v26.cwld_engine_of()`）：光看一个 dcd
分辨不出它是哪个引擎跑的。`force_cost.py` 和 `block_time.py` 已显式钉死 `customgb`。

插件从 repo 内的 `openmm-localcwld/build-cuda/` 加载，**不装进共享的 conda env** ——
那个 env 有别的项目在用，往它的 plugin 目录里放东西会让机器上所有 OpenMM 进程
都加载这个力。

⚠ **一棵构建树只能属于一台机器**：`CMakeCache.txt` 里存的是编译器和工具链的绝对
路径。这个仓库在 NFS 上被多台机器挂着，共用一个 build 目录会让后一台的 cmake 拿着
前一台的配置去建、失败、并删掉前一台已建好的产物（2026-09-05 实际发生过）。
别的机器用 `LOCALCWLD_BUILD_DIR=build-cuda-<机器名>` 指到自己那棵树。

## 必须守住的约束

这几条都是踩过坑换来的，不是风格偏好。

**1. 体系加载只能走 `lips.systems`。**
读体系、按 `meta.json` 还原 HID/HIE、规范化离子残基名、取 qbase——这一串只有一份
实现。打包之前它在三个脚本里各写一份、各自演化，结果 `normalize_ion_resnames()`
只在其中一份里被调到；漏掉它会让金属的 `dens_source` 静默变 0，**CWLD 机制整个关掉
且不报错**，实测废掉过一整批轨迹。`tests/test_single_entry_point.py` 会在有人写第二份
时失败。

**2. `1CKK` 相关的默认值是复现锚点，不要"清理"。**
`L_IPS_CLOSURE=zmm`、`L_IPS_LIGAND_DPOLAR=None`、`CHARGED_SIDECHAIN_POLAR_ATOMS`
里的 `"HIS"` 原条目——它们存在的唯一理由是保 1CKK 逐位复现。

**3. 路径只能来自 `lips.paths`，挪数据之前先确认没人在跑。**
`DATA_DIR` / `RESULTS_DIR` 默认是 `<repo>/data`，可用 `L_IPS_DATA_DIR` /
`L_IPS_RESULTS_DIR` 覆盖。**不要在代码里写死路径。**

这条原本写的是"数据和产物不由代码移动"，起因是 assistant 把 85 个 csv/json 挪进
`results/`，正撞上另一个会话在跑分析。2026-09-05 又挪了一次（118 个文件、17 GB 从
仓库根进 `data/`，清单 `data/MOVED_FROM_ROOT_2026-09-05.txt`，回滚照着挪回去）。
**这次先确认了没有进程在写、两小时内无改动、所有读取都走 `DATA_DIR`。**
约束的本意是"不要在别人算的时候动数据"，不是"永远不能整理"——**先查，再挪，留清单**。

**4. 分类按元素判，不按残基名判。**
`force_audit.species_of()` 原本写死 `el == "Ca"`（1CKK 钙调蛋白时代），换到 1AAY 之后
**锌被静默归进 `protein_heavy`**，金属那一列和整个分壳 breakdown 消失且不报错。
新增元素改 `DIVALENT_BY_ELEMENT` / `lips.systems.DIVALENT_METAL_SPECIES`。

**5. 坏轨迹有两种坏法，mdtraj 对两种都只警告不报错。**
帧交错（并发写同一路径）能被 `dcd_integrity.check_dcd()` 的自洽性检查抓到；
截断（`mv` 活文件）**抓不到**，因为文件头尾自洽——那需要外部给 `expected_frames`。
`flag_frame_count_outliers()` 是同批比较的启发式，只报告不拦截。
已隔离的坏数据在 `results_cwld_corrupt/` 和 `salvage/`，不要让它们进任何分析。

**6. 产物文件名必须带上"是什么跑出来的"。**
2026-09-05：插件成为 CWLD 默认实现之后，一次 0.5 ns 的计时验证跑和 2026-09-03 那批
CustomGBForce 的 5 ns 轨迹**同名**，`--force` 一加就把 984 MB 覆盖成 61 MB，
无备份不可恢复（详见 `results_cwld_corrupt/OVERWRITTEN_2026-09-05_README.md`）。
更危险的是名字仍然正常：一条 0.5 ns 混在两条 5 ns 里做批量分析不会报错。
现在 `tag` 里带引擎（插件加 `_plugin`，旧引擎保持原名）。
**加新维度（引擎、精度、closure…）时先问"两次不同的跑会不会同名"。**

## 环境

```
/home/ruigengji/miniforge3/envs/openmm_dev
  Python 3.12   OpenMM 8.5.2.dev-36a30cb（CUDA 12 构建，NVRTC 12.9）
                git_revision 36a30cbca54e727b216b606f3c011b67201eb8b4
  numpy 2.4  scipy 1.17  pandas  mdtraj 1.11  torch 2.12(+cu12.9, 可选)
  GPU: RTX 2080 Ti,  驱动 580.178.04（CUDA 13.0 能力）
```

⚠ **OpenMM 版本是钉死的，不要"升级修正"它。** mamba 渠道上那个 8.6.0 是 **alpha
预发布**，没有 GPU 支持，生产上用不了。别处文档写 8.6.0 的一律不要照着改环境；
版本以解释器为准（`openmm.version.full_version`），不是文档。
等正式版带 GPU 支持发布再重新评估。

版本还有第二层意义：`LocalCWLDForce` 的 kernel 读的是 OpenMM **内部**的邻居表布局
（`exclusionTiles` / `interactingAtoms` / `interactionCount`，不在安装的头文件里），
是照着 `36a30cb` 的源码转写的。换版本**不会编译报错也不会加载失败，只会静默算错力**。
`v26.ensure_localcwld_plugin()` 里有一道 `RuntimeWarning` 比对 `git_revision`。

平台固定 CUDA，精度 **float32**（`openmm-localcwld/docs/decisions/DEC-004-precision.md`）：
生产轨迹本来就跑单精度，float64 只保留在 CPU 审计 oracle 里。

⚠ 这台机器的 `/home/ruigengji` 是 NFSv3，该导出**拒绝写 mode 0444 的文件**，
而 git 的松散对象正是 0444 建的 —— 所以**在这棵树里任何 git 写操作都会失败**。
另外仓库目录属主不是当前用户，git 还会报 dubious ownership。两个坑叠加的结果是
git 在这里基本不可用。

## 已知未完成

- `engine/v26.py` 是近 2900 行单体，同时是 closure 数学 / 元数据 / 力场装配 / MD
  驱动 / 分析 / 实验矩阵调度。拆分**暂缓**：同一套数学在仓库里有多份独立实现
  （见 `docs/reports/MATH_CHANGE_MAP.md`），动它有让它们静默失同步的风险。
- 分析层的 `_local_dens_at` 是 CustomGBForce 密度表达式的手抄镜像，**尚未做过
  一致性验证**——它是几份实现里唯一没验过的那份，而报告 §4.1 的 Δq 数字出自它。
- 轨迹缺 provenance sidecar。`cwld_engine` 字段补上了引擎这一维，但"某批数据跑在
  哪个 commit 的代码上"仍然只能靠翻 `logs/`。
- 本仓库当前**不在有效的版本控制下**（`/home/ruigengji/L-IPS/.git` 最后一次提交
  是 2026-05-26，只跟踪 11 个 demo 阶段文件），而且如上所述 NFS 让 git 写不进去。
  `archive/` 里那些 `.bak-*` 正是这件事的代偿。
- `archive/` `salvage/` `results_cwld_corrupt/` `results_pswfz2_partial/` 四个目录
  没有清理过，也没查过还有没有代码在引用它们。
