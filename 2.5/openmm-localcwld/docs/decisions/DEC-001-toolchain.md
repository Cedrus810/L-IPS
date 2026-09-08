# DEC-001: 工具链与开发环境冻结

- 状态：**冻结**（`openmm_dev` 环境已直接验证；仅剩 SWIG/多节点同构性两项留待用户确认，见第 4 节）
- 日期：2026-08-28
- 工单：LCWLD-000
- 记录人：Claude (assistant)，证据来自当前交互式 sandbox，直接调用了真实的
  `openmm_dev` conda 环境（见第 2 节）——不是道听途说或猜测。

## 1. 环境定位说明

仓库根目录的 `test.zsh`（PBS 提交脚本）引用的路径是
`/home/ruigengji/mambaforge/...`，这个路径在当前 sandbox 上**不存在**。但同一台
机器上存在等价的 `/home/ruigengji/miniforge3/envs/openmm_dev/`，其解释器可以
直接被调用，且确认装有完整的 OpenMM + PDBFixer + numpy + pytest 工具链。下文
第 2 节的所有条目都是用这个环境**直接跑出来的**，不是猜测。

⚠️ 需要用户确认一句话：真正提交 PBS 作业时用的 `openmm_dev` 是不是就是这一个
（`miniforge3` 而不是 `mambaforge`）？如果机器上确实有两套 conda 发行版且用的
是不同的那一套，下面的版本号需要在那一套上重新采集一遍（用第 4 节的核对脚本，
只需替换 `source`那一行的路径）。

## 2. 已用真实 `openmm_dev` 环境直接验证的事实

| 项 | 值 | 来源 |
|---|---|---|
| 环境路径 | `/home/ruigengji/miniforge3/envs/openmm_dev` | 直接 `find`/调用 |
| Python | 3.12.13 | `python3.12 --version` |
| OpenMM | `8.5.2` (conda 包 `openmm-8.5.2-py312h5a97af1_0`；`openmm.version.version` 报告为 `8.5.2.dev-36a30cb`) | `openmm.version.version`, conda-meta |
| OpenMM 头文件/库 | `.../envs/openmm_dev/include/OpenMM.h`, `.../envs/openmm_dev/lib/libOpenMM.so` | `find` |
| OpenMM plugin 目录 | `.../envs/openmm_dev/lib/plugins/` | `find` |
| 已安装的 runtime plugin | `libOpenMMCUDA.so`, `libOpenMMOpenCL.so`, `libOpenMMCPU.so`, 以及 Amoeba/Drude/RPMD/PME/Torch 的 CUDA/OpenCL/HIP/Reference 变体（**含 HIP**，但 DEC-003 仍把 v0.1 范围锁定在 Reference+CUDA） | `ls lib/plugins/` |
| 可用 Platform（运行时） | `Reference`, `CPU`, `CUDA`, `OpenCL` 四个全部注册成功 | `Platform.getPlatform(i).getName()` |
| OpenMM 编译用 CUDA toolkit | **12.9**（`cuda-nvcc-12.9.86`, `cuda-cudart-12.9.79`, `cuda-version-12.9`，均为 conda 包，装在 env 内部：`envs/openmm_dev/bin/nvcc` 报告 `release 12.9, V12.9.86`） | conda-meta JSON 文件名 + `envs/openmm_dev/bin/nvcc --version` |
| OpenMM 编译用 C/C++ 编译器 | **conda-forge gcc/g++ 14.3.0**（`gcc_linux-64-14.3.0`, `gxx_linux-64-14.3.0`；env 内可执行名为 `x86_64-conda-linux-gnu-gcc`/`-g++`） | conda-meta + 直接调用 `--version` |
| numpy | 2.4.3 | `numpy.__version__` |
| pytest | 9.1.1 | `pytest.__version__` |
| pdbfixer / mdtraj / pandas / matplotlib / scipy | 均可正常 `import`（`test_lips_vs_pmeV2.6.py` 模块级依赖全部满足） | `import` 测试 |
| CMake | env 内**没有**自带 cmake，需要用系统 cmake 4.4.2（见下）或额外装 | `envs/openmm_dev/bin/cmake` 不存在 |
| OpenMM CMake package config | 未找到 `OpenMMConfig.cmake`／`OpenMM-config.cmake`（这个 conda 包没装 CMake config，`FindOpenMM.cmake` 需要手写，正合计划第 21/57 节的假设） | `find` 未命中 |
| SWIG | env 内**没有**（`envs/openmm_dev/bin/swig` 不存在），LCWLD-080 前必须单独装 | `ls`/`which` 未命中 |
| 真实 1CKK 体系构建 | 成功，`31358` 原子（Amber19SB + TIP3P + 1.5 nm 水盒 + 0.15 M NaCl），几十秒内完成，只是系统构建，未做任何积分/生产步 | 直接调用 `v26.build_1ckk_system()` |

**关键结论**：`nvcc`(12.9)/`gcc`(14.3.0) 才是这个 OpenMM 安装实际链接使用的
版本，跟 sandbox 系统自带的 `nvcc`(13.3)/`gcc`(16.2.1) 都不一样。**LCWLD-010/
110 配置 CMake 时必须显式指向 conda 环境内的 `x86_64-conda-linux-gnu-gcc`/
`-g++`（或至少一个 ABI 兼容的 GCC 14.x），不能默认用系统 GCC 16**，否则很可能
在链接期或运行期出现 libstdc++ ABI 不兼容。CUDA 编译同理，优先使用 env 内的
`nvcc`（12.9）而不是系统的 `nvcc`（13.3）。

系统层信息（sandbox 本身，仅供 CUDA driver 是否够新做参考，不代表
`openmm_dev` 的构建工具链）：

| 项 | 值 |
|---|---|
| OS | CachyOS Linux（Arch-like rolling release），kernel `7.2.0-1-cachyos` |
| 系统 CMake | 4.4.2 |
| 系统 gcc/g++ | GCC 16.2.1（**不要用来建 LocalCWLDForce 插件**，见上） |
| 系统 `nvcc` | CUDA 13.3（**不是**建 OpenMM 用的那个，编译插件请用 env 内 12.9） |
| NVIDIA driver | 580.178.04，report CUDA 13.0（新于 12.9 工具链，向后兼容，没问题） |
| GPU | 1x NVIDIA GeForce RTX 2080 Ti，11264 MiB，`00000000:AF:00.0` |

`LocalCWLDForce` 已经在这个真实环境里做了第一个交叉验证：`metadata.py` 对
**真实 1CKK 体系**（31358 原子）跑出的九个数组，与
`build_phase_cwld_metadata()` 逐位相同（`max_abs_diff == 0`），细节见
LCWLD-020 报告。

## 3. 由此确定的 LCWLD-010 输入

- `OPENMM_DIR` = `/home/ruigengji/miniforge3/envs/openmm_dev`
- OpenMM 头文件在 `$OPENMM_DIR/include`，库在 `$OPENMM_DIR/lib`
- Plugin 安装目标目录 = `$OPENMM_DIR/lib/plugins`
- C/C++ 编译器 = `$OPENMM_DIR/bin/x86_64-conda-linux-gnu-gcc` /
  `$OPENMM_DIR/bin/x86_64-conda-linux-gnu-g++`（conda-forge GCC 14.3.0）
- CUDA 编译器 = `$OPENMM_DIR/bin/nvcc`（12.9），不是系统 `nvcc`
- CMake 用系统的 4.4.2（env 内没有）
- SWIG 需要额外安装（`mamba install -n openmm_dev swig` 或同等操作），版本待定

## 4. 仍需要用户确认的点（UNKNOWN，未在此 sandbox 验证）

| 项 | 状态 |
|---|---|
| PBS 作业实际使用的 conda 发行版是不是就是这个 `miniforge3/envs/openmm_dev`（而不是脚本里写的 `mambaforge`路径） | **RESOLVED 2026-08-31**：用户确认换服务器后统一用 `miniforge3/envs/openmm_dev`；新机器无 PBS（见本文件末尾追加节） |
| SWIG 版本（还没装） | UNKNOWN，待装后确认 |
| `groupG` 队列的其它计算节点是否与本机同构（同一 GPU 型号、同一 driver 版本、同一份 `openmm_dev`） | **已作废 2026-08-31**：换服务器，新机器上没有 `groupG`，也没有任何调度器 |

## 5. 已确定、不需要用户重新验证的决策

- v0.1 目标计算环境固定为 Linux + NVIDIA CUDA GPU（已直接确认 CUDA 13 系
  driver + 1 张 RTX 2080 Ti + 可用的 CUDA/OpenCL/Reference/CPU platform）。
- 不使用 Windows/OpenCL/HIP 作为 v0.1 目标（见 DEC-003；OpenCL/HIP 插件虽然
  已经装在环境里，但 v0.1 不承诺测试/支持它们）。
- 源码位置：先放在本仓库子目录 `2.5/openmm-localcwld/`，不新开独立仓库。

---

## 追加：2026-08-31 换服务器后的环境重测（用户确认）

用户明确：**项目已迁到新服务器，工作区改为 `/home/ruigengji/L-IPS/2.5`，此后
计算环境统一 `/home/ruigengji/miniforge3/envs/openmm_dev`，平台统一 CUDA。**
以下为在新机器（`kasuga01`, Linux 7.2.0-1-cachyos）上直接实测的结果：

| 项目 | 新机器实测 | 与旧记录的差别 |
| --- | --- | --- |
| 解释器 | `/home/ruigengji/miniforge3/envs/openmm_dev/bin/python` | 确认（不再是 UNKNOWN） |
| OpenMM | 8.5.2.dev-36a30cb | 同 |
| 可用 platform | `Reference` / `CPU` / `CUDA` / `OpenCL` | 同 |
| GPU | 1x RTX 2080 Ti, 11264 MiB, driver 580.178.04 | 同 |
| env 内 `nvcc` | 12.9 (V12.9.86) | 同 |
| env 内 `gcc` | conda-forge gcc 14.3.0 | 同 |
| **系统 `nvcc`** | **不存在**（`/usr/bin/nvcc` 没有） | 旧记录写的系统 nvcc 13.3 已不存在，env 内 12.9 现在是唯一的 nvcc |
| 系统 `gcc` | 16.2.1 | 同 |
| `cmake` | `/usr/bin/cmake` 4.4.2 | LCWLD-010 可直接用 |
| **调度器** | **没有** —— `qsub` / `sbatch` / `pbsnodes` / `sinfo` / `squeue` 全部不存在 | 旧 PBS `groupG` 提交方式作废 |
| `module` 系统 | `/home/apps/Modules/init/profile.sh` 不存在（`~/modulefiles` 目录还在但没用） | 旧 `test.zsh` 的 module 段作废 |
| `mambaforge` | 目录存在但 **没有 `envs/`**，是空壳 | 旧 `test.zsh` 激活路径作废 |

对 LCWLD-010 的影响：

- 编译器/工具链选择不变（env 内 `nvcc` 12.9 + conda gcc 14.3.0），但**不再需要
  区分"系统 nvcc 还是 env nvcc"**，系统上已经没有第二个 nvcc 了。
- GPU 就在本机，作业直接跑，不再需要为 PBS 排队/walltime 设计任何东西。
- 旧的"其它计算节点是否同构"问题消失（只有这一台）。

---

## 追加：2026-09-03 CUDA 版本链核实（更正前述记录）

用户指出"这台机器是 CUDA 13.0，为什么还在用 12.9"。核实结果如下，**用户对
驱动版本的说法是对的，但真正决定插件构建的不是驱动**。以下每一条都是本次
直接实测，不是推断。

### 实测到的四层

| 层 | 版本 | 证据 |
|---|---|---|
| NVIDIA 驱动 | **CUDA 13.0** | `cuDriverGetVersion()` 返回 `13000`；`/usr/lib/libcuda.so.580.178.04` |
| 系统 toolkit | nvcc **13.3** | `/opt/cuda/bin/nvcc --version` |
| env 内 toolkit | nvcc **12.9** | `$OPENMM_DIR/bin/nvcc --version` |
| **env 内 OpenMM 8.5.2 本身** | **CUDA 12 构建** | `ldd libOpenMMCUDA.so` → `libnvrtc.so.12`；env 内 `libcudart.so.12.9.79`、`libnvJitLink.so.12.9.86`；`targets/x86_64-linux/include/cuda.h` 的 `CUDA_VERSION == 12090` |

### 决定性事实：这个插件不需要 nvcc

- 计划第 21 节的 `platforms/cuda/src/` **全是 `.cpp`，没有 `.cu`**。
- device kernel 是 `platforms/common/kernels/localCWLD.cc`，以**字符串**形式在
  运行时编译。
- OpenMM 走 **NVRTC** 而不是外部 nvcc：`CudaCompiler` 属性默认值为空字符串，
  且 `libOpenMMCUDA.so` 里有 `nvrtcCreateProgram` / `nvrtcCompileProgram` 符号，
  链接的是 `libnvrtc.so.12`（12.9.86）。

**结论：把 CMake 指向 CUDA 13.3 的 nvcc 不会让 device kernel 变成 CUDA 13。**
编译 device 代码的永远是 OpenMM 链接的那个 NVRTC 12.9。因此
`LOCALCWLD_BUILD_CUDA=ON` 已改为**完全不调用 `enable_language(CUDA)`**，只需要
CUDA driver API（`cuda.h` + `libcuda`），新增 `cmake/FindCUDADriverAPI.cmake`。
12.9 的 driver API 头 + 13.0 驱动是正常的向后兼容组合。

### 已验证可编译可运行

用 env 的 g++ 14.3 + env 的 `targets/x86_64-linux/include`（CUDA 12.9 头）编译
一个包含 `openmm/cuda/CudaContext.h` 的程序，链接 `libOpenMM.so` 和
`/usr/lib/libcuda.so`，运行输出：

```text
cuInit=0 devices=1 driverVersion=13000 (CUDA 13.0)
CUDA_VERSION header macro = 12090
```

即 LCWLD-110 会用到的头文件组合已经提前跑通。

注意：`libcuda` 要优先取 `/usr/lib/libcuda.so`（真驱动），不要取
`/opt/cuda/targets/x86_64-linux/lib/stubs/libcuda.so`——那是 13.3 toolkit 的
stub，和 12.9 的头凑一起等于把第三个 CUDA 版本拉进来，没有任何好处。
`FindCUDADriverAPI.cmake` 已按此排序。

### 如果要整条栈上 CUDA 13

**唯一有意义的做法是换一个 CUDA-13 构建的 OpenMM**（新建 env 或升级现有 env），
不是换 nvcc。驱动 13.0 支持这么做。但这会换掉整个生产环境，`2.5/` 下所有脚本
和已有轨迹的复现条件都要重新确认，**属于用户决定，本文件不替用户拍板**。
在换掉之前，插件必须按 CUDA 12 / NVRTC 12.9 这条链来构建。

### 更正前述记录

2026-08-31 追加节写的"系统 `nvcc` 不存在，env 内 12.9 是唯一的 nvcc"**已不成立**：
`/opt/cuda/bin/nvcc` 是 13.3。因此 `-DCMAKE_CUDA_COMPILER` 之类的设置不能靠
PATH 撞对——不过按上面的决定，本插件根本不需要它。

## 追加：2026-09-03 两个 conda 环境下的工具链陷阱（实测）

写 LCWLD-110 时踩到的，换机器还会遇到：

1. **`Platform::getDefaultPluginsDirectory()` 在 conda 环境下返回错误路径。**
   它返回的是编译进 `libOpenMM` 的硬路径 `/usr/local/openmm/lib/plugins`，
   这个目录在 conda 环境里不存在，所以 C++ 侧调它会加载 **0 个**插件，
   后果是 `There is no registered Platform called "CUDA"`。
   （Python 包能正常工作是因为它自己定位插件目录。）
   C++ 测试/工具必须自己给路径，本项目用环境变量 `OPENMM_PLUGIN_DIR`，
   由 CMake 在 ctest 里设成 `${OPENMM_ROOT_DIR}/lib/plugins`。

2. **common-compute 层不在 `libOpenMM.so` 里。**
   `ComputeContext` / `ComputeForceInfo` / `ComputeArray` 这些符号定义在
   `lib/plugins/libOpenMMCUDA.so`。只链 `libOpenMM` 能编译通过，
   **dlopen 时才失败**（`undefined symbol: _ZTIN6OpenMM16ComputeForceInfoE`）。
   插件目标必须链接 `libOpenMMCUDA`。

   ⚠ 但**测试可执行文件不要链它**：OpenMM 自己会 dlopen 同一个库，
   直接链接会导致初始化两次，进程退出时 `double free or corruption`——
   而且是在所有断言通过之后才崩，看着像析构玄学，实际是链接行写错。
