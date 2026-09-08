# LCWLD-010 报告：仓库与 CMake 最小骨架

- 日期：2026-09-03
- 前置：LCWLD-000（DEC-001 工具链冻结、DEC-003 平台范围冻结）
- 状态：**完成，验收命令已实跑通过**

## 做了什么

按计划第 21 节建出目录树，并写了可 configure / build / install 的 CMake 骨架。
**没有实现任何物理计算**——这是工单的硬约束。

新增文件：

```text
CMakeLists.txt                                       # 根构建脚本
cmake/FindOpenMM.cmake                               # 手写 finder
cmake/EncodeKernelFiles.cmake                        # .cc → C++ 字符串表
README.md
LICENSE
openmmapi/CMakeLists.txt
openmmapi/include/openmm/LocalCWLDVersion.h          # 占位内容
openmmapi/include/openmm/internal/windowsExportLocalCWLD.h
openmmapi/src/LocalCWLDVersion.cpp
platforms/reference/CMakeLists.txt
platforms/common/CMakeLists.txt
platforms/cuda/CMakeLists.txt
serialization/CMakeLists.txt
python/CMakeLists.txt
```

`.gitignore` 追加了 SWIG/encoded kernel 生成物、`stage/`、`*.dcd`、`logs/`。
空目录（`tests/unit`、`tests/integration`、`tests/benchmarks`、`schemas`、
`examples`、各 `src`/`tests`）已建出但没有占位文件，git 不会跟踪它们。

## 验收命令（实跑，输出见下）

```bash
export OPENMM_PREFIX=/home/ruigengji/miniforge3/envs/openmm_dev
cmake -S . -B build \
  -DOPENMM_DIR="$OPENMM_PREFIX" \
  -DCMAKE_CXX_COMPILER="$OPENMM_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
  -DLOCALCWLD_BUILD_REFERENCE=OFF -DLOCALCWLD_BUILD_CUDA=OFF -DLOCALCWLD_BUILD_PYTHON=OFF
cmake --build build --parallel
cmake --install build --prefix stage
```

结果：

- configure 成功，`Found OpenMM: .../openmm_dev/include`，编译器识别为 GNU 14.3.0。
- build 成功，产出 `libOpenMMLocalCWLD.so`。
- install 成功，`stage/` 内容为：

```text
stage/include/openmm/LocalCWLDVersion.h
stage/include/openmm/internal/windowsExportLocalCWLD.h
stage/lib/libOpenMMLocalCWLD.so -> .so.0 -> .so.0.1.0
```

源码树内没有生成物；`build/`、`build-cuda/`、`stage/` 全部被 `.gitignore` 排除。
**CUDA 关闭时不要求 CUDA toolkit**（`enable_language(CUDA)` 只在
`LOCALCWLD_BUILD_CUDA=ON` 时执行）——已确认关闭态 configure 不触碰 nvcc。

### 额外验证（工单没要求，但值得提前做）

1. **全开关 configure**：`-DLOCALCWLD_BUILD_REFERENCE=ON -DLOCALCWLD_BUILD_CUDA=ON
   -DLOCALCWLD_BUILD_PYTHON=ON -DBUILD_TESTING=ON`，配合
   `-DCMAKE_CUDA_COMPILER=$OPENMM_PREFIX/bin/nvcc`
   `-DCMAKE_CUDA_HOST_COMPILER=$OPENMM_PREFIX/bin/x86_64-conda-linux-gnu-g++`，
   **configure 成功**。这提前排掉了计划风险 E 里最贵的一条：env 内的
   nvcc 12.9 能接受 conda gcc 14.3 作为 host compiler。尚未实现的子目录各自
   打印一行 `no sources yet (LCWLD-0X0 pending) -- target skipped` 后 return，
   不会因为找不到源文件而 configure 失败。

2. **链接冒烟测试**：用 staged 头文件和库编译了一个独立小程序，同时链接
   `libOpenMMLocalCWLD.so` 和真实 `libOpenMM.so`，运行输出：

   ```text
   localcwld 0.1.0 physics=v0.1 openmm=8.5.2
   ```

   即 ABI 与真实 OpenMM 8.5.2 兼容，install RPATH 生效。

## 几个需要主 agent / 用户知道的决定

1. **占位 target 的源文件**：工单要求建 `OpenMMLocalCWLD` 占位 target，但
   target 需要至少一个源文件，而 §21 列出的 `LocalCWLDForce.cpp` 归 LCWLD-040
   所有。为不侵占 040 的文件，本工单另建了 `LocalCWLDVersion.{h,cpp}`（只返回
   版本字符串和"物理语义版本 v0.1"，无任何物理）。`openmmapi/CMakeLists.txt`
   用 `GLOB_RECURSE ... CONFIGURE_DEPENDS` 收集 `src/*.cpp`，所以 LCWLD-040/050
   加文件时**不需要改动本工单的任何文件**。

2. **export header 的归属重叠**：§24 把 export header 列在 LCWLD-040 的可改
   文件里，但 LCWLD-010 的步骤 6 又要求"设置 Windows export 宏"。本工单按步骤 6
   建了 `windowsExportLocalCWLD.h`。040 若需调整可直接改，这里只是先让骨架能编。

3. **LICENSE 选了 MIT，署名 Ruigeng Ji**。这是**待用户确认**项——许可证是用户的
   决定，不是执行方的。要换（Apache-2.0 / LGPL / 不公开）直接替换该文件即可，
   没有其它地方引用它。

4. **DEC-001 有一处已过时**：其 2026-08-31 追加节写"系统 `nvcc` 不存在"，但本次
   实测系统上有 `/usr/bin/nvcc`（CUDA 13.3），与 env 内的 12.9 并存。结论不变
   （仍用 env 内 12.9），但"只有一个 nvcc"这句不再成立，`-DCMAKE_CUDA_COMPILER`
   必须显式指定，不能靠 PATH 撞对。

5. **不硬编码 home 路径**：根 `CMakeLists.txt` 里没有任何 `/home/ruigengji`。
   编译器与 `OPENMM_DIR` 全部从命令行传入。作为补偿，configure 时会检查
   `OPENMM_DIR/bin` 里是否有 conda 自带的 g++ 而当前用的不是它，是则打 WARNING
   ——DEC-001 记录的 libstdc++ ABI 坑不该等到链接期才暴露。

## 下一步

依赖图上现在解锁的是 **LCWLD-040（C++ Public API）**，前置 LCWLD-010 + LCWLD-030
均已完成。040 只写 `LocalCWLDForce.h/.cpp` 和 API 测试，不碰 kernel。

注意：LCWLD-120 的阈值表要按 **DEC-004（精度 = float32）** 重写，不能沿用
计划第 9/25 节按 double 写的那份。
