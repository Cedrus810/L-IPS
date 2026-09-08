# DEC-003: v0.1 平台支持范围

- 状态：冻结
- 日期：2026-08-28
- 工单：LCWLD-000

## 结论

- v0.1 承诺支持并在 release-gate 中验证：**Linux + OpenMM Reference platform + OpenMM CUDA platform**。
- v0.1 **不承诺**支持：Windows、OpenCL、HIP/ROCm、macOS。
- Common Compute host 层（`platforms/common`）代码结构应保持 CUDA/OpenCL/HIP 可共用，为将来扩展留口子，但 v0.1 只实际编译/测试 CUDA 后端；OpenCL/HIP 的 CMake target 可以先声明为 `OFF` 且不要求过编译。
- PBC：v0.1 必须支持 OpenMM 合法的一般周期盒（含 triclinic），不允许"只支持 orthorhombic 但自称通用 PBC"。若 Reference 早期里程碑临时只做 orthorhombic，必须在此文档显式追加限制并在 release 前解除（见计划第 47 节）。

## 依据

- DEC-001 确认当前可访问的 sandbox 上有 NVIDIA driver 580.178.04 / CUDA 13 系工具链 / 1x RTX 2080 Ti；真实作业通过 PBS `groupG` 队列提交，`nodes=groupG:ppn=32:gpus=1`，队列名暗示这是一组 GPU 节点，但具体节点是否同构（是否都用 NVIDIA GPU、驱动版本是否一致）仍是 DEC-001 里的 UNKNOWN 项，需要用户确认后如有出入回来更新本文件。
- 项目当前唯一的科学目标体系（1CKK，见计划第 8/24 节 LCWLD-140）在蛋白质/水/离子体系上运行，没有出现 lipid bilayer 等需要 triclinic 盒或跨平台可移植性的迫切需求，但计划本身（第 6/47 节）要求不因当前一个体系而阉割 PBC 通用性，因此这里不缩小 PBC 范围，只缩小"操作系统 + compute backend"范围。

## 不做的事（记录以避免后续 agent 重新讨论）

- 不为 v0.1 建 Windows CI lane。
- 不为 v0.1 建 OpenCL/HIP CI lane，即使 Common Compute 代码理论上可移植。
- 不因为当前 sandbox 只看到 1 张 GPU 就把"多 GPU/多 Context 并行"纳入 v0.1（计划第 53 节已经把它列为 backlog 第 10 项，明确排除在 v0.1 外）。

---

## 追加：2026-08-31 换服务器

- 上面「依据」里提到的 PBS `groupG` 队列、`nodes=groupG:ppn=32:gpus=1`、以及
  「其它节点是否同构」的疑问**全部作废**：项目已迁到新服务器，新机器上没有任何
  调度器，GPU（1x RTX 2080 Ti，driver 580.178.04）就挂在本机，作业直接跑。
- **结论本身不变**：v0.1 仍然只承诺 Linux + Reference + CUDA。用户 2026-08-31
  明确「之后全是 CUDA」，因此 v2.6 主脚本的 platform 选择已改成硬要求 CUDA
  （详见 `2.5/test_lips_vs_pmeV2.6.py::select_platform`），OpenCL 只作为显式
  opt-in 的调试出口，不进 release-gate。
- 详细环境实测见 `DEC-001-toolchain.md` 末尾的追加节。
