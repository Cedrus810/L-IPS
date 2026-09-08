#!/bin/zsh
# L-IPS v2.5 运行脚本（新服务器版，2026-08-31 迁移）
#
# 与 2.4 的差别（旧 PBS 版已留档为 test.zsh.bak-pbs-2.4）：
#   1. 这台机器上没有任何调度器（qsub / sbatch / pbsnodes / sinfo 都不存在），
#      GPU 就挂在本机上，所以不再走 PBS，直接前台/nohup 跑。
#   2. 没有 /home/apps/Modules/init/profile.sh，module 那一套整段删掉。
#   3. conda 发行版是 miniforge3，不是 mambaforge（mambaforge 目录还在，但没有 envs/）。
#   4. 平台固定 CUDA：跑之前先硬校验，不可用就直接退出，绝不静默掉到 CPU。
#
# 用法：
#   ./test.zsh                       # 跑下面选中的 RUN_MODE，日志同时写到 logs/
#   nohup ./test.zsh &> /dev/null &  # 后台长跑（日志照样进 logs/）

set -e
set -u

WORKDIR=/home/ruigengji/L-IPS/2.5
CONDA_ROOT=/home/ruigengji/miniforge3
ENV_NAME=openmm_dev

# conda/mamba 的 shell hook 不是 nounset-safe（会碰未定义的 precmd_functions），
# 所以只在激活这一段临时关掉 set -u。
set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
source "$CONDA_ROOT/etc/profile.d/mamba.sh"
mamba activate "$ENV_NAME"
set -u

cd "$WORKDIR"

# 只用 0 号卡；有多卡时想换卡改这里，别去改 python。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ---- 出发前硬校验：解释器 + CUDA platform ----
echo "python     : $(which python)"
echo "GPU        : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
python - <<'PYCHK'
import sys
import openmm as mm
names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
print(f"OpenMM     : {mm.version.version}")
print(f"platforms  : {names}")
if "CUDA" not in names:
    sys.exit("❌ 这个 OpenMM 装不出 CUDA platform，先修环境再跑，别浪费 walltime。")
try:
    mm.Platform.getPlatformByName("CUDA")
except Exception as exc:
    sys.exit(f"❌ CUDA platform 存在但初始化失败：{exc}")
print("CUDA check : ✅")
PYCHK

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)

# ---- 选一个 RUN_MODE 取消注释 ----
#MODE=mts_validation
#MODE=ca_source_weight_sweep
#MODE=polarization_ablation
#MODE=profile_force_cost
#MODE=default_ablation
# 最小 replicate 矩阵：5 configs x 3 seeds = 15 条独立轨迹，明显更慢。
MODE=long_convergence

LOG="logs/${MODE}_${STAMP}.log"
echo "run mode   : $MODE"
echo "log        : $LOG"
L_IPS_RUN_MODE=$MODE python -u test_lips_vs_pmeV2.6.py 2>&1 | tee "$LOG"
