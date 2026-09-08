#!/bin/zsh
#PBS -q default
#PBS -l select=1:ncpus=16:ngpus=1:mem=50gb:host=yayoi31
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -N training
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
# run the environment module
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH

export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev
cd /home/ruigengji/L-IPS/2.5

cd /home/ruigengji/L-IPS/2.5 && L_IPS_LIGAND_DPOLAR=-0.15 L_IPS_LIGAND_SCOPE=all python -u run_zn_job.py --system 1aay --method cwld --ns 5 --seeds 3 > logs/T3_all_$(date +%H%M%S).log 2>&1; echo "exit=$?"; tail -25 $(ls -t logs/T3_all_*.log | head -1)