#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH -p  qTRDGPUM
#SBATCH -t  1440
#SBATCH -J  mrahman21
#SBATCH --mem=40g
#SBATCH -e /data/users2/mrahman21/Saliency/logs/ADDICTION_gain_selection_UFPT_models-%a.err
#SBATCH -o /data/users2/mrahman21/Saliency/logs/ADDICTION_gain_selection_UFPT_models-%a.out
#SBATCH -A  PSYC0002

echo "$SLURM_ARRAY_TASK_ID"

sleep 5s
export OMP_NUM_THREADS=1
source ~/.bashrc
conda activate myenv

# Args: dataset index, UFPT/NPT, test start index/gain
python milcLSTMonly_arch_chk_addictionData.py 4 1 $SLURM_ARRAY_TASK_ID

sleep 10s
