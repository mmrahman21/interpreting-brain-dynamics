#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH -p  qTRDGPUL
#SBATCH -t  1440
#SBATCH -J  mrahman21
#SBATCH --mem=20g
#SBATCH -e /data/users2/mrahman21/Saliency/logs/ABIDE_gainSelect_LSTMonly_UFPT_small_data_toward_RAR_models-%a.err
#SBATCH -o /data/users2/mrahman21/Saliency/logs/ABIDE_gainSelect_LSTMonly_UFPT_small_data_toward_RAR_models-%a.out
#SBATCH -A  PSYC0002

echo "$SLURM_ARRAY_TASK_ID"

sleep 5s
export OMP_NUM_THREADS=1
source ~/.bashrc
conda activate myclone

python milcLSTMonly_smallData_gainSelection.py $SLURM_ARRAY_TASK_ID 1

sleep 10s

