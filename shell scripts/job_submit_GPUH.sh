#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH -p  qTRDGPUH
#SBATCH --nodelist="trendsdgxa001.rs.gsu.edu" 
#SBATCH -t  1440
#SBATCH -J  mrahman21
#SBATCH --mem=40g
#SBATCH -e /data/users2/mrahman21/Saliency/logs/Inception_synthetic_saliency_IG_random_dataset-%a.err
#SBATCH -o /data/users2/mrahman21/Saliency/logs/Inception_synthetic_saliency_IG_random_dataset-%a.out
#SBATCH -A  PSYC0002

echo "$SLURM_ARRAY_TASK_ID"

sleep 5s
export OMP_NUM_THREADS=1
source ~/.bashrc
conda activate myenv

python inception_synthetic.py $SLURM_ARRAY_TASK_ID 1 4

sleep 10s
