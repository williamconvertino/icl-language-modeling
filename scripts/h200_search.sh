#!/bin/bash
#SBATCH --job-name=h200_train
#SBATCH --mem=32G
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --account=h200ea
#SBATCH --partition=h200ea
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
source ~/.bashrc
conda activate icl

lrs=(0.00005 0.0001 0.0003)
lr=${lrs[$SLURM_ARRAY_TASK_ID]}

cd ..
python main.py "$@" training.optimizer.lr=$lr dataset=slimpajama_6b training.compile=true training.num_save_steps=50000 training.num_workers=8 training.epochs=10