#!/bin/bash
#SBATCH --job-name=h200_train
#SBATCH --mem=32G
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --account=h200ea
#SBATCH --partition=h200ea
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
source ~/.bashrc
conda activate icl

cd ..
python main.py "$@" dataset=slimpajama_6b training.compile=true training.num_save_steps=100000 training.num_workers=8 training.epochs=10