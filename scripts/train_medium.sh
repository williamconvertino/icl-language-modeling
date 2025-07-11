#!/bin/bash
#SBATCH --job-name=icl-train-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --partition=scavenger-gpu
#SBATCH --nodelist=dcc-allenlab-gpu-[01-04],dcc-allenlab-gpu-[05-12],dcc-majoroslab-gpu-[01-08]
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
source ~/.bashrc
conda activate icl

cd ..
python main.py "$@" training.optimizer.lr=0.0002