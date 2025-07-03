#!/bin/bash
#SBATCH --job-name=icl-language
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=scavenger-gpu
#SBATCH --exclude=dcc-youlab-gpu-28,dcc-gehmlab-gpu-56
#SBATCH --nodelist=dcc-allenlab-gpu-[01-04],dcc-allenlab-gpu-[05-12],dcc-majoroslab-gpu-[01-08],dcc-yaolab-gpu-[01-08],dcc-wengerlab-gpu-01,dcc-engelhardlab-gpu-[02-04],dcc-motesa-gpu-[01-04],dcc-pbenfeylab-gpu-[01-04],dcc-vossenlab-gpu-[01-04],dcc-youlab-gpu-[01-56],dcc-mastatlab-gpu-01,dcc-viplab-gpu-01,dcc-youlab-gpu-57
#SBATCH --requeue
#SBATCH --array=0-6
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -e
source ~/.bashrc
conda activate icl

cd ..

# Define the list of learning rates
lrs=(0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3)

# Get the current learning rate based on SLURM_ARRAY_TASK_ID
lr=${lrs[$SLURM_ARRAY_TASK_ID]}
echo "Running with learning rate: $lr"

# Run the training script with the selected learning rate
python main.py training.optimizer.lr=$lr "$@" training.epochs=3