#!/bin/bash
#SBATCH --job-name=exp1_all
#SBATCH --array=0-2                  # 3 jobs: indices 0, 1, 2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --partition=lovelace
#SBATCH --output=src/train/slurm/runs/exp1_all_%A_%a.out   # %A = job ID, %a = array index
#SBATCH --error=src/train/slurm/runs/exp1_all_%A_%a.err

#activate my env
source ~/.bashrc
conda activate carnn_venv

#map array to script name
SCRIPTS=("exp1_rnn" "exp1_lstm" "exp1_gru")
SCRIPT=${SCRIPTS[$SLURM_ARRAY_TASK_ID]}
echo "Running : $SCRIPT"
python -m src.train.${SCRIPT} --dataset nca --n-trials 45 --use-gpu
