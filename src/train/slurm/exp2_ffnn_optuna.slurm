#!/bin/bash
#SBATCH --job-name=exp2_ffnn_optuna
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=turing-wide
#SBATCH --output=src/train/slurm/runs/exp2_ffnn_optuna_%A_%a.out
#SBATCH --error=src/train/slurm/runs/exp2_ffnn_optuna_%A_%a.err

# Experiment 2: Optuna search for the FFNN, one array task per dataset
# (0 = nca, 1 = nn). Submit from the repo root:
#   mkdir -p src/train/slurm/runs && sbatch src/train/slurm/exp2_ffnn_optuna.slurm
# Trials go to optuna/ffnn_<DATA_TAG>_<dataset>.db; resubmitting adds N_TRIALS
# more trials to the same study. TensorBoard logs go to runs/exp2/.
# CPU only: the FFNN is small, the run time is dominated by data loading.

source ~/.bashrc
conda activate carnn_venv
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATASETS=(nca nn)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
DATA_TAG=1        # experiment-1 data (3 years of training data, tau=6)
N_TRIALS=60
NUM_EPOCHS=5

echo "Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID | dataset=$DATASET | data=$DATA_TAG | trials=$N_TRIALS | epochs=$NUM_EPOCHS"

python -m src.train.exp2_ffnn \
    --dataset "$DATASET" \
    --data-tag "$DATA_TAG" \
    --n-trials "$N_TRIALS" \
    --num-epochs "$NUM_EPOCHS" \
    --num-workers 4 \
    --no-use-gpu

python -m src.tests.optuna_report --model ffnn --dataset "$DATASET" --data-tag "$DATA_TAG"
