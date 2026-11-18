#!/bin/bash
#SBATCH --job-name=rollout
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=turing-wide
#SBATCH --output=src/tests/slurm/runs/rollout_%j.out
#SBATCH --error=src/tests/slurm/runs/rollout_%j.err

# Rollout errors over every start of the test set, then the plots, one after the other.


CHECKPOINT=models/lstm_1_nca_cpu_0917_1447.pt
FILL=neighbour_mean  # or ground_truth
ERRORS=outputs/ca_results/rollout_errors/$(basename "$CHECKPOINT" .pt)_${FILL}_allstarts_job${SLURM_JOB_ID}.json

#activate my env
source ~/.bashrc
conda activate carnn_venv

# after the env setup (.bashrc and conda activate can trip -u): stop at the first error, so plotting never runs on a missing or partial errors file

set -euo pipefail

# torch would otherwise use every core of the node, not the ones Slurm gave the job
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Checkpoint: $CHECKPOINT | fill: $FILL"
python -m src.tests.compute_rollout_errors --checkpoint "$CHECKPOINT" --fill "$FILL" --out "$ERRORS"

for METRIC in mae rmse mse; do
    python -m src.tests.plot_rollout "$ERRORS" --metric "$METRIC" --baseline
done

