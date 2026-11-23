#!/bin/bash
#SBATCH --job-name=rollout
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=turing-wide
#SBATCH --output=src/tests/slurm/runs/rollout_%A_%a.out
#SBATCH --error=src/tests/slurm/runs/rollout_%A_%a.err


source ~/.bashrc
conda activate carnn_venv
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

FILL=ground_truth  #neighbour_mean
CHECKPOINTS=(models/final_training_28/*.pt)
CHECKPOINT=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}
ERRORS=outputs/ca_results/rollout_errors/$(basename "$CHECKPOINT" .pt)_${FILL}_allstarts.json

PLOTS=outputs/rollout_plots
mkdir -p "$PLOTS"
BASE=$(basename "$CHECKPOINT" .pt)
IFS=_ read -r MODEL _  VARIANT _ <<< "$BASE"
OUT=$PLOTS/${MODEL}_${VARIANT}_${FILL}

echo "$MODEL"

case "$MODEL" in
    gru)         COL=pink ;;
    lstm)        COL=duck ;;
    vanillarnn)  COL=yellow ;;
esac

echo "$COL"
echo "Checkpoint: $CHECKPOINT | fill: $FILL"

python -m src.tests.compute_rollout_errors --checkpoint "$CHECKPOINT" --fill "$FILL" --out "$ERRORS"

if [ ! -s "$ERRORS" ]; then
    python -m src.tests.compute_rollout_errors --checkpoint "$CHECKPOINT" --fill "$FILL" --out "$ERRORS"
fi

for METRIC in mae rmse mse; do
    python -m src.tests.plot_rollout "$ERRORS" --metric "$METRIC" --col "$COL" --baseline all --out "${OUT}_${METRIC}"
done

