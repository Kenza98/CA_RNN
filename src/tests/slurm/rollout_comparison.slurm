#!/bin/bash
#SBATCH --job-name=rollout_comparison
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --partition=turing-wide
#SBATCH --output=src/tests/slurm/runs/rollout_comparison_%j.out
#SBATCH --error=src/tests/slurm/runs/rollout_comparison_%j.err

# Comparison plots: the all-starts mean of every checkpoint on one plot, one plot for nca and one for nn.
# Needs the error files of rollout.slurm, so submit it once that array job is done:
#   sbatch --dependency=afterok:<array job id> src/tests/slurm/rollout_comparison.slurm


source ~/.bashrc
conda activate carnn_venv
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

FILL=ground_truth  #neighbour_mean
ERRORS_DIR=outputs/ca_results/rollout_errors
ERRORS_FILES=("$ERRORS_DIR"/*_${FILL}_allstarts.json)
PLOTS=outputs/rollout_plots
METRICS=(mae rmse)

mkdir -p "$PLOTS"

# without a match, the glob stays as the literal pattern: stop rather than plot nothing
if [ ! -e "${ERRORS_FILES[0]}" ]; then
    echo "No error files matching $ERRORS_DIR/*_${FILL}_allstarts.json" >&2
    exit 1
fi

echo "Fill: $FILL | ${#ERRORS_FILES[@]} error file(s):"
printf '  %s\n' "${ERRORS_FILES[@]}"

for METRIC in "${METRICS[@]}"; do
    python -m src.tests.plot_rollout_comparison "${ERRORS_FILES[@]}" \
        --metric "$METRIC" --baseline all --out-dir "$PLOTS"
done
