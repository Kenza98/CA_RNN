#!/bin/bash
#SBATCH --job-name=ablation_1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=lovelace
#SBATCH --output=src/tests/slurm/runs/ablation_1_%j.out
#SBATCH --error=src/tests/slurm/runs/ablation_1_%j.err

#activate my env
source ~/.bashrc

conda activate carnn_venv

python -m src.tests.ablation_1
