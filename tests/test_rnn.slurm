#!/bin/bash
#SBATCH --job-name=vrnn_test
#SBATCH --output=runs/vrnn_test_%j.out
#SBATCH --error=runs/vrnn_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate carnn_venv

cd /u/kbenjelloun/CA_RNN
python -m tests.test_RNN_CA --use-gpu
