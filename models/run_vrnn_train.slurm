#!/bin/bash
#SBATCH --job-name=vrnn_train
#SBATCH --output=runs/vrnn_train_%j.out
#SBATCH --error=runs/vrnn_train_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate carnn_venv

cd /u/kbenjelloun/CA_RNN
python -m models.RnnMoore.py --use-gpu



