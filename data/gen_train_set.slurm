#!/bin/bash
#SBATCH --job-name=gen_train_data
#SBATCH --output=runs/gen_data_%j.out
#SBATCH --error=runs/gen_data_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate carnn_venv

python xarray_prep.py

