#!/bin/bash
#SBATCH --job-name=train_fixed_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --partition=lovelace
#SBATCH --output=src/train/slurm/runs/train_fixed_all_%j.out
#SBATCH --error=src/train/slurm/runs/train_fixed_all_%j.err

#activate my env
source ~/.bashrc
conda activate carnn_venv

#run the 6 experiments
#python -m src.train.train_fixed --model rnn  --dataset nca --use-gpu --hidden-dim 5 --num-layer 4 --num-epochs 5
python -m src.train.train_fixed --model lstm --dataset nca --use-gpu --hidden-dim 125 --num-layers 1 --num-epochs 25 --lr 2e-5
#python -m src.train.train_fixed --model gru  --dataset nca --use-gpu --hidden-dim 5 --num-layer 4 --num-epochs 5
#python -m src.train.train_fixed --model rnn  --dataset nn  --use-gpu --hidden-dim 5 --num-layer 4 --num-epochs 5
python -m src.train.train_fixed --model lstm --dataset nn  --use-gpu --hidden-dim 125 --num-layers 1 --num-epochs 25 --lr 2e-5
#python -m src.train.train_fixed --model gru  --dataset nn  --use-gpu --hidden-dim 5 --num-layer 4 --num-epochs 5

