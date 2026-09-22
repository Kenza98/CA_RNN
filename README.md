# CA_RNN: Cellular Automata Recurrent Neural Network for Adriatic Data

## Overview

This repository contains the implementation of **CA_RNN**, a set of recurrent
architectures (vanilla RNN, GRU, LSTM) trained and evaluated on a cellular-automaton
formulation of spatiotemporal dynamics from the Adriatic Sea reanalysis dataset,
taken from Copernicus (https://www.copernicus.eu).

---
## Repository Structure

```
CA_RNN/
├── data/                  # Raw and processed datasets (.pt tensors, raw/ subfolder)
├── models/                # Trained model checkpoints (.pt)
├── outputs/                # Training/evaluation outputs (csv, plots, logs)
├── optuna/                # Optuna hyperparameter search results (sqlite db)
├── results/                # Test/evaluation results
├── ca_viz_app/             # Visualization app and notes
├── paper_visu/             # Figures/visualizations for the paper
├── archive/                 # Older notes and reports
├── src/
│   ├── data/                # Dataset preparation, downloading, normalization
│   ├── models/               # Model definitions (VanillaRNN, GRU, LSTM)
│   ├── train/                 # Training entry points (exp1_*.py, train_*.py) + SLURM scripts
│   ├── tests/                  # Evaluation, ablations, plotting scripts
│   └── utils/                   # Shared helpers (train loop, checkpoints, evaluation, plots)
├── tests/                    # Misc inspection scripts (e.g. inspect_pt.py)
├── environment.yml           # Conda environment specification
├── licence.txt               # Licence specification
└── README.md                 # This file
```

---
## Requirements

- Python **3.11**
- Conda / Miniforge (recommended) or pip
- Dependencies are defined in `environment.yml`

---
## Installation

### Option 1: Using Conda (recommended)

1. **Create the environment** from the provided spec:
   ```bash
   conda env create -f environment.yml
   ```
2. **Activate it**:
   ```bash
   conda activate carnn_venv
   ```

### Option 2: Using pip

Create your own virtual environment, then install the packages listed in
`environment.yml` manually (no `requirements.txt`/`pyproject.toml` is currently
maintained in this repo).

---
## Usage

Training and evaluation are organized under `src/`:

- **Training**: run one of the experiment entry points in `src/train/`, e.g.
  ```bash
  python -m src.train.exp1_rnn
  ```
  (equivalents exist for GRU and LSTM: `exp1_gru.py`, `exp1_lstm.py`). Trained
  checkpoints are written to `models/`.
- **Testing / evaluation**: scripts in `src/tests/` (e.g. `test_vrnn.py`,
  `test_gru.py`, `test_lstm.py`) load a checkpoint from `models/` and write
  results to `results/` or `outputs/`.
- **Visualization**: `src/tests/visualize_results.py` and `src/tests/ca_plots.py`
  produce plots from test outputs.
- **Hyperparameter search**: Optuna studies are stored in `optuna/` as SQLite
  databases; see `src/tests/optuna_report.py` for generating reports from them.
- **SLURM**: batch job scripts for cluster training are under
  `src/train/slurm/` and `src/tests/slurm/`.

---
## Licence

This source code is made available under the licence CC BY-NC-SA 4.0. Please read `licence.txt`.

---
## Note to reader

If you have suggestions for new metrics to use for training, testing, or
visualization, please open an issue and it will be addressed.
