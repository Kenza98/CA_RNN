# CA_RNN: Cellular Automata Recurrent Neural Network for Adriatic Data
## Overview
This repository contains the implementation of the **CA_RNN** model, a recurrent neural network (RNN) designed for spatiotemporal modeling of Adriatic Sea reanalysis dataset taken from Copernicus.

https://www.copernicus.eu

It is organized as a Python package with structured modules for data preparation, model definition, and evaluation.


---
## Repository Structure

```
CA_RNN_repo/
├── data/                 # Raw and processed datasets
├── models/               # Neural network definitions
├── tests/                # Evaluation and plotting scripts
├── pyproject.toml        # Poetry configuration and dependencies
├── poetry.lock           # Locked dependency versions
└── README.md             # This file

```

---
## Requirements

- Python **3.11**
- Poetry (recommended) or pip
- The package dependencies are defined in `pyproject.toml` and `poetry.lock`.

---

## Installation

### Option 1: Using Poetry (easiest because used in this package)

1. **Install Poetry** (if not already installed):
```bash
   pip install poetry
```

2. Install dependencies:
```bash
   poetry install
```

3. activate the thuis created Virtual Environment:
```bash
   poetry shell
```

### Option 2 : Using **Anaconda** (recommended)

1. **Create a new environment**:
   ```bash
   conda create -n fvg-adriatic-data-py3.11 python=3.11
   ```
2. **Activate the environment**:
   ```bash
   conda activate fvg-adriatic-data-py3.11
   ```
3. **Install dependencies** (using pip inside conda):
   ```bash
   poetry export -f requirements.txt --output requirements.txt
   pip install -r requirements.txt
   ```


### Option 3: Using pip (Not recommended at all)
poetry export -f requirements.txt --output requirements.txt
pip install -r requirements.txt


---

### Usage

Once installed, the package can be used in the following way:

1. Running **RnnMoore.py** will train the model and write the final model's parameters into **rnn_moore.pt**
2. Running **test_RNN.CA.py** will test the model and write the test results to **rnn_moore.pt**
3. The files where visualizations scripts are available are:
- visualize_results.py
4. Some visualizations are available in the **.png** files.
---

### Note to reader :
If you have any suggestions of new metrics to use for training / testing or visualizations you wish to be developped, please open an issue and it will be adressed.






















































