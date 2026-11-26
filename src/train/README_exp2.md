# Experiment 2: feed-forward baseline (FFNN)

## Question

Experiment 1 trained three recurrent models (vanilla RNN, GRU, LSTM), each with and without the cellular-automaton framing (`nca`: 3×3 neighbourhood per cell; `nn`: the cell's own history).
Experiment 2 asks whether recurrence is needed at all.
A plain feed-forward network (`src/models/ffnn.py`) sees the same 6-day window, flattened into one vector, and predicts the next day.
If it matches the RNNs, the gain comes from the input (history, neighbourhood), not from the recurrent architecture.

A first FFNN trained with untuned hyperparameters (width 64, 1 hidden layer, lr 1e-4, 3 epochs) already lands within about 1% of the RNNs' one-step test error (`nca`: RMSE 0.345 °C, persistence 0.379 °C).
Each model was trained with a single seed, so no ranking is drawn from this yet.

## Data

Same files as experiment 1 (`data/1_{nca,nn}_{train,val,test}.pt`), Copernicus Mediterranean reanalysis SST, northern Adriatic (12–16°E, 44.5–45.5°N):

| split | dates | samples |
|---|---|---|
| train | 2021-01-01 – 2023-12-31 (3 years) | 806,949 |
| val | 2024 | 266,760 |
| test | 2025-01-01 – 2026-07-31 | 423,111 |

Window length τ = 6 days. Every split is normalised with the training mean and std.

## Pipeline

| step | script | output |
|---|---|---|
| 1. Hyperparameter search | `src/train/exp2_ffnn.py` (cluster: `src/train/slurm/exp2_ffnn_optuna.slurm`) | `optuna/ffnn_1_<dataset>.db`, `results/exp2_optuna_ffnn_1_<dataset>.json`, TensorBoard logs in `runs/exp2/` |
| 2. Search report | `src/tests/optuna_report.py --model ffnn --dataset <dataset>` | `results/optuna_report_ffnn_1_<dataset>.json` |
| 3. Final checkpoint | `src/train/fixed_ffnn.py` with the best parameters (the search prints the command) | `models/feedforwardnn_1_<dataset>_<run>.pt`, training plots in `outputs/exp2_ffnn/<run>/` |
| 4. Test set | `src/tests/test_ffnn.py` | `results/exp2_ffnn_test_<checkpoint>.json`, error histogram in `outputs/exp2_ffnn/test/` |
| 5. Comparison with the RNNs | multi-seed runs + Mann–Whitney U test on aggregated errors (being designed) | |

### 1. Hyperparameter search

```bash
mkdir -p src/train/slurm/runs
sbatch src/train/slurm/exp2_ffnn_optuna.slurm          # array: task 0 = nca, task 1 = nn
# or locally
python -m src.train.exp2_ffnn --dataset nca --n-trials 60
```

Search space (TPE sampler, seed 42, 60 trials, 5 epochs per trial as in experiment 1):

| parameter | values |
|---|---|
| `width` | 16, 32, 64, 128, 256 |
| `num_layers` | 1 – 4 hidden layers |
| `batch_size` | 128, 256, 512 |
| `lr` | 1e-5 – 1e-3, log scale |

The objective is the validation MSE (°C²) after the last epoch.
Validation MSE is reported after every epoch, and a median pruner stops trials that fall behind (after 8 complete trials and the first epoch).
Every trial starts from the same seed, so trials differ only by their hyperparameters.
Resubmitting the job adds trials to the same study (`load_if_exists`).

Unlike experiment 1, the learning rate is searched: the FFNN reached a 7× higher validation MSE with lr 1e-5 than with 1e-4 after 3 epochs.
The experiment-1 RNN searches kept lr fixed at 1e-4, so the FFNN gets one more tuned parameter than the RNNs; this is to be kept in mind when the two are compared.

### Viewing the results in TensorBoard

```bash
tensorboard --logdir runs/exp2
```

On the cluster, run it on the login node and forward the port (`ssh -L 6006:localhost:6006 <cluster>`), or copy `runs/exp2/` back and run it locally.

- **Scalars**, one run per trial (`ffnn_1_<dataset>/trial_XXXX`): per-step training loss (°C²), total gradient norm after clipping (every 10 steps), per-epoch train and validation MSE, validation MAE.
- **HParams**: one row per trial with its hyperparameters, final and best validation MSE; sortable, with parallel-coordinates and scatter views.
- **Images**, run `ffnn_1_<dataset>/study`: Optuna's optimisation history, parameter importances (PED-ANOVA), slice plot, parallel coordinates and per-epoch learning curves of all trials.

## Status

- RNN checkpoints for the first comparison batch (experiment-1 data, τ = 6): done.
- FFNN: Optuna search to run, then the final checkpoint (step 3) completes the batch.
- Multi-seed training and U test on aggregated errors: in design, starts once the batch is complete.

## Next steps

Two factors will be varied, and every model (vanilla RNN, GRU, LSTM, FFNN; `nca` and `nn`) retrained each time:

1. **Amount of training data.**
   Experiment 1 (the data in use now) trains on 3 years.
   The training set will be extended to 5 years, then to 10 years.
   Validation and test periods should stay unchanged, so that every model is scored on the same days.
2. **Window length τ.**
   From 6 days now to 10, then 15.

Each new dataset is built with `src/data/prep.py` (`--experiment`, `--seq-length`) and gets its own file prefix; `exp2_ffnn.py --data-tag <prefix>` and `optuna_report.py --data-tag <prefix>` then point at it, and the study is stored as `ffnn_<prefix>_<dataset>`.

To do before the τ = 10 / 15 runs:

- `src/train/train_fixed.py` does not store `seq_length` in RNN checkpoints, and the rollout scripts (`src/tests/rollout.py`) default to τ = 6. Both need to read τ from the checkpoint.
- `src/tests/rollout.py` has no FFNN rollout yet (`rollout_ffnn`).
- `src/tests/test_ffnn.py` and `src/train/fixed_ffnn.py` read the experiment-1 files only (`1_<dataset>_*`); they need a `--data-tag` option as well.
