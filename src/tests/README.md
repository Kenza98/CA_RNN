# src/tests

Evaluation scripts. Run them from the repo root as modules, e.g.
`python -m src.tests.compute_rollout_errors --checkpoint models/<file>.pt`.

## Autoregressive rollouts

A rollout starts from 6 real frames, predicts the next grid, feeds that
prediction back in, and repeats (30 steps by default).

### Running them

1. `compute_rollout_errors.py` runs the model's rollout and the
   neighbour-average baseline from every start of the test set (or one, with
   `--start`), and saves MAE and RMSE per start and per step to
   `outputs/ca_results/rollout_errors/<checkpoint>_<fill>_<allstarts|startN>_<MMDD_HHMMSS>.json`.
   The rollout is picked from the checkpoint: `nca` uses the CA framing, `nn` the one without CA.
2. `plot_rollout.py <errors.json> --metric mae|rmse|mse` plots that file:
   the mean over starts with the 25-75 percentile band, or a single start with
   `--start N`. `--baseline` adds the neighbour average in grey.

On the cluster, `slurm/rollout.slurm` runs both one after the other (CPU only;
set the checkpoint and fill at the top of the script).

### Building blocks: `rollout.py`

| Function | Role |
|---|---|
| `load_sst_series(split)` | Raw SST of a split (`data/raw/sst_<split>.nc`) as `(T, H, W)` in °C |
| `load_checkpoint(path, device)` | Rebuilds a model saved by `src/train/train_fixed.py`, along with its train mean and std |
| `predict_next_grid(model, window, extractor, ...)` | One step: a window of `seq_length` grids → the next grid. It reuses the training extractors and NaN mask from `src/data/extractors.py`. Cells the model can't predict are left as NaN |
| `fill_with_ground_truth` | Unpredicted sea cells get the real next values |
| `fill_with_neighbour_mean` | Unpredicted sea cells get the mean of their available (non-NaN) 3×3 neighbours, in passes, with the previous value as a fallback. No ground truth is used |
| `rollout_ca_rnn(...)` | RNN + CA: each cell sees its 3×3 neighbourhood (`extract_neighborhood`) |
| `rollout_rnn(...)` | RNN without CA: each cell sees its own history (`extract_pointwise`) |
| `rollout_neighbour_average(...)` | Baseline without a model: each step is the 3×3 mean of the previous grid |
| `errors_per_step` | MAE and RMSE for each step, computed **only on cells the model predicted** |

The cells the model can't predict are the 1-cell outer ring of the grid and
coastal cells whose 3×3 neighbourhood touches land. On the test split this
is 274 of the 1015 sea cells. A fill function decides what goes into those
cells before the grid is fed back (`--fill ground_truth|neighbour_mean`,
default `neighbour_mean`). Without CA, filled cells never reach a predicted
cell's input, so the fill does not change that rollout's errors.

All three rollouts predict the same cells, so their errors are directly comparable.

Unit tests: `python -m unittest unitary_tests.test_rollout -v`.

Still to add: `rollout_ffnn`, a feed-forward model seeded from a single initial grid.

### Legacy: `cellular_automaton_gru.py` and `ca_plots.py`

This was the first CA rollout of the GRU. It is kept as is so that the
results in `outputs/ca_results/v1_baseline/` can be reproduced and checked.
`rollout.py` was written because of these problems:

- **Ground truth leaked in.** The outer ring and coastal cells were filled
  with real values at every step and fed back into the window. The coastal
  cells were also counted in the MAE, which lowered the reported error.
  In `rollout.py` this is one optional fill mode, and the metric never
  includes filled cells.
- **No normalisation.** Raw °C went into the model, which was most likely
  trained on normalised data (`train_gru.py` uses the `_norm` sets). `rollout.py` normalises inputs and converts outputs back, using the
  stats stored in the checkpoint.
- **Hard-coded setup.** Sequence length 4, `GRU(9, 56, 1)`, and "latest
  `gru*.pt` in `models/`". Now the sequence length is 6 and the architecture
  comes from the checkpoint.
- **Off-by-four.** The ground-truth generator stopped 4 frames early, and
  `zip` silently dropped the last steps.
- **Overwrites.** Every run wrote to the same `ca_errors.json`. Its outputs
  have been moved to `outputs/ca_results/v1_baseline/`, and both legacy
  scripts now point there.
