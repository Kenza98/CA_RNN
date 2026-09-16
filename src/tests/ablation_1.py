"""
Evaluate the fixed-hyperparameter checkpoints from src/train/train_fixed.py
on the held-out experiment-1 test sets (1_nca_test.pt, 1_nn_test.pt), for
each of GRU, LSTM and VanillaRNN.

The val set was already used by the Optuna search (exp1_*.py) to compare
hyperparameter configs, so it's not reused here.
"""

import json
import argparse
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.gru import GRU
from src.models.lstm import LSTM
from src.models.VanillaRNN import VanillaRNN
from src.utils.normalize import normalize
from src.utils.evaluate import evaluate_model

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = ["nca", "nn"]
MODEL_CLASSES = {"gru": GRU, "lstm": LSTM, "rnn": VanillaRNN}
STATE_DICT_KEYS = {
    "gru": "GRUStateDict",
    "lstm": "LSTMStateDict",
    "rnn": "VanillaRNNStateDict",
}
# train_fixed.py tags checkpoints with model.__class__.__name__.lower(), not the CLI --model value
CHECKPOINT_PREFIXES = {"gru": "gru", "lstm": "lstm", "rnn": "vanillarnn"}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=list(MODEL_CLASSES),
    default=None,
    help="Evaluate only this model; omit to evaluate all three",
)

args = parser.parse_args()

models_to_eval = (
    {args.model: MODEL_CLASSES[args.model]} if args.model else MODEL_CLASSES
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)


def latest_checkpoint(model_key, dataset):
    pattern = re.compile(rf"^{CHECKPOINT_PREFIXES[model_key]}_1_{dataset}_.*\.pt$")
    matches = sorted(
        (f for f in MODEL_DIR.iterdir() if pattern.match(f.name)),
        key=lambda f: f.stat().st_mtime,
    )
    return matches[-1] if matches else None


results = {}
hyperparams = {}  # first level above "nn" and "nca"

for dataset in DATASETS:
    test_data = torch.load(
        DATA_DIR / f"1_{dataset}_test.pt", map_location="cpu", weights_only=False
    )
    X_test, Y_test = test_data["X"], test_data["Y"]
    input_dim = X_test.shape[-1]
    output_dim = 1

    for model_key, model_class in models_to_eval.items():
        print(f"\n=== {model_key.upper()} on {dataset} ===")

        model_file = latest_checkpoint(model_key, dataset)
        if model_file is None:
            print(f"No checkpoint found for {model_key}/{dataset}, skipping.")
            continue
        print(f"Loading: {model_file}")

        # load the model checkpoint
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
        # get hyperparams from checkpoint
        hidden_dim = checkpoint["hidden_dim"]
        num_layers = checkpoint["num_layers"]
        lr = checkpoint["lr"]
        seed = checkpoint["seed"]
        global_mean = checkpoint["global_mean"]
        global_std = checkpoint["global_std"]

        current = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "lr": lr,
            "seed": seed,
        }
        if hyperparams and hyperparams != current:
            # this should only go off if hyperparams had values and changed
            print(f"Warning: {model_key}/{dataset} hyperparams differ: {current}")

        hyperparams = current

        # load the model from input_dim, hidden_dim, output_dim, num_layers
        model = model_class(
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
        )
        model.load_state_dict(checkpoint[STATE_DICT_KEYS[model_key]])

        # normalize and standardize the results with checkpoint stored stats
        X, Y = normalize(X_test, Y_test, global_mean, global_std)
        test_loader = DataLoader(TensorDataset(X, Y), batch_size=256, shuffle=False)

        # get test metrics by calling utils evaluate_model
        metrics = evaluate_model(model, test_loader, device)  # still standardized
        mse = metrics["mse"].item()
        mae = metrics["mae"].item()
        # de-standardize
        mse = mse * global_std**2
        mae = mae * global_std
        rmse = mse**0.5
        print(
            f"- - - >> MSE = {mse:.6f} °C²\n >> RMSE = {rmse:.6f} °C\n >> MAE = {mae:.6f} °C \n"
        )

        results.setdefault(dataset, {})[model_key] = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        }

suffix = f"_{args.model}" if args.model else ""
with open(RESULTS_DIR / f"ablation_1{suffix}.json", "w") as f:
    json.dump({"hyperparams": hyperparams, "test results": results}, f, indent=2)
