"""
Report findings from an experiment-1 Optuna study (exp1_{gru,lstm,rnn}.py),
read directly from its SQLite storage under optuna/.
"""

import argparse
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OPTUNA_DIR = PROJECT_ROOT / "optuna"

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["gru", "lstm", "rnn"], required=True)
parser.add_argument("--dataset", choices=["nca", "nn"], required=True)
args = parser.parse_args()

study_name = f"{args.model}_1_{args.dataset}"
db_path = OPTUNA_DIR / f"{study_name}.db"

if not db_path.exists():
    raise SystemExit(f"No study found at {db_path}")

study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")

trials = study.trials
completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]

print(f"=== {study_name} ===")
print(f"Storage: {db_path}")
print(
    f"Total trials: {len(trials)} | complete: {len(completed)} | pruned: {len(pruned)} | failed: {len(failed)}"
)

if not completed:
    print("No completed trials yet.")
    raise SystemExit(0)

print(f"\nBest val MSE: {study.best_value:.6f}")
print(f"Best trial: #{study.best_trial.number}")
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

if study.best_trial.user_attrs:
    print("Best trial user attrs:")
    for k, v in study.best_trial.user_attrs.items():
        print(f"  {k}: {v}")

values = [t.value for t in completed if t.value is not None]
print(
    f"\nval MSE across completed trials: min={min(values):.6f} | max={max(values):.6f} | "
    f"mean={sum(values)/len(values):.6f}"
)
