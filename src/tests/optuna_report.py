"""
Report findings from an experiment-1 Optuna study (exp1_{gru,lstm,rnn}.py),
read directly from its SQLite storage under optuna/.
"""

import argparse
import json
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


def trial_to_dict(t):
    return {
        "number": t.number,
        "state": t.state.name,
        "value": t.value,
        "params": t.params,
        "user_attrs": t.user_attrs,
        "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
        "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
        "duration_seconds": t.duration.total_seconds() if t.duration else None,
    }


trials_sorted = sorted(trials, key=lambda t: t.datetime_start or t.datetime_complete or t.number)
report = {
    "study_name": study_name,
    "storage": str(db_path),
    "trials": [trial_to_dict(t) for t in trials_sorted],
}

completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
if completed:
    report["best_trial"] = trial_to_dict(study.best_trial)

print(json.dumps(report, indent=2))
