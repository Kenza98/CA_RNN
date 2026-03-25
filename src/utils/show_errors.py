import torch
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "shuffle_True_ep100"


def find_result_file(pattern_str):
    pattern = re.compile(pattern_str)
    files = [f for f in OUT_DIR.iterdir() if pattern.match(f.name)]
    if not files:
        raise FileNotFoundError(f"No file matching '{pattern_str}' in {OUT_DIR}")
    return sorted(files, key=lambda f: f.stat().st_mtime)[-1]


# loading error dicts for baseline
bf = find_result_file(r".*baseline.*\.pt$")  # baseline file
lf = find_result_file(r".*lstm.*\.pt$")  # baseline file
gf = find_result_file(r".*gru.*\.pt$")  # baseline file
vf = find_result_file(r".*vrnn.*\.pt$")  # baseline file

# print(f"The files :\n1. {bf}\n2. {lf}\n3. {gf}\n4. {vf}\n\n-----------------\n\n")

baseline = torch.load(bf, weights_only=False)
vanilla_rnn = torch.load(vf, weights_only=False)
lstm = torch.load(lf, weights_only=False)
gru = torch.load(gf, weights_only=False)

abs_errs = [baseline["mae"], vanilla_rnn["mae"], lstm["mae"], gru["mae"]]

labels = ["Baseline MAE = ", "Vanilla MAE = ", "LSTM MAE = ", "GRU MAE = "]

for l, e in zip(labels, abs_errs):
    print(f"{l} : {e}\n")

ae_tensors = [
    baseline["absolute_error"],
    vanilla_rnn["absolute_error"],
    lstm["absolute_error"],
    gru["absolute_error"]
]

min_ae = [torch.min(t).item() for t in ae_tensors]

for l, e in zip(labels, min_ae):
    print(f"{l} : {e}\n")


max_ae = [torch.max(t).item() for t in ae_tensors]

for l, e in zip(labels, max_ae):
    print(f"{l} : {e}\n")



"""
for e in E:
    print(f"{e.keys()}\n")
"""
