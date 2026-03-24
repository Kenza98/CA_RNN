import torch
import numpy as np
from pathlib import Path
from datetime import date
from random import randint

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

test_data_file = DATA_DIR / "sst_test_set.pt"

data = torch.load(test_data_file)
X, Y = data["X"], data["Y"]
N = Y.shape[0]
print(N)
I = [randint(0, N-1) for i in range(5)]

print(X[I])






exit(0)
print(torch.isnan(Y).sum())

for k, v in data.items():
    print(k)

print(X.shape, Y.shape)

start_test_date = date(2025, 1, 1)
end_test_date = date(2026, 2, 28)

print(start_test_date, end_test_date)

x_nan_count = torch.isnan(X).sum()
print(x_nan_count)