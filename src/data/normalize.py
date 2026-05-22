import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# compute stats from train set only
train_data = torch.load(DATA_DIR / "sst_train_set_47339.pt", map_location="cpu", weights_only=False)
X_train = train_data["X"]
Y_train = train_data["Y"]

mean = X_train.mean()
std = X_train.std()
print(f"Train mean: {mean:.4f}, std: {std:.4f}")

def normalize(X, Y, mean, std):
    return (X - mean) / std, (Y - mean) / std

# normalize and save all splits
for split in ["train", "val", "test"]:
    data = torch.load(DATA_DIR / f"sst_{split}_set_47339.pt", map_location="cpu", weights_only=False)
    X_norm, Y_norm = normalize(data["X"], data["Y"], mean, std)
    torch.save({
        "X": X_norm,
        "Y": Y_norm,
        "mean": mean,
        "std": std,
        "start_time": data["start_time"],
        "end_time": data["end_time"],
    }, DATA_DIR / f"sst_{split}_set_norm.pt")
    print(f"Saved normalized {split} set")

print("Done.")
