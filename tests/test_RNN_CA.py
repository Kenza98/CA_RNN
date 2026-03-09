import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os, sys
from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


print(PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

from models.VanillaRNN import VanillaRNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA version: {torch.version.cuda}", flush=True)


test_data_file = DATA_DIR / "sst_test_set.pt"   #\\CHECK all data test + train in this dir?
model_file = MODEL_DIR / "rnn_moore.pt"
output_file = OUT_DIR / "testset_results.pt"


data = torch.load(test_data_file, map_location=device)
model_loader = torch.load(model_file, map_location=device)

print("Top-level keys in model file:", model_loader.keys(), flush=True)

# recuperate test data
Xn = data["X"]
X = Xn[:, :, 4]  # select only the central cell.

Y = data["Y"]  # not normalized but nan free.
total_samples = X.shape[0]

print(f"X has shape N= {X.shape}\n\n")
print(f"Y has shape N= {Y.shape}\n\n")

test_dataset = TensorDataset(Xn, Y)
baseline_dataset = TensorDataset(X, Y)

batch_size = 256
output_dim = 1
input_dim = 9
hidden_dim = 7 * 8

test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
baseline_loader = DataLoader(baseline_dataset, batch_size, shuffle=True)

model = VanillaRNN(input_dim, hidden_dim, output_dim)
model.load_state_dict(model_loader["rnnMoore_stateDict"])
model = model.to(device)
"""
print("=== Infos sur le modèle ===")
print(model)  # architecture complète

# Nombre total de paramètres
n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params : {n_params:,} | Entraînables : {n_trainable:,}")

# Vérifie input/output
print(f"Input dim attendu : {model.rnn.input_size}")
print(f"Hidden dim : {model.rnn.hidden_size}")
print(f"Num layers : {model.rnn.num_layers}")
print(f"Output dim : {model.fc.out_features}")

# Teste avec un batch factice
dummy = torch.randn(2, 8, 1).to(device)   # batch=2, seq_len=8, feat=1
with torch.no_grad():
    out = model(dummy)
print(f"Dummy input shape : {tuple(dummy.shape)}")
print(f"Dummy output shape: {tuple(out.shape)}")
"""
total_samples = X.shape[0]

criterion = nn.MSELoss(reduction="sum")  # sum on GPU; divide later

neigh_indices = [i for i in range(9) if i != 4]

total_loss = torch.tensor(0.0, device=device)
total_mare = torch.tensor(0.0, device=device)
total_baseline_mse = torch.tensor(0.0, device=device)
total_baseline_mare = torch.tensor(0.0, device=device)
num_mare = torch.tensor(0.0, device=device)  # count for MARE denom

all_preds = []
all_baselines = []

model.eval()

with torch.no_grad():
    #baseline that only computes an average
    for batch_seq, batch_tar in baseline_loader:
        batch_seq = batch_seq.to(device, non_blocking=True)
        batch_tar = batch_tar.to(device, non_blocking=True)
        local_only = batch_seq[:, -1, neigh_indices]
        local_navg = local_only.mean(dim=1)
        # MSE (sum on GPU)
        total_baseline_mse += criterion(local_navg, batch_tar.squeeze_())
        # MARE with mask (GPU)
        target = batch_tar.squeeze(-1).abs()
        mask = target > 1e-6
        total_baseline_mare += (
            (local_navg - batch_tar.squeeze(-1)).abs()[mask] / target[mask]
        ).sum()
        all_baselines.append(local_navg.cpu())
    
    #actual model test
    for batch_seq, batch_tar in test_loader:
        batch_seq = batch_seq.to(device, non_blocking=True)
        batch_tar = batch_tar.to(device, non_blocking=True)
        # model (on GPU maintenant)
        outputs = model(batch_seq.unsqueeze_(-1))  # add feature dim

        # MSE (sum on GPU)
        total_loss += criterion(outputs, batch_tar)

        # MARE with mask (GPU)
        target = batch_tar.abs()
        target = batch_tar.squeeze_().abs()
        mask = target > 1e-6
        total_mare += ((outputs - batch_tar.squeeze()).abs()[mask] / target[mask]).sum()
        num_mare += mask.sum()
        all_preds.append(outputs.cpu())


# move to CPU once
avg_mse = (total_loss / total_samples).item()
avg_mare = (total_mare / num_mare).item()
baseline_mse = (total_baseline_mse / total_samples).item()
baseline_mare = (total_baseline_mare / num_mare).item()


#######CORRECT THE NEXT###############""
# *** MSE and baseline mse ***

print(f"average baseline MSE err : {baseline_mse}")
print(f"average MSE err : {avg_mse}")

print(f"Trial completed with average test MARE loss : {avg_mare:.4f}")
print(f"average baseline MARE err : {baseline_mare}")


# save results to output pt file

my_dict = {}
my_dict["avg_mse"] = avg_mse
my_dict["avg_mare"] = avg_mare
my_dict["baseline_mse"] = baseline_mse
my_dict["baseline_mare"] = baseline_mare
my_dict["num_samples"] = total_samples



# Concatenate all samples

my_dict["Y_model"] = torch.cat(all_preds)
my_dict["Y_baseline"] = torch.cat(all_baselines)
my_dict["Y_true"] = Y

torch.save(my_dict, output_file)

