import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os, sys
from pathlib import Path
from datetime import datetime

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


test_data_file = (
    DATA_DIR / "sst_test_set.pt"
)  # \\CHECK all data test + train in this dir?

model_file = MODEL_DIR / "rnn_moore.pt"
output_file = OUT_DIR / "testset_results.pt"

timestamp = os.path.getmtime(model_file)
print("Last model update:", datetime.fromtimestamp(timestamp))

data = torch.load(test_data_file, map_location=device)
model_loader = torch.load(model_file, map_location=device)

print("Top-level keys in model file:", model_loader.keys(), flush=True)

# recuperate test data
Xn = data["X"]
X = Xn[:, :, 4]  # select only the central cell.

Y = data["Y"]  # not normalized but nan free.
total_samples = X.shape[0]

print(f"Xn (test model) has shape N= {Xn.shape}\n\n")

print(f"X (test baseline) has shape N= {X.shape}\n\n")
print(f"Y has shape N= {Y.shape}\n\n")

test_dataset = TensorDataset(Xn, Y)
baseline_dataset = TensorDataset(Xn, Y)

batch_size = 256
output_dim = 1
input_dim = 9
hidden_dim = 7 * 8

test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
baseline_loader = DataLoader(baseline_dataset, batch_size, shuffle=True)

model = VanillaRNN(input_dim, hidden_dim, output_dim)
model.load_state_dict(model_loader["rnnMoore_stateDict"])
model = model.to(device)
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

mse_model_list = []
mse_baseline_list = []
mare_model_list = []
mare_baseline_list = []

with torch.no_grad():
    # baseline that only computes an average
    for batch_seq, batch_tar in baseline_loader:
        batch_seq = batch_seq.to(device, non_blocking=True)
        batch_tar = batch_tar.to(device, non_blocking=True)
        local_only = batch_seq[:, -1, neigh_indices]
        local_navg = local_only.mean(dim=1)
        # MSE (sum on GPU)
        # total_baseline_mse += criterion(local_navg, batch_tar.squeeze_())
        sq_err = (local_navg - batch_tar.squeeze(-1)) ** 2
        mse_baseline_list.append(sq_err.cpu())
        # MARE with mask (GPU)
        target = batch_tar.squeeze(-1).abs()
        mask = target > 1e-6
        target = batch_tar.squeeze(-1).abs()
        mask = target > 1e-6

        mare_vals = (local_navg - batch_tar.squeeze(-1)).abs()[mask] / target[mask]
        mare_baseline_list.append(mare_vals.cpu())

        # total_baseline_mare += (
        #     (local_navg - batch_tar.squeeze(-1)).abs()[mask] / target[mask]
        # ).sum()
        
        all_baselines.append(local_navg.cpu())

    # actual model test
    for batch_seq, batch_tar in test_loader:
        batch_seq = batch_seq.to(device, non_blocking=True)
        batch_tar = batch_tar.to(device, non_blocking=True)
        # model (on GPU maintenant)
        outputs = model(batch_seq)  # add feature dim

        # MSE (sum on GPU)
        #total_loss += criterion(outputs, batch_tar)
        #(outputs.squeeze(-1) - batch_tar.squeeze(-1)) ** 2
        sq_err = criterion(outputs, batch_tar)
        mse_model_list.append(sq_err.cpu())

        # MARE with mask (GPU)
        target = batch_tar.squeeze_().abs()
        mask = target > 1e-6
        #total_mare += ((outputs - batch_tar.squeeze()).abs()[mask] / target[mask]).sum()
        mare_vals = (outputs.squeeze(-1) - batch_tar.squeeze(-1)).abs()[mask] / target[mask]
        mare_model_list.append(mare_vals.cpu())
        
        #num_mare += mask.sum()
        all_preds.append(outputs.cpu())


# # move to CPU once
# avg_mse = (total_loss / total_samples).item()
# avg_mare = (total_mare / num_mare).item()
# baseline_mse = (total_baseline_mse / total_samples).item()
# baseline_mare = (total_baseline_mare / num_mare).item()


#######CORRECT THE NEXT###############""
# *** MSE and baseline mse ***
mse_model_all = torch.cat(mse_model_list)
mse_baseline_all = torch.cat(mse_baseline_list)
mare_model_all = torch.cat(mare_model_list)
mare_baseline_all = torch.cat(mare_baseline_list)



# save results to output pt file

my_dict = {}

my_dict["mse_model_all"] = mse_model_all
my_dict["mse_baseline_all"] = mse_baseline_all
my_dict["mare_model_all"] = mare_model_all
my_dict["mare_baseline_all"] = mare_baseline_all

# Concatenate all samples

my_dict["Y_model"] = torch.cat(all_preds)
my_dict["Y_baseline"] = torch.cat(all_baselines)
my_dict["Y_true"] = Y

torch.save(my_dict, output_file)


#SHOW USEFUL STATS
# ===== MSE baseline =====
mse_b = mse_baseline_all.detach().cpu().flatten()

mse_b_q1 = torch.quantile(mse_b, 0.25)
mse_b_med = torch.quantile(mse_b, 0.50)
mse_b_q3 = torch.quantile(mse_b, 0.75)
mse_b_iqr = mse_b_q3 - mse_b_q1
mse_b_low = mse_b_q1 - 1.5 * mse_b_iqr
mse_b_high = mse_b_q3 + 1.5 * mse_b_iqr

print("\nMSE BASELINE")
print("min:", mse_b.min().item())
print("q1:", mse_b_q1.item())
print("median:", mse_b_med.item())
print("q3:", mse_b_q3.item())
print("max:", mse_b.max().item())
print("mean:", mse_b.mean().item())
print("std:", mse_b.std(unbiased=False).item())
print("iqr:", mse_b_iqr.item())
print("lower fence:", mse_b_low.item())
print("upper fence:", mse_b_high.item())


# ===== MSE model =====
mse_m = mse_model_all.detach().cpu().flatten()

mse_m_q1 = torch.quantile(mse_m, 0.25)
mse_m_med = torch.quantile(mse_m, 0.50)
mse_m_q3 = torch.quantile(mse_m, 0.75)
mse_m_iqr = mse_m_q3 - mse_m_q1
mse_m_low = mse_m_q1 - 1.5 * mse_m_iqr
mse_m_high = mse_m_q3 + 1.5 * mse_m_iqr

print("\nMSE MODEL")
print("min:", mse_m.min().item())
print("q1:", mse_m_q1.item())
print("median:", mse_m_med.item())
print("q3:", mse_m_q3.item())
print("max:", mse_m.max().item())
print("mean:", mse_m.mean().item())
print("std:", mse_m.std(unbiased=False).item())
print("iqr:", mse_m_iqr.item())
print("lower fence:", mse_m_low.item())
print("upper fence:", mse_m_high.item())


# ===== MARE baseline =====
mare_b = mare_baseline_all.detach().cpu().flatten()

mare_b_q1 = torch.quantile(mare_b, 0.25)
mare_b_med = torch.quantile(mare_b, 0.50)
mare_b_q3 = torch.quantile(mare_b, 0.75)
mare_b_iqr = mare_b_q3 - mare_b_q1
mare_b_low = mare_b_q1 - 1.5 * mare_b_iqr
mare_b_high = mare_b_q3 + 1.5 * mare_b_iqr

print("\nMARE BASELINE")
print("min:", mare_b.min().item())
print("q1:", mare_b_q1.item())
print("median:", mare_b_med.item())
print("q3:", mare_b_q3.item())
print("max:", mare_b.max().item())
print("mean:", mare_b.mean().item())
print("std:", mare_b.std(unbiased=False).item())
print("iqr:", mare_b_iqr.item())
print("lower fence:", mare_b_low.item())
print("upper fence:", mare_b_high.item())


# ===== MARE model =====
mare_m = mare_model_all.detach().cpu().flatten()

mare_m_q1 = torch.quantile(mare_m, 0.25)
mare_m_med = torch.quantile(mare_m, 0.50)
mare_m_q3 = torch.quantile(mare_m, 0.75)
mare_m_iqr = mare_m_q3 - mare_m_q1
mare_m_low = mare_m_q1 - 1.5 * mare_m_iqr
mare_m_high = mare_m_q3 + 1.5 * mare_m_iqr

print("\nMARE MODEL")
print("min:", mare_m.min().item())
print("q1:", mare_m_q1.item())
print("median:", mare_m_med.item())
print("q3:", mare_m_q3.item())
print("max:", mare_m.max().item())
print("mean:", mare_m.mean().item())
print("std:", mare_m.std(unbiased=False).item())
print("iqr:", mare_m_iqr.item())
print("lower fence:", mare_m_low.item())
print("upper fence:", mare_m_high.item())
