import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os

# Create a unique log directory for this run
log_dir = os.path.join("runs", f"rnnca_job_{os.environ.get('SLURM_JOB_ID', 'local')}")
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard log dir: {log_dir}", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA version: {torch.version.cuda}", flush=True)


class VanillaRNN(nn.Module):
    # This method initializes the layers of the model:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # super() alls the parent class constructor to set up the nn.Module infrastructure :
        super().__init__()

        self.rnn = nn.RNN(input_dim, hidden_dim, 4, batch_first=True)
        # my module has an RNN "brain"
        self.fc = nn.Linear(hidden_dim, output_dim)
        # my NN module has a fully connected / dense layer
        # -->i need this in the forward method to process the hidden state output from the RNN

    """
    forward() function is necessary in any nn.Module subclass.
    It specifies how the input x moves through the layers of the model.
    """

    def forward(self, x):
        out, _ = self.rnn(
            x
        )  # out contains the hidden states for each time step in the sequence
        # _ cuz i didn't need the final hidden state. I use the one from last time step at each run
        out = self.fc(out[:, -1, :])  # Use the last hidden state
        return out

# paths
data_file = "./sst_test_set.pt"
model_file = "../data/sst_train_set.pt"
output_file = "./testset_results.pt"
data = torch.load(data_file, map_location="cpu")
print(data.keys())
model_loader = torch.load(model_file, map_location="cpu")
# recuperate test data

Xn = data["X"]
X = Xn[:, :, 4]  # select only the central cell.

Y = data["Y"]  # not normalized nut nan free.
total_samples = X.shape[0]

test_dataset = TensorDataset(X, Y)
baseline_dataset = TensorDataset(Xn, Y)

batch_size = 32
output_dim = 1
input_dim = 1
hidden_dim = 7 * 8

test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
baseline_loader = DataLoader(baseline_dataset, batch_size, shuffle=True)

model = VanillaRNN(input_dim, hidden_dim, output_dim)
model.load_state_dict(model_loader["model_state_dict"])
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
criterion = nn.MSELoss(reduction="sum")
model.eval()

total_samples = X.shape[0]
model.eval()
criterion = nn.MSELoss(reduction="sum")  # sum on GPU; divide later

neigh_indices = [i for i in range(9) if i != 4]
total_loss = torch.tensor(0.0, device=device)
total_mare = torch.tensor(0.0, device=device)
total_baseline_mse = torch.tensor(0.0, device=device)
total_baseline_mare = torch.tensor(0.0, device=device)
num_mare = torch.tensor(0.0, device=device)  # count for MARE denom

all_preds = []
all_baselines = []

with torch.no_grad():
    for batch_seq, batch_tar in test_loader:
        batch_seq = batch_seq.to(device, non_blocking=True)
        batch_tar = batch_tar.to(device, non_blocking=True)
        # model (on GPU maintenant)
        outputs = model(batch_seq.unsqueeze(-1))
        #print("target", batch_tar.shape, "output", outputs.shape)
        #exit()
        # MSE (sum on GPU)
        total_loss += criterion(outputs, batch_tar)

        # MARE with mask (GPU)
        target = batch_tar.abs()
        mask = target > 1e-6
        total_mare += (
            (outputs - batch_tar).abs()[mask] / target[mask]
        ).sum()
        num_mare += mask.sum()
        all_preds.append(outputs.cpu())

    for batch_seq, batch_tar in baseline_loader:
        batch_seq = batch_seq.to(device, non_blocking=True)
        batch_tar = batch_tar.to(device, non_blocking=True)
        local_only = batch_seq[:, neigh_indices]
        local_navg = local_only.mean(dim=1)
        # MSE (sum on GPU)
        total_baseline_mse += criterion(local_navg, batch_tar.squeeze(-1))
        # MARE with mask (GPU)
        target = batch_tar.squeeze(-1).abs()
        mask = target > 1e-6
        total_baseline_mare += (
            (local_navg - batch_tar.squeeze(-1)).abs()[mask] / target[mask]
        ).sum()
        all_baselines.append(local_navg.cpu())


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


writer = SummaryWriter(log_dir="runs/test_aitems")
writer.add_scalar("loss/test_rnn_mse", avg_mse)
writer.add_scalar("loss/test_rnn_mare", avg_mare)
writer.add_scalar("loss/test_rnn_baseline_mse", baseline_mse)
writer.add_scalar("loss/test_rnn_baseline_mare", baseline_mare)


# Concatenate all samples

my_dict["Y_model"] = torch.cat(all_preds)
my_dict["Y_baseline"] = torch.cat(all_baselines)
my_dict["Y_true"] = Y

torch.save(my_dict, output_file)
writer.close()
