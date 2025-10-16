import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


load_file = "synthetic_data_interpolation/data/ca02_lr.pt"
data = torch.load(load_file)

# recuperate test data
X_test, Y_test = data["X_TEST"], data["Y_TEST"]
test_dataset = TensorDataset(X_test, Y_test)

batch_size = 32
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

# Reconstruct the model from memory
output_dim = 1
input_dim = 9
model = nn.Sequential(nn.Linear(input_dim, output_dim))
model.load_state_dict(data["model_state_dict"])
criterion = nn.MSELoss()

model.eval()

total_mare = 0
total_loss = 0
num_batches = 0
total_samples = 0
all_preds = []
all_targets = []
epsilon = 1e-8
smape = 0

with torch.no_grad():
    for batch_seq, batch_tar in test_loader:
        batch_seq = batch_seq.view(batch_seq.size(0), -1)
        outputs = model(batch_seq)
        # MSE loss
        loss = criterion(outputs, batch_tar)
        total_loss += loss.item()

        mask = batch_tar != 0
        Aj = torch.abs(batch_tar - outputs)
        Pj = torch.abs(batch_tar)

        # SMAP loss

        Sj = (torch.abs(batch_tar) + torch.abs(outputs)) / 2
        safe_Sj = torch.clamp(Sj, min=epsilon)

        smape += (Aj / safe_Sj).sum()

        # MARE loss
        safe_Pj = torch.maximum(Pj, torch.tensor(epsilon, device=Pj.device))
        mare = (
            Aj / safe_Pj
        ).sum()  # does it do the division for all elements in the batch?
        total_samples += safe_Pj.numel()
        total_mare += mare
        num_batches += 1
        all_preds.append(outputs.cpu())
        all_targets.append(batch_tar.cpu())


for item in model.parameters():
    print(item)

avg_smape = smape / num_batches
average_loss = total_loss / num_batches
avg_mare = total_mare / total_samples

data["mse_test_error"] = average_loss
data["mare_test_error"] = avg_mare

torch.save(data, load_file)

diff = len(test_dataset) - total_samples
print(f"diff = {diff}\n\n")
print(f"Trial completed with average test MSE loss : {average_loss:.4f}")
print(f"Trial completed with average test MARE loss : {avg_mare:.4f}")

writer = SummaryWriter(log_dir="synthetic_data_interpolation/runs/test_eval")

writer.add_scalar("loss/test_mse", average_loss)
writer.add_scalar("loss/test_mare", avg_mare)


# Concatenate all samples
y_pred = torch.cat(all_preds).squeeze().numpy()
y_true = torch.cat(all_targets).squeeze().numpy()

# Create matplotlib figure
fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(y_true, label="Target", marker="o", markersize=3, linestyle="-", alpha=0.7)
ax.plot(y_pred, label="Prediction", marker="x", markersize=3, linestyle="-", alpha=0.7)
ax.set_title("Full Test Set: Predictions vs Targets")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Value")
ax.legend()

# Log to TensorBoard
writer.add_figure("Prediction_vs_Target_FullTest", fig)

writer.close()
