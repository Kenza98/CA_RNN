import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


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


load_file = "data/cop_ml_ready.pt"
data = torch.load(load_file)

# recuperate test data
X_test, Y_test = data["X_TEST"], data["Y_TEST"] #these are not normalized and not nan free

X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

nan_mask_X = torch.isnan(X_test).any(dim=(1, 2))  # Check for NaNs in each sample
nan_mask_Y = torch.isnan(Y_test).any(dim=(1))
nan_mask = nan_mask_X | nan_mask_Y

valid_indices = (~nan_mask).nonzero(as_tuple=True)[0] 

X_test_valid = X_test[valid_indices]
Y_test_valid = Y_test[valid_indices]

test_dataset = TensorDataset(X_test_valid, Y_test_valid)

batch_size = 32
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

# Reconstruct the model from memory
output_dim = 1
input_dim = 9
hidden_dim = 7 * 8

model = VanillaRNN(input_dim, hidden_dim, output_dim)
model.load_state_dict(data["model_state_dict"])
criterion = nn.MSELoss()

model.eval()

total_mare = 0
total_loss = 0
num_batches = 0
total_samples = 0
all_preds = []
all_targets = []

inf = torch.iinfo(torch.int64).max

with torch.no_grad():
    for batch_seq, batch_tar in test_loader:
        outputs = model(batch_seq)
        # MSE loss
        loss = criterion(outputs, batch_tar)
        total_loss += loss.item()

        # MARE loss
        mask = batch_tar > 0.05
        Aj = torch.abs(batch_tar - outputs)

        Pj = torch.abs(batch_tar)
        mare = (
            Aj[mask] / Pj[mask]
        ).sum()  # does it do the division for all elements in the batch?

        total_samples += batch_tar[mask].numel()
        total_mare += mare
        num_batches += 1
        all_preds.append(outputs.cpu())
        all_targets.append(batch_tar.cpu())


average_loss = total_loss / num_batches
avg_mare = total_mare / total_samples

data["mse_test_error"] = average_loss
data["mare_test_error"] = avg_mare

torch.save(data, load_file)

diff = (len(test_dataset) - total_samples) / len(test_dataset)
print(f"diff = {diff}\n\n")
print(f"Trial completed with average test MSE loss : {average_loss:.4f}")
print(f"Trial completed with average test MARE loss : {avg_mare:.4f}")

writer = SummaryWriter(log_dir="synthetic_data_interpolation/runs/test_aitems")

writer.add_scalar("loss/test_rnn_mse", average_loss)
writer.add_scalar("loss/test_rnn_mare", avg_mare)


# Concatenate all samples
y_pred = torch.cat(all_preds).squeeze().numpy()
y_true = torch.cat(all_targets).squeeze().numpy()

#checking some stuff about conv

y_std = np.std(y_true)
y_mean = np.mean(y_true)
print(f"Target Mean: {y_mean:.4f}")
print(f"Target Std Dev: {y_std:.4f}")

writer.add_scalar("stats/test_target_std", y_std)
writer.add_scalar("stats/test_target_mean", y_mean)
# to denormalize y_pred_denorm = y_pred * Y_std + Y_mean

#histogram of predictions vs targets
plt.hist(y_true, bins=50, alpha=0.7, label="Targets")
plt.hist(y_pred, bins=50, alpha=0.7, label="Predictions")
plt.legend()
plt.title("Distribution of Predictions vs Targets")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
# Create matplotlib figure
M = 300
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_true[:M], label="Target", marker="o", markersize=3, linestyle="-", alpha=0.7)
ax.plot(y_pred[:M], label="Prediction", marker="x", markersize=3, linestyle="-", alpha=0.7)
ax.set_title("Zoomed-in: First 300 Test Samples")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Value")
ax.legend()

plt.show()
# Log to TensorBoard
writer.add_figure("Prediction_vs_Target_RNNTest", fig)

writer.close()
