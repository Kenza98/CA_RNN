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
data_file = "sst_test_set.pt"
model_file = "../data/sst_train_set.pt"
output_file = "testset_results.pt"


data = torch.load(data_file, map_location="cpu")
model_loader = torch.load(model_file, map_location="cpu")
# recuperate test data
X, Y = data["X"], data["Y"]  # these are not normalized and not nan free

test_dataset = TensorDataset(X, Y)

batch_size = 32
output_dim = 1
input_dim = 9
hidden_dim = 7 * 8

test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

model = VanillaRNN(input_dim, hidden_dim, output_dim)
model.load_state_dict(model_loader["model_state_dict"])
model = model.to(device)

criterion = nn.MSELoss()

model.eval()


total_mare = 0
total_loss = 0
num_batches = 0
total_samples = 0
total_samples_marre = 0
avg_neighbors = 0  # for baseline error
all_preds = []
all_targets = []

inf = torch.iinfo(torch.int64).max

with torch.no_grad():
    for batch_seq, batch_tar in test_loader:
        v = batch_seq.shape[0]
        if v != batch_size:
            print(v)
        total_samples += v
        # move batch to gpu
        batch_seq = batch_seq.to(device)
        batch_tar = batch_tar.to(device)

        # model forward pass
        outputs = model(batch_seq)

        # compute the baseline error
        local_navg = torch.mean(batch_seq, dims=(1, 2))
        print(type(local_navg))
        # avg_neighbors += torch.mean( torch.abs(outputs - torch.mean(batch_seq, dim=(1,2))))
        # print(f"Average of neighbors : {avg_neighbors/ batch_size} \n Y_pred mean : {outputs/batch_size}")
        # print(f"Baseline error : {avg_neighbors}")
        # MSE loss
        loss = criterion(outputs, batch_tar)
        total_loss += loss.item()
        avg_loss = total_loss / total_samples
        writer.add_scalar("Loss/train", avg_loss)

        # MARE loss
        mask = batch_tar > 0.05
        total_samples_marre += batch_tar[mask].numel()

        Aj = torch.abs(batch_tar - outputs)

        Pj = torch.abs(batch_tar)
        mare = (Aj[mask] / Pj[mask]).sum()
        total_mare += mare
        num_batches += 1
        all_preds.append(outputs.cpu())
        all_targets.append(batch_tar.cpu())
        break

"""
average_loss = total_loss / num_batches
avg_mare = total_mare / total_samples

#test = torch.load(output_file, map_location="cpu")

my_dict = {}
my_dict["avg_mse"] = average_loss
my_dict["avg_mare"] = avg_mare

torch.save(my_dict, output_file) #\\TODO check this works

diff = (len(test_dataset) - total_samples_marre) #should yield 0 -> neg value
print(f"samples not processed by MARE = {diff}\n\n")
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
hist_fig ,hist_ax = plt.subplots(figsize=(12,6))

hist_ax.hist(y_true, bins=50, alpha=0.7, label="Targets")
hist_ax.hist(y_pred, bins=50, alpha=0.7, label="Predictions")
hist_ax.legend()
hist_ax.set_title("Distribution of Predictions vs Targets")
hist_ax.set_xlabel("Value")
hist_ax.set_ylabel("Frequency")
out_dir = os.path.dirname(__file__)
hist_fig.savefig(os.path.join(out_dir, "hist_pred_vs_target.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "zoom_pred_vs_target.png"), dpi=150, bbox_inches="tight")

writer.add_figure("Histogram_Pred_vs_Target", hist_fig)
plt.close(hist_fig)


# Create matplotlib figure
M = 300
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_true[:M], label="Target", marker="o", markersize=3, linestyle="-", alpha=0.7)
ax.plot(y_pred[:M], label="Prediction", marker="x", markersize=3, linestyle="-", alpha=0.7)
ax.set_title("Zoomed-in: First 300 Test Samples")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Value")
ax.legend()
fig.savefig(os.path.join(out_dir ,"zoom_pred_vs_target.png"), dpi=150, bbox_inches="tight")
writer.add_figure("Prediction_vs_Target_RNNTest", fig)
plt.close(fig)

writer.add_figure("Prediction_vs_Target_RNNTest", fig)
writer.close()
"""
