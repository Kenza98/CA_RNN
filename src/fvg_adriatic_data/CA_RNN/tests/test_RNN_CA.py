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
X = data["X"][:,:,4] # select only the central cell.
Y = data["Y"]  # not normalized nut nan free.
total_samples = X.shape[0]

test_dataset = TensorDataset(X, Y)

batch_size = 32
output_dim = 1
input_dim = 1
hidden_dim = 7 * 8

test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

model = VanillaRNN(input_dim, hidden_dim, output_dim)
model.load_state_dict(model_loader["model_state_dict"])
model = model.to(device)

criterion = nn.MSELoss(reduction="sum") 
model.eval()

total_mare = 0
total_loss = 0

total_baseline_mse = 0  # for baseline mse error
total_baseline_mare = 0  # for baseline mare error
all_preds = []
all_baselines = []

inf = torch.iinfo(torch.int64).max
neigh_indices = [i for i in range(9) if i != 4]

with torch.no_grad():
    for batch_seq, batch_tar in test_loader:
        # move batch to gpu
        batch_seq = batch_seq.to(device)
        batch_tar = batch_tar.to(device)

        # model forward pass
        outputs = model(batch_seq)
        outputs.squeeze_()  # remove the last dimension if it's 1
        #outputs = outputs.to(device) ---> redundant, already on GPU
        local_only = batch_seq[:, -1, neigh_indices] 
        #print(local_only.shape)
        local_navg = torch.mean(local_only, dim=1)

        local_navg = local_navg.to(device)  # move to device

        # compute the baseline errors
        # *** MSE ***
        #mse_baseline = ((batch_tar - local_navg) ** 2).sum()  # mse for this batch
        local_navg.squeeze_()
        batch_tar.squeeze_()
        
        total_baseline_mse += criterion(local_navg, batch_tar).item()
        
        # ****MARE ***
        mare_baseline = torch.abs(
            (batch_tar - local_navg) / batch_tar
        )  # for this batch
        total_baseline_mare += mare_baseline.sum()  # accumulate

        # MSE loss
        total_loss += criterion(outputs, batch_tar).item()

        # MARE loss
        mare = torch.abs(batch_tar-outputs)/batch_tar
        total_mare += mare.sum() #sum of MARE of the current batch
        all_preds.append(outputs.cpu())
        all_baselines.append(local_navg.cpu())


# *** MSE and baseline mse ***
final_baseline_mse = total_baseline_mse / total_samples
average_loss = total_loss / total_samples
print(total_samples)

print(f"average baseline MSE err : {final_baseline_mse}")
print(f"average MSE err : {average_loss}")

# *** MARE and baseline MARE ***
avg_mare = total_mare / total_samples
final_baseline_mare = total_baseline_mare / total_samples

print(f"Trial completed with average test MARE loss : {avg_mare:.4f}")
print(f"average baseline MARE err : {final_baseline_mare}")


# save results to output pt file

my_dict = {}
my_dict["avg_mse"] = average_loss
my_dict["avg_mare"] = avg_mare
my_dict["baseline_mse"] = final_baseline_mse
my_dict["baseline_mare"] = final_baseline_mare
my_dict["num_samples"] = total_samples


writer = SummaryWriter(log_dir="runs/test_aitems")
writer.add_scalar("loss/test_rnn_mse", average_loss)
writer.add_scalar("loss/test_rnn_mare", avg_mare)
writer.add_scalar("loss/test_rnn_baseline_mse", final_baseline_mse)
writer.add_scalar("loss/test_rnn_baseline_mare", final_baseline_mare)


# Concatenate all samples
Y_model = torch.cat(all_preds)
Y_baseline = torch.cat(all_baselines)

my_dict["Y_model"] = Y_model
my_dict["Y_baseline"] = Y_baseline
my_dict["Y_true"] = Y

torch.save(my_dict, output_file)


writer.close()
