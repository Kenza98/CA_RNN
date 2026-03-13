import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import argparse
import torch.nn as nn
import torch.optim as optim
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

from src.models.gru import GRU
from src.utils.plots_model import plot_grad_hist, plot_loss_per_epoch

# GPU usage if available
parser = argparse.ArgumentParser()
parser.add_argument(
    "--use-gpu", action="store_true", help="Use GPU if CUDA module available"
)

args = parser.parse_args()

# Device selection
device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}", flush=True)

# Define paths
load_file = DATA_DIR / "sst_train_set.pt"
model_file = MODEL_DIR / "gru_moore.pt"

# Load files
data = torch.load(load_file, map_location="cpu")
X, Y = data["X"], data["Y"]
N = X.shape[0]  # nb of samples

train_dataset = TensorDataset(X, Y)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)

# if i already jave a GRU checkpoint, load it on cpu ?why?
if model_file.exists():
    checkpoint = torch.load(model_file, map_location="cpu")

else:
    checkpoint = {}

# I/O dimensions
nb_features = 9
input_dim = nb_features
output_dim = 1


# Hyperparameters
learning_rate = 1e-4
num_epochs = 50
hidden_dim = 7 * 8


# GRU model
model = GRU(input_dim, hidden_dim, output_dim, num_layers=1)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train()

train_loss = []
grad_history = {}

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}\ncomputing ...\n...\n...")
    epoch_start = time.time()
    epoch_loss = 0.0
    for x_batch, y_batch in train_loader:
        # move x and y to GPU
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()  # reinitialize the gradients to avoid exploding (?)
        y_pred = model(x_batch)  # one forward step
        loss = criterion(y_pred, y_batch)
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if name not in grad_history:
                        grad_history[name] = []
                    grad_history[name].append(grad_norm)
        optimizer.step()
        epoch_loss += loss.item()  # .cpu() ???

    epoch_avg_loss = epoch_loss / N
    train_loss.append(epoch_avg_loss)
    epoch_time = time.time() - epoch_start
    # GPU memory usage
    if device == "cuda":
        mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
        mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"GPU Mem: {mem_alloc:.1f}MB/{mem_reserved:.1f}MB\n")
    # Print epoch loss, time
    print(
        f"Loss: {epoch_avg_loss:.4f}\nTime: {epoch_time/60:.2f}min",
        flush=True,
    )

# save model checkpoint
checkpoint["gruMooreStateDict"] = model.state_dict()
model_class = model.__class__.__name__
checkpoint["model_type"] = model_class

torch.save(checkpoint, model_file)
print(f"Model {model_class} successfully saved to {model_file}\n")

# Plots
save_path = OUT_DIR / "gru_train_loss.png"
plot_loss_per_epoch(train_loss, save_path)

fp = OUT_DIR / "gru_grad.png"
plot_grad_hist(grad_history, fp)

print(f"Plots saved to {OUT_DIR}\n", flush=True)
