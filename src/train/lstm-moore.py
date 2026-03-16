import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.utils.plots_model import *

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"   #to get training data
MODEL_DIR = PROJECT_ROOT / "models"  #to save the trained model
OUT_DIR = PROJECT_ROOT / "outputs"   #to save the plots post-training

#lstm and rnn trained under identical conditions
from src.models.lstm import *  #imports the model class

# GPU usage if available
parser = argparse.ArgumentParser()
parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
args = parser.parse_args()

# Device selection
device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}", flush=True)

# Define paths
load_file = DATA_DIR / "sst_test_set.pt"
model_file = MODEL_DIR / "lstm_moore.pt"

# Load files
data = torch.load(load_file, map_location="cpu")

# If I already have an LSTM checkpoint, load it
#conda → PyTorch → CUDA → GPU
if model_file.exists():
    checkpoint = torch.load(model_file, map_location="cpu")
    print(type(checkpoint))
else:
    checkpoint = {}

X = data["X"]   # moore neighborhood enriched TS already in sequence format
Y = data["Y"]

train_dataset = TensorDataset(X, Y)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

# I/O dimensions
nb_features = 9
input_dim = nb_features
output_dim = 1

# Hyperparameters
learning_rate = 1e-4
num_epochs = 100

seq_length = 8
hidden_dim = 7 * 8
num_layers = 1

# Model
model = LSTM(input_dim, hidden_dim, output_dim, num_layers=num_layers)
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
        #moving the x,y batches to GPU now
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)

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
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(avg_loss, flush=True)
    train_loss.append(avg_loss)

    # GPU memory usage (MB)
    if device.type == "cuda":
        mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
        mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
    else:
        mem_alloc = mem_reserved = 0

    epoch_time = time.time() - epoch_start

    print(
        f"Loss: {avg_loss:.4f} | "
        f"GPU Mem: {mem_alloc:.1f}MB/{mem_reserved:.1f}MB | "
        f"Time: {epoch_time:.2f}s | "
        f"{epoch_time/60:.2f} min",
        flush=True,
    )

# Save model
checkpoint["lstmMoore_stateDict"] = model.state_dict()
checkpoint["model_type"] = model.__class__.__name__

torch.save(checkpoint, model_file)
print(f"Model saved to {model_file}")

# Plots
save_path = OUT_DIR / "lstm_train_loss.png"
plot_loss_per_epoch(train_loss, save_path)

fp = OUT_DIR / "lstm_grad.png"
plot_grad_hist(grad_history, fp)