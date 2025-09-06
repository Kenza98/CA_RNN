# from torch.autocast_mode import autocast
import torch
import time, os
import argparse

# Command-line args
parser = argparse.ArgumentParser()
parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
args = parser.parse_args()

# Device selection
device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}", flush=True)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
import VanillaRNN as vrnn
import matplotlib.pyplot as plt

load_file = "data/sst_train_set.pt"
model_file = "models/rnn_moore.pt"
data = torch.load(load_file, map_location="cpu")
checkpoint = torch.load(model_file, map_location="cpu")

X = data["X"]  # moore neighborhood enriched TS
Y = data["Y"]

train_dataset = TensorDataset(X, Y)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)

nb_features = 9  # (input dim) -> unused
learning_rate = 1e-4
num_epochs = 1
batch_size = 32
output_dim = 1
input_dim = 9
seq_length = 8
hidden_dim = 7 * 8

model = vrnn.VanillaRNN(input_dim, hidden_dim, output_dim)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train()

train_loss = []  # to store loss over epochs

grad_history = {}  # store gradient over epoch to plot smt, the log takes time...

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}\ncomputing ...\n...\n...")
    epoch_start = time.time()
    epoch_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)

        loss = criterion(y_pred, y_batch)

        loss.backward()  # propagate the gradients

        with torch.no_grad():
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if name not in grad_history:
                        grad_history[name] = []
                    grad_history[name].append(grad_norm)

        optimizer.step()
        epoch_loss += loss.item()
    # Average loss
    avg_loss = epoch_loss / len(train_loader)
    train_loss.append(avg_loss)
    print(train_loss[-1], flush=True)
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
        f"Time: {epoch_time:.2f}s | ",
        f" {epoch_time/60:.2f} min",
        flush=True,
    )

# saving model to pt file
checkpoint["rnnMoore_stateDict"] = model.state_dict()
checkpoint["model_type"] = model.__class__.__name__

# Save everything back to the same .pt file
torch.save(checkpoint, model_file)
print(f"Model saved to {model_file}")

out_dir = os.path.dirname(__file__)

# --- 1) Training loss ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_loss, label="Train MSE")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Training Loss Over Epochs")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "train_loss_curve.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# --- 2) Gradient norms per parameter ---
fig, ax = plt.subplots(figsize=(10, 6))
for name, norms in grad_history.items():
    ax.plot(norms, label=name)
# ax.set_yscale("log")  # optional: gradients often benefit from log scale
ax.set_xlabel("Epoch")
ax.set_ylabel("Gradient Norm")
ax.set_title("Gradient Norms per Parameter Across Epochs")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "gradient_norms.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
