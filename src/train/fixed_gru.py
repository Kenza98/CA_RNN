import time, os
from datetime import datetime
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.utils.plots_model import *
from src.utils.train_loop import train_model
from src.models.gru import *

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

parser = argparse.ArgumentParser()
parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
args = parser.parse_args()

device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}", flush=True)

job_id = os.environ.get("SLURM_JOB_ID")
timestamp = datetime.now().strftime("%m%d_%H%M")
run_id = f"gpu_{job_id}" if device.type == "cuda" else f"cpu_{timestamp}"
data = torch.load(DATA_DIR / "sst_train_set_fixed_47355.pt", map_location="cpu", weights_only=False)
X, Y = data["X"], data["Y"]
train_loader = DataLoader(TensorDataset(X, Y), batch_size=256, shuffle=True, num_workers=4)

input_dim = 9
output_dim = 1

learning_rate = 1e-4
num_epochs = 30
hidden_dim = 28
k = 5

model = GRU(input_dim, hidden_dim, output_dim, num_layers=k)
model_class = model.__class__.__name__
print(f"Model: {model_class} | hidden={hidden_dim} | layers={k} | lr={learning_rate}", flush=True)

model_file = MODEL_DIR / f"{model_class.lower()}_fixed_{run_id}.pt"
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss, grad_history = train_model(
    model, train_loader, optimizer, criterion, num_epochs, device
)

checkpoint = {
    f"{model_class}StateDict": model.state_dict(),
    "model_type": model_class,
    "hidden_dim": hidden_dim,
    "num_layers": k,
    "lr": learning_rate,
}
torch.save(checkpoint, model_file)
print(f"Saved to {model_file}", flush=True)

save_path = OUT_DIR / f"{model_class}_fixed_{run_id}_train_loss.png"
plot_loss_per_epoch(train_loss, save_path)
fp = OUT_DIR / f"{model_class}_fixed_{run_id}_grad.png"
plot_grad_hist(grad_history, fp)
