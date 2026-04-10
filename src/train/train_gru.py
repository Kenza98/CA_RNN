import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import argparse
import torch.nn as nn
import torch.optim as optim
import os
from src.utils.train_loop import train_model
from datetime import datetime

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

#create name for pt file
job_id = os.environ.get("SLURM_JOB_ID") #for gpu name
timestamp = datetime.now().strftime("%m%d_%H%M") #for cpu name

if device.type == "cpu":
    run_id = f"cpu_{timestamp}"
else:
    run_id = f"gpu_{job_id}"
# Define paths

# Load data
load_file = DATA_DIR / "sst_train_set.pt"
data = torch.load(load_file, map_location="cpu", weights_only=False)
X, Y = data["X"], data["Y"]
N = X.shape[0]  # nb of samples
print("Size of sample : ", N)
train_dataset = TensorDataset(X, Y)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

# I/O dimensions
nb_features = 9
input_dim = nb_features
output_dim = 1

# Hyperparameters
learning_rate = 1e-4
num_epochs = 30
hidden_dim = 56
k = 4
# GRU model
model = GRU(input_dim, hidden_dim, output_dim, num_layers=k)
model_class = model.__class__.__name__
print(f"Model Class name is : {model_class}\n")

if device.type == "cpu":
    run_is = f"cpu_{timestamp}"
else:
    run_id = f"gpu_{job_id}"
    
model_file = MODEL_DIR / f"{model_class.lower()}_{run_id}.pt"

checkpoint = {} #start fresh each time train script is ran

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss, grad_history = train_model(model, train_loader, optimizer, criterion, num_epochs, device)

# save model checkpoint
checkpoint[f"{model_class}StateDict"] = model.state_dict()
checkpoint["model_type"] = model_class

torch.save(checkpoint, model_file)
print(f"Model {model_class} successfully saved to {model_file}\n")

# Plots
save_path = OUT_DIR / f"{model_class.lower()}_{run_id}_train_loss.png"
plot_loss_per_epoch(train_loss, save_path)

fp = OUT_DIR / f"{model_class.lower()}_{run_id}_grad.png"
plot_grad_hist(grad_history, fp)
