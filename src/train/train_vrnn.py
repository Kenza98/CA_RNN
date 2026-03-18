import time, os
import argparse
from datetime import datetime
from pathlib import Path
from src.utils.plots_model import *
from src.utils.train_loop import train_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.VanillaRNN import *



PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

# GPU usage if available
parser = argparse.ArgumentParser()
parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
args = parser.parse_args()

# Device selection
device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}", flush=True)


#create name for pt file
job_id = os.environ.get("SLURM_JOB_ID") #for gpu name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") #for cpu name

if device.type == "cpu":
    run_id = f"cpu_{timestamp}"
else:
    run_id = f"gpu_{job_id}"

# create the train data loader
load_file = DATA_DIR / "sst_train_set.pt" #data file
data = torch.load(load_file, map_location="cpu")
X = data["X"]  # moore neighborhood enriched TS already in sequence format
Y = data["Y"]
train_dataset = TensorDataset(X, Y)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

#I/O DIMs
nb_features = 9  # (input dim)
input_dim = nb_features
output_dim = 1

#HYP PARAMETERS
learning_rate = 1e-4
num_epochs = 10
batch_size = 32
seq_length = 8
hidden_dim = 7 * 8


#MODEL
model = VanillaRNN(input_dim, hidden_dim, output_dim)
model_class = model.__class__.__name__  # move this up here
print(f"Model Class name is : {model_class}\n")
if device.type == "cpu":
    run_is = f"cpu_{timestamp}"
else:
    run_id = f"gpu_{job_id}"

model_file = MODEL_DIR / f"{model_class.lower()}_{run_id}.pt"
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss, grad_history = train_model(
    model,
    train_loader,
    optimizer,
    criterion,
    num_epochs,
    device
)

#checkpoint saving
checkpoint = {}
checkpoint[f"{model_class}StateDict"] = model.state_dict()
checkpoint["model_type"] = model_class

# Save everything back to the same .pt file
torch.save(checkpoint, model_file)
print(f"{model_class} model successfully saved to {model_file}")


# Plots
save_path = OUT_DIR / f"{model_class.lower()}_{run_id}_train_loss.png"
plot_loss_per_epoch(train_loss, save_path)

fp = OUT_DIR / f"{model_class.lower()}_{run_id}_grad.png"
plot_grad_hist(grad_history, fp)