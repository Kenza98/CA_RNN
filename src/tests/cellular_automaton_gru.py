import torch
import re
import datetime
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "ca_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
from src.models.gru import GRU

#load the data
data_file = DATA_DIR / "ca_data.pt"
sst_tensor = torch.load(data_file)["full_data"]

#print(sst_tensor.shape)
#exit(0)

def find_result_file(pattern_str):
    pattern = re.compile(pattern_str)
    files = [f for f in MODEL_DIR.iterdir() if pattern.match(f.name)]
    if not files:
        raise FileNotFoundError(f"No file matching '{pattern_str}' in {OUT_DIR}")
    return sorted(files, key=lambda f: f.stat().st_mtime)[-1]

model_file =find_result_file(r".*gru.*\.pt$") # GRU file
mtime = model_file.stat().st_mtime
print(f"Model: {model_file.name}")
print(f"Created: {datetime.datetime.fromtimestamp(mtime)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict = torch.load(model_file, map_location=device)

print(f"Running the model on {device}\n")
print(model_dict.keys())

#load model

# load the model checkpoint on disk (cpu)
output_dim = 1
input_dim = 9
hidden_dim = 56

model = GRU(input_dim=9, hidden_dim=56, output_dim=1, num_layers=4)

model.load_state_dict(model_dict["GRUStateDict"])
model_class = model.__class__.__name__
model.eval()

def ground_truth_frames(sst_tensor, start=4, end=15):
    #get the truth pixel
    max_days = sst_tensor.shape[0]
    for t in range(start, end):
        yield sst_tensor[t]  #whole grid for time t (24, 97)


def predictions(sst_tensor, model, seq_length=4, num_steps=14):
    #num_steps = sst_tensor.shape[0]
    window = sst_tensor[:seq_length]  # (4, 24, 97)
    for t in range(seq_length, ):
        neigh = window.unfold(1, 3, 1).unfold(2, 3, 1)                    # (4, 22, 95, 3, 3)
        X = neigh.contiguous().view(seq_length, -1, 9).permute(1, 0, 2)   # (2090, 4, 9)

        valid_mask = ~torch.isnan(X).any(dim=-1).any(dim=-1)               # (2090,)
        
        next_full = sst_tensor[t].clone()                                  # (24, 97) ground truth
        
        if valid_mask.any():
            pred_flat = model(X[valid_mask])                               # (N_valid, 1)
            next_full[1:-1, 1:-1][valid_mask] = pred_flat.squeeze(1)      # overwrite valid pixels
        
        yield next_full
        window = torch.cat([window[1:], next_full.unsqueeze(0)])        



for pred, true in zip(predictions(sst_tensor, model), ground_truth_frames(sst_tensor)):
    error = torch.abs(pred - true)