import torch
import re
import datetime
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "ca_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
from src.models.gru import GRU

#helper function to print NaN percentage
def nan_ratio(tensor, name="tensor"):
    total = tensor.numel()
    nans = torch.isnan(tensor).sum().item()
    print(f"{name}: {nans}/{total} NaN ({100 * nans / total:.1f}%)\n")
    print(f"Number of present pixels : {total - nans}\n")

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

#print(f"Model: {model_file.name}")
#print(f"Created: {datetime.datetime.fromtimestamp(mtime)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict = torch.load(model_file, map_location=device)

print(f"Running the model on {device}\n")

#load model

# load the model checkpoint on disk (cpu)
output_dim = 1
input_dim = 9
hidden_dim = 56

model = GRU(input_dim=9, hidden_dim=56, output_dim=1)

model.load_state_dict(model_dict["GRUStateDict"])
model_class = model.__class__.__name__
model.eval()

N = sst_tensor.shape[0]
def ground_truth_frames(sst_tensor, sequence_length=4):
    """
    This function yields the next grid at time t
    Useful for computing errors
    """
    num_steps = N - sequence_length
    for t in range(sequence_length, num_steps):
        yield sst_tensor[t]  #whole grid for time t (24, 97)

seq_length=4
steps = N - seq_length
def predictions(sst_tensor, model , num_steps=steps):
    window = sst_tensor[:seq_length] # ---seed
    for t in range(seq_length, seq_length + num_steps):
        #first : get the 3*3 neighborhood with unfold :
        neigh_lat = window.unfold(1,3,1)# unfold along dim 1=lats (4,64, 88, 3)
        neigh_lat_lon = neigh_lat.unfold(2,3,1)  # unfold along dim 2=lons(4, 64, 88, 3, 3)
        
        # next : materialise this view into a freshly allocated, sequentially ordered block of memory :
        neigh = neigh_lat_lon.contiguous()

        #now let's get the correct view :
        X = neigh.view(seq_length, -1, 9)
        #last : permute to have batch_first for model :
        X = X.permute(1, 0, 2)   # (5632, 4, 9)
        
        #nan_ratio(X, "X")

        valid_mask = ~torch.isnan(X).any(dim=-1).any(dim=-1)  # True where no NaN in any of the 9 neighbors across all 4 timesteps → (5632,)
        assert valid_mask.any(), "No valid pixels to evolve!"
        # 1. start with full ground truth
        next_full = sst_tensor[t].clone()  

        # 2. run model on valid pixels only            
        pred_flat = model(X[valid_mask])          #    (N_valid, 1)

        # 3. extract interior as a flat clone
        next_interior = next_full[1:-1, 1:-1]
        flat_next_interior = next_interior.reshape(-1)
        flat = flat_next_interior.clone()  # (5632,) fresh memory

        # 4. write predictions into flat at valid positions
        flat_temp_count = torch.isnan(flat).sum()
        #print(f"before: {flat_temp_count}")
        flat[valid_mask] = pred_flat.squeeze(1).detach()
        flat_temp_count = torch.isnan(flat).sum()
        #print(f"after: {flat_temp_count}")
        
        # 5. write flat back into next_full interior
        next_full[1:-1, 1:-1] = flat.view(next_full.shape[0]-2, next_full.shape[1]-2)
        
        
        yield next_full
        window = torch.cat([window[1:], next_full.unsqueeze(0)])

def neighborhood_average_predictions(sst_tensor, num_steps=steps):
    window = sst_tensor[:seq_length]
    for t in range(seq_length, seq_length + num_steps):
        neigh_lat = window.unfold(1, 3, 1)
        neigh_lat_lon = neigh_lat.unfold(2, 3, 1)          # (4, 64, 88, 3, 3)
        neigh = neigh_lat_lon.contiguous()
        X = neigh.view(seq_length, -1, 9).permute(1, 0, 2) # (5632, 4, 9)
        
        valid_mask = ~torch.isnan(X).any(dim=-1).any(dim=-1)
        
        next_full = sst_tensor[t].clone()
        
        if valid_mask.any():
            # average over the 9 neighbors at the last timestep only
            flat = next_full[1:-1, 1:-1].reshape(-1).clone()
            flat[valid_mask] = X[valid_mask, -1, :].mean(dim=-1).detach()
            next_full[1:-1, 1:-1] = flat.view(next_full.shape[0]-2, next_full.shape[1]-2)
        
        yield next_full
        window = torch.cat([window[1:], next_full.unsqueeze(0)])

"""
gen = predictions(sst_tensor, model)
first_pred = next(gen)
print(first_pred.shape)
nan_ratio(first_pred, "first prediction")
print(f"non-NaN values: {(~torch.isnan(first_pred)).sum().item()}")
print(f"sample values: {first_pred[~torch.isnan(first_pred)][:5]}")

exit(0)


prev = None
for i, pred in enumerate(predictions(sst_tensor, model)):
    if prev is not None:
        valid = ~torch.isnan(pred) & ~torch.isnan(prev)
        changed = ((pred[valid] - prev[valid]).abs() > 0.001).sum().item()
        print(f"step {i}: {changed}/{valid.sum().item()} pixels changed by >0.001°C")
    prev = pred.clone()

"""
gru_errors = []
baseline_errors = []

for pred, true in zip(predictions(sst_tensor, model, num_steps=steps), ground_truth_frames(sst_tensor)):
    interior_pred = pred[1:-1, 1:-1].contiguous().view(-1)
    interior_true = true[1:-1, 1:-1].contiguous().view(-1)
    valid = ~torch.isnan(interior_pred) & ~torch.isnan(interior_true)
    mae = torch.abs(interior_pred[valid] - interior_true[valid]).mean().item()
    gru_errors.append(mae)

for pred, true in zip(neighborhood_average_predictions(sst_tensor, num_steps=steps), ground_truth_frames(sst_tensor)):
    interior_pred = pred[1:-1, 1:-1].contiguous().view(-1)
    interior_true = true[1:-1, 1:-1].contiguous().view(-1)
    valid = ~torch.isnan(interior_pred) & ~torch.isnan(interior_true)
    mae = torch.abs(interior_pred[valid] - interior_true[valid]).mean().item()
    baseline_errors.append(mae)

results = {"gru": gru_errors, "baseline": baseline_errors}
output_path = OUT_DIR / "ca_errors.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved to {output_path}")
