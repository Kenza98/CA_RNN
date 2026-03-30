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

#load model

# load the model checkpoint on disk (cpu)
output_dim = 1
input_dim = 9
hidden_dim = 56

model = GRU(input_dim=9, hidden_dim=56, output_dim=1, num_layers=4)

model.load_state_dict(model_dict["GRUStateDict"])
model_class = model.__class__.__name__
model.eval()
#load the data
data_file = DATA_DIR / "ca_data.pt"
sst_tensor = torch.load(data_file)["full_data"]

print(sst_tensor.shape)

exit(0)

def ground_truth_frames(sst_tensor, start=4):
    for t in range(start, sst_tensor.shape[0]):
        #generate neighborhood
        # return the whole sequence
        yield sst_tensor[t]  # (24, 97)

def predictions(sst_tensor, model, seq_length=4):
    window = sst_tensor[:seq_length]  # (4, 24, 97) — initial seed
    for _ in range(sst_tensor.shape[0] - seq_length):
        next_step = model(window)  # (24, 97)
        yield next_step
        window = torch.cat([window[1:], next_step.unsqueeze(0)])  # slide



for pred, true in zip(predictions(sst_tensor, model), ground_truth_frames(sst_tensor)):
    error = torch.abs(pred - true)