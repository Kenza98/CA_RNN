import torch
from torch.utils.data import TensorDataset, DataLoader
import os, sys
from pathlib import Path
from datetime import datetime
import re
import argparse
# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
#imports that need project root
from src.models.VanillaRNN import VanillaRNN
from src.utils.evaluate import evaluate_model, quick_test_sanity
####
#continue defining more paths...
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs" / "shuffle_True_ep100"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#get the model file
parser = argparse.ArgumentParser()
parser.add_argument("--model-file", type=str, default=None,
                    help="Path to .pt file, e.g. models/vrnn_gpu_44121.pt. If not provided, uses most recent vrnn_gpu*.pt in MODEL_DIR.")
args = parser.parse_args()

if args.model_file is None:
    # fall back to most recent vrnn_gpu*.pt
    pattern = re.compile(r"^vrnn_gpu.*\.pt$")
    pt_files = sorted(
        [f for f in MODEL_DIR.iterdir() if pattern.match(f.name)],
        key=lambda f: f.stat().st_mtime
    )
    if not pt_files:
        raise FileNotFoundError(f"No matching vrnn_gpu*.pt files found in {MODEL_DIR}")
    model_file = pt_files[-1]
else:
    model_file = PROJECT_ROOT / args.model_file


# check if cuda devise available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA version: {torch.version.cuda}", flush=True)


# LOAD DATA CREATE LOADER
test_data_file = DATA_DIR / "sst_test_set.pt"
data = torch.load(test_data_file, map_location=device)
X = data["X"]
Y = data["Y"]
total_samples = X.shape[0]

print(f"X has shape N= {X.shape}")
print(f"Y has shape N= {Y.shape}")
batch_size = 256

test_dataset = TensorDataset(X, Y)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)


# LOAD PARAMETERS RECREATE MODEL

model_loader = torch.load(model_file, map_location=device)
print("Top-level keys in model file:", model_loader.keys(), flush=True)
timestamp = os.path.getmtime(model_file)
print("Last model update date: ", datetime.fromtimestamp(timestamp))

output_dim = 1
input_dim = 9
hidden_dim = 7 * 8

model = VanillaRNN(input_dim, hidden_dim, output_dim)
model.load_state_dict(model_loader["VanillaRNNStateDict"])

# GET RESULTS WITH HELPER FCT
result_dico = evaluate_model(model, test_loader, device)

# SAVE TO PT FILE
output_file = OUT_DIR / "test_vrnn_result.pt"

torch.save(result_dico, output_file)
print(f"Saved {model.__class__.__name__} evaluation\n")


# PRINT SOME QUANTILES
#\\TODO check that the following is good:
mse = result_dico["mse"]
mae= result_dico["mae"]
ae_tensor = result_dico["absolute_error"]
se_tensor = result_dico["squared_error"]
quick_test_sanity(mse, mae, ae_tensor, se_tensor)
