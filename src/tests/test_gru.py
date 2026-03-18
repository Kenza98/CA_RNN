import torch
from torch.utils.data import TensorDataset, DataLoader
import sys
from pathlib import Path
from src.utils.evaluate import *
# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
#sys.path.append(str(PROJECT_ROOT))
#print(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

file_name = "test_gru_result.pt"
output_file = OUT_DIR / file_name

### CHECK IF TESTS EXIST ###
if output_file.exists():
    print(f"Tests already ran and saved to {file_name}.\nExiting.\n")
    exit(0)

from src.models.gru import GRU
from src.utils.evaluate import evaluate_model

# check if file was ran with --use-gpu + if cuda devise available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA version: {torch.version.cuda}", flush=True)

# LOAD DATA CREATE LOADER
test_data_file = DATA_DIR / "sst_test_set.pt"
data = torch.load(test_data_file, map_location="cpu")
X, Y = data["X"], data["Y"]
test_dataset = TensorDataset(X, Y)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# LOAD MODEL FILE, GET MODELS TEST RESULTS
#model_file = MODEL_DIR / "gru_moore.pt"
model_file = MODEL_DIR / "gru_gpu_43125.pt"


# load the model checkpoint on disk (cpu)
output_dim = 1
input_dim = 9
hidden_dim = 7 * 8
checkpoint = torch.load(model_file, map_location="cpu")
model = GRU(input_dim, hidden_dim, output_dim)

model.load_state_dict(checkpoint["GRUStateDict"])
model = model.to(device)

# ***GETTING RESULTS WITH HELPER FCT***
my_dict = evaluate_model(model, test_loader, device)

# ***SAVING RESULTS TO PT FILE***
torch.save(my_dict, output_file)
print(f"Saved {model.__class__.__name__} evaluation\n")

# ***PRINT SOME QUANTILES***
mse = my_dict["mse"]
mae= my_dict["mae"]
ae_tensor = my_dict["absolute_error"]
se_tensor = my_dict["squared_error"]
quick_test_sanity(mse, mae, ae_tensor, se_tensor)

'''
print(checkpoint.keys())
exit(0)
'''
