import torch
import random

def inspect_pt_file(pt_path, num_samples=5):
    data = torch.load(pt_path)
    X = data["X"]
    Y = data["Y"]

    print(f"Loaded {pt_path}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    total_samples = X.shape[0]
    indices = random.sample(range(total_samples), min(num_samples, total_samples))

    for idx in indices:
        print(f"\nSample {idx}:")
        print("X (sequence):")
        print(X[idx])
        print("Y (target):", Y[idx].item())

if __name__ == "__main__":
    for name in ["sst_test_set.pt",  "training_set.pt"] :
        inspect_pt_file(name, num_samples=5)