"""Quick sanity check on generated .pt files."""

import sys
from pathlib import Path

import torch

for path in sorted(Path("data").glob("1_n*.pt")):
    d = torch.load(path, map_location="cpu", weights_only=False)
    X, Y, cfg = d["X"], d["Y"], d["config"]

    print(f"\n=== {path.name} ===")
    print(f"  X {tuple(X.shape)} {X.dtype}   Y {tuple(Y.shape)} {Y.dtype}")
    print(f"  split={cfg['split']}  features={cfg['features']}  seq_length={cfg['seq_length']}")
    print(f"  range: {cfg['start_time']} -> {cfg['end_time']}")
    print(f"  mean={cfg['global_mean']:.4f}  std={cfg['global_std']:.4f}")
    print(f"  X: min={X.min():.3f} max={X.max():.3f} mean={X.mean():.3f}")
    print(f"  Y: min={Y.min():.3f} max={Y.max():.3f} mean={Y.mean():.3f}")
    print(f"  NaNs: X={torch.isnan(X).sum().item()}  Y={torch.isnan(Y).sum().item()}")
    print(f"  size on disk: {path.stat().st_size / 1e6:.1f} MB")
