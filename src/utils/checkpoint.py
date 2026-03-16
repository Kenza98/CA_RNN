import torch
from pathlib import Path
import sys
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

def check_checkpoint(name: str) -> None:
    if not name.endswith(".pt"):
        name+=".pt"
    
    path = MODEL_DIR / name

    if not path.exists():
        print(f"Aucun checkpoint trouvé : {path}")
        return
    
    stat = path.stat()
    size_kb = stat.st_size / 1024
    
    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
 
    print(f"Checkpoint : {path}")
    print(f"  Date/heure : {modified}")
    print(f"  Taille     : {stat.st_size:,} octets ({size_kb:.1f} Ko)")
 
    checkpoint = torch.load(path, map_location="cpu")
 
    print(f"  Contenu    :")
    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            if hasattr(value, "shape"):
                print(f"    {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f"    {key}: dict avec {len(value)} clés")
            else:
                print(f"    {key}: {type(value).__name__} = {value}")
    else:
        print(f"    (pas un dict) type={type(checkpoint).__name__}")
 


if __name__ == "__main__":
    if len(sys.argv) < 2:
        name = input("Nom du modèle : ").strip()
    else:
        name = sys.argv[1]
 
    check_checkpoint(name)
 
