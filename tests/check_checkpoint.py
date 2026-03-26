import torch

checkpoint = torch.load("checkpoints/last.pt", map_location="cpu")
print("Checkpoint keys:", checkpoint.keys())
print("\nCheckpoint structure:")
for key, value in checkpoint.items():
    if isinstance(value, dict):
        print(f"\n{key} (dict):")
        for subkey in list(value.keys())[:10]:
            print(f"  {subkey}: {value[subkey].shape if hasattr(value[subkey], 'shape') else type(value[subkey])}")
    else:
        print(f"{key}: {value}")
