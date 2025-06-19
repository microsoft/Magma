import torch

print(f"Pytorch version: {torch.version}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version Pytorch is using: {torch.version.cuda}")