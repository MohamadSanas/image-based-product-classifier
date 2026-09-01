import torch

print("=" * 70)
print("🚀 GPU CHECK")
print("=" * 70)

print("PyTorch :", torch.__version__)
print("CUDA available :", torch.cuda.is_available())
print("GPU count :", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(
        f"GPU {i}: "
        f"{torch.cuda.get_device_name(i)}"
    )

print("=" * 70)