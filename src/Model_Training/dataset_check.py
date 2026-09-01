import os

DATASET_ROOT = "/kaggle/input/datasets/diyer22/retail-product-checkout-dataset"  # This path is used when this file is run on hte Kaggle resources

print("=" * 70)
print("🔍 RPC DATASET CHECK")
print("=" * 70)

if not os.path.exists(DATASET_ROOT):
    print("❌ Dataset path does not exist!")
else:
    print("✅ Dataset found!")
    print("\nContents:\n")

    for item in sorted(os.listdir(DATASET_ROOT)):
        path = os.path.join(DATASET_ROOT, item)

        if os.path.isdir(path):
            try:
                count = len(os.listdir(path))
            except:
                count = "?"

            print(f"📁 {item:<35} {count} files")
        else:
            size = os.path.getsize(path) / (1024**2)
            print(f"📄 {item:<35} {size:.2f} MB")