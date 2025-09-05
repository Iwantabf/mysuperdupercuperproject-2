import os
import numpy as np

labels = np.load("labels.npy")  

base_path = "dataset"
os.makedirs(base_path, exist_ok=True)

for label in labels:
    label_path = os.path.join(base_path, label)
    os.makedirs(label_path, exist_ok=True)
    print(f"✅ Created folder: {label_path}")
