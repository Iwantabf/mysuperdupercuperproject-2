# dataset.py
import numpy as np
import os

data_dir = "dataset"
sequences = []
labels = []

label_map = np.load("label_classes.npy", allow_pickle=True)
label_map = {label: index for index, label in enumerate(label_map)}

for label in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label)
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith(".npy"):
            sequence = np.load(os.path.join(label_path, file))
            if sequence.shape == (30, 126):  # ตรวจความถูกต้องของ shape
                sequences.append(sequence)
                labels.append(label_map[label])

X = np.array(sequences)
y = np.array(labels)

np.save("X.npy", X)
np.save("y.npy", y)

print("✅ สร้าง X.npy และ y.npy เรียบร้อยแล้วจ้า")
print("จำนวน sequences ที่ได้:", len(sequences))
print("จำนวน labels ที่ได้:", len(labels))
print("Shape ของ sequences:", sequences[0].shape if sequences else "ไม่มีข้อมูล")

