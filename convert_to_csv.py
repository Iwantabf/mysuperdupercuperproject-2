import numpy as np
import pandas as pd

# โหลดข้อมูล
X = np.load('X.npy')           # shape: (N, 30, 126)
y = np.load('y.npy')           # one-hot หรือ integer
labels = np.load('label_classes.npy')  # ['สวัสดี', 'ป่วย', ...]

# ถ้า y เป็น one-hot ให้แปลงเป็นตัวเลขก่อน
if len(y.shape) > 1:
    y = np.argmax(y, axis=1)

# เตรียมข้อมูลลิสต์ไว้บันทึก
data_rows = []

for i in range(len(X)):
    flattened = X[i].flatten()  # (30 x 126) → 3780 ค่า
    label = labels[y[i]]        # แปลง index กลับเป็นชื่อ
    row = list(flattened) + [label]
    data_rows.append(row)

# สร้างชื่อคอลัมน์: keypoint_0 ... keypoint_3779 + label
columns = [f'kp_{i}' for i in range(X.shape[1] * X.shape[2])] + ['label']

# สร้าง DataFrame
df = pd.DataFrame(data_rows, columns=columns)

# บันทึกเป็น .csv
df.to_csv('sign_data.csv', index=False, encoding='utf-8-sig')
print("✅ บันทึกไฟล์เป็น sign_data.csv เรียบร้อย")
