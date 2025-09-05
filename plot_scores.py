import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import tensorflow as tf

# โหลดข้อมูล
X = np.load('X.npy')                      # keypoints ลำดับภาพ
y = np.load('y.npy')                      # label (one-hot หรือเลข class)
labels = np.load('label_classes.npy')           # รายชื่อคลาสทั้งหมด (เป็น array of str)
model = tf.keras.models.load_model('signlang_lstm.h5')  # โหลดโมเดล

# ทำนาย
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred, axis=1)

# แปลง y เป็น class index ถ้ายังเป็น one-hot
if y.ndim == 2:
    y_true = np.argmax(y, axis=1)
else:
    y_true = y

# คำนวณ metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, zero_division=0)

# พล็อตกราฟ
x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(16, 6))
plt.bar(x - width, precision, width=width, label='Precision', color='skyblue')
plt.bar(x, recall, width=width, label='Recall', color='lightgreen')
plt.bar(x + width, f1, width=width, label='F1-score', color='salmon')

plt.xticks(x, labels, rotation=90)
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.title('Precision / Recall / F1-score per Class')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
