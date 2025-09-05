import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# โหลดข้อมูล (แก้ชื่อไฟล์ให้ตรงกับของคุณ)
X = np.load('X.npy')  # ข้อมูล keypoints
y = np.load('y.npy')  # label (one-hot หรือ label ตัวเลข)

# กำหนด labels ให้ตรงกับคำที่เทรนจริง (ตัวอย่าง 3 คำ)
labels = np.array([''])

print("Shape ของ X:", X.shape)
print("จำนวนตัวอย่างใน X:", X.shape[0])
if X.shape[0] == 0:
    raise ValueError("X ไม่มีตัวอย่างข้อมูลเลย! กรุณาตรวจสอบไฟล์ X.npy")
print("Shape ของ y:", y.shape)


# แปลง y เป็น label ตัวเลข (ถ้าเป็น one-hot)
if y.ndim > 1 and y.shape[1] > 1:
    y_true = np.argmax(y, axis=1)
else:
    y_true = y.astype(int)

# โหลดโมเดล LSTM ที่เทรนไว้
model = load_model('signlang_lstm.h5')
print("Model output shape:", model.output_shape)

# ทำนาย label ของข้อมูล X
y_pred_probs = model.predict(X)
y_pred = np.argmax(y_pred_probs, axis=1)

labels = np.load('label_classes.npy')
print("จำนวน labels ที่โหลดได้:", len(labels))  # ต้องได้ 26

# แสดงรายงานผลการทำนาย (precision, recall, f1-score)
print(classification_report(y_true, y_pred, target_names=labels))

# สร้าง confusion matrix และ plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='plasma',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

