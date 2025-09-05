import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
import joblib

# ---------- CONFIG ----------
DATASET_PATH = "dataset"
SEQUENCE_LENGTH = 30
INPUT_SHAPE = (SEQUENCE_LENGTH, 126)
EPOCHS = 100
BATCH_SIZE = 16
MODEL_NAME = "signlang_lstm.h5"
# ----------------------------

# ---------- LOAD DATA ----------
X, y = [], []

# กรองเฉพาะโฟลเดอร์ label ที่แท้จริง
labels = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

print(f"📁 พบ labels: {labels}")

for label in labels:
    label_dir = os.path.join(DATASET_PATH, label)
    for file in os.listdir(label_dir):
        if file.endswith(".npy"):
            path = os.path.join(label_dir, file)
            sequence = np.load(path)
            if sequence.shape == INPUT_SHAPE:
                X.append(sequence)
                y.append(label)

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples from {len(labels)} labels")

# ---------- ENCODE LABELS ----------
encoder = LabelEncoder()
y_integer = encoder.fit_transform(y)

# ✅ Split ด้วย integer labels
X_train, X_test, y_train_int, y_test_int = train_test_split(
    X, y_integer, test_size=0.2, stratify=y_integer, random_state=42
)

# ✅ แปลงเป็น one-hot ด้วยจำนวน class ที่แน่นอน
y_train = to_categorical(y_train_int, num_classes=len(labels))
y_test = to_categorical(y_test_int, num_classes=len(labels))

# ✅ Debug shapes
print(f"📊 X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"📊 X_test: {X_test.shape}, y_test: {y_test.shape}")

# ---------- MODEL ----------
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=INPUT_SHAPE))
model.add(Dropout(0.4))
model.add(LSTM(64))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------- TRAIN ----------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(MODEL_NAME, save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ---------- SAVE ----------
model.save(MODEL_NAME)
joblib.dump(encoder, "label_encoder.pkl")
print(f"\n🎉 Training complete! Model saved as '{MODEL_NAME}' and encoder as 'label_encoder.pkl'")
model.save("sign_model.h5")





