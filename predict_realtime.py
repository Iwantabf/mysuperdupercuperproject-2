import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model # type: ignore
import time

# โหลดโมเดลและ labels
model = load_model("signlang_lstm.h5")
labels = np.load("label_classes.npy")

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ฟังก์ชันดึง keypoints (ซ้าย + ขวา = 126 ค่า)
def extract_keypoints(results):
    lh = np.zeros(63)
    rh = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]).flatten()
            label = handedness.classification[0].label
            if label == "Left":
                lh = keypoints
            elif label == "Right":
                rh = keypoints
    return np.concatenate([lh, rh])

# กล้อง
cap = cv2.VideoCapture(0)

sequence = []
predicted_text = ''
last_predict_time = 0
cooldown = 1.5  # หน่วงแสดงผล

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        # วาด keypoints และกรอบ
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # วาดกรอบสี่เหลี่ยมรอบมือ
            h, w, _ = frame.shape
            cx_min, cy_min = w, h
            cx_max, cy_max = 0, 0

            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cx_min = min(cx_min, cx)
                cy_min = min(cy_min, cy)
                cx_max = max(cx_max, cx)
                cy_max = max(cy_max, cy)

            cv2.rectangle(frame, (cx_min - 20, cy_min - 20), (cx_max + 20, cy_max + 20), (255, 255, 0), 2)

        # ดึง keypoints
        keypoints = extract_keypoints(result)
        sequence.append(keypoints)

        if len(sequence) == 30:
            input_data = np.expand_dims(sequence, axis=0)
            prediction = model.predict(input_data, verbose=0)[0]
            max_index = np.argmax(prediction)
            confidence = prediction[max_index]

            if confidence > 0.9 and (time.time() - last_predict_time) > cooldown:
                predicted_text = f"{labels[max_index]} ({confidence:.2f})"
                last_predict_time = time.time()

            sequence = []

    else:
        sequence = []  # รีเซ็ตเมื่อไม่เจอมือ

    # แสดงผลลัพธ์บนหน้าจอ
    if predicted_text:
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(frame, predicted_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("🖐️ Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


