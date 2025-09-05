import cv2
import numpy as np
import mediapipe as mp
import os
import time

# ---------------- CONFIG ----------------
LABELS = ["wound_inflamed"]
SEQUENCE_LENGTH = 30
BASE_DIR = "dataset"
# ----------------------------------------

# เตรียมโฟลเดอร์ทุก label
for label in LABELS:
    os.makedirs(os.path.join(BASE_DIR, label), exist_ok=True)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# เริ่มจาก label แรก
label_index = 0
current_label = LABELS[label_index]
sample_id = len(os.listdir(os.path.join(BASE_DIR, current_label)))
sequence = []
recording = False

print("🎥 Press 's' to start recording | 'n' to next label | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            landmark = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            if hand_label == "Left":
                left_hand = landmark
            elif hand_label == "Right":
                right_hand = landmark
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    keypoints = np.concatenate([left_hand, right_hand])

    if recording:
        sequence.append(keypoints)
        cv2.putText(image, f"Recording {len(sequence)}/{SEQUENCE_LENGTH}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if len(sequence) == SEQUENCE_LENGTH:
            save_path = os.path.join(BASE_DIR, current_label, f"{sample_id}.npy")
            np.save(save_path, np.array(sequence))
            print(f"✅ Saved {current_label}/{sample_id}.npy")
            sample_id += 1
            sequence = []
            recording = False
            time.sleep(1)

    else:
        cv2.putText(image, f"Label: {current_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    cv2.imshow("Multi-Label Collector", image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s') and not recording:
        print(f"▶️ Start recording: {current_label}")
        recording = True
        sequence = []
    elif key == ord('n'):  # Next label
        label_index = (label_index + 1) % len(LABELS)
        current_label = LABELS[label_index]
        sample_id = len(os.listdir(os.path.join(BASE_DIR, current_label)))
        print(f"➡️ Switched to label: {current_label}")

cap.release()
cv2.destroyAllWindows()
