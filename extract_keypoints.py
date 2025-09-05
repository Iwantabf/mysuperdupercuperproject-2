import cv2
import mediapipe as mp
import os
import csv

label = 'sawasdee'
dataset_path = os.path.join('dataset', label)
output_csv = f'{label}_keypoints.csv'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

data = []

# loop รูปทุกภาพ
for img_name in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])  # x, y, z
            row.append(label)
            data.append(row)

# เขียน CSV
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    # เขียน header
    header = [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')] + ['label']
    writer.writerow(header)
    writer.writerows(data)

print(f'✅ Done! Saved keypoints to {output_csv}')
