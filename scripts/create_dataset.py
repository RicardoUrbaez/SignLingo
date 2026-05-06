# create_dataset.py
# Processes images in data/ and extracts MediaPipe hand landmarks into data.pickle.
#
# Key fixes vs original:
#   - Only uses the FIRST detected hand per image (prevents variable-length feature vectors)
#   - Guarantees exactly 42 features (21 landmarks x 2 coords) per sample
#   - Skips images where no hand is detected or landmark count is wrong
#   - Removes noisy debug prints, shows a clean progress summary instead

import os
import pickle
from pathlib import Path

import mediapipe as mp
import cv2

EXPECTED_FEATURES = 42   # 21 landmarks × 2 (x, y) — single hand only

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data' / 'raw'
OUTPUT_PATH = REPO_ROOT / 'models' / 'data.pickle'

data = []
labels = []

all_classes = sorted(os.listdir(DATA_DIR), key=lambda x: int(x) if x.isdigit() else x)

total_saved = 0
total_skipped = 0

print("\n=== SignLingo - Building Dataset ===")

for dir_ in all_classes:
    class_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    saved = 0
    skipped = 0

    for img_file in images:
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            skipped += 1
            continue

        # Use only the FIRST detected hand — keeps feature length fixed at 42
        hand_landmarks = results.multi_hand_landmarks[0]

        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]

        data_aux = []
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        # Sanity-check: skip if landmark count is wrong (corrupted image, partial hand)
        if len(data_aux) != EXPECTED_FEATURES:
            skipped += 1
            continue

        data.append(data_aux)
        labels.append(dir_)
        saved += 1

    total_saved += saved
    total_skipped += skipped
    print(f"  Class {dir_:>3}: {saved} samples saved, {skipped} skipped (no hand / bad read)")

print(f"\nTotal: {total_saved} samples saved, {total_skipped} images skipped.")

if total_saved == 0:
    print("ERROR: No samples were saved. Check that your data/ images show a clear hand.")
    exit(1)

with OUTPUT_PATH.open('wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"data.pickle saved successfully at {OUTPUT_PATH}.")
print("Next step: python train_classifier.py")
