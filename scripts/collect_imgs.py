# collect_imgs.py
# Collects training images for each ASL sign via your webcam.
#
# CRITICAL: The frame is flipped horizontally (cv2.flip) BEFORE saving.
# This matches what app.py does at inference time. Without this flip, the model
# trains on mirror-opposite orientations and confuses signs (e.g. L predicted as X).
#
# Run order:
#   1. python collect_imgs.py
#   2. python create_dataset.py
#   3. python train_classifier.py
#   4. python app.py

import os
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data' / 'raw'
os.makedirs(DATA_DIR, exist_ok=True)

# 850 images per class — more varied samples improve robustness to distance/angle
DATASET_SIZE = 850

# Class index -> sign name. Must match labels_dict in app.py exactly.
LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}
NUMBER_OF_CLASSES = len(LABELS)

# CAP_DSHOW: Windows DirectShow backend — avoids the MSMF "can't grab frame" error
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("ERROR: Cannot open webcam. Check that no other app is using it.")
    exit(1)

print("\n=== SignLingo - Data Collection ===")
print(f"Will collect {DATASET_SIZE} images for each of {NUMBER_OF_CLASSES} sign classes.")
print("Controls:  Q = start capturing    S = skip this sign    Q during wait = quit")
print("TIP: Move slowly: close, medium, far, left/right, up/down, slight wrist rotation.\n")

for j in range(NUMBER_OF_CLASSES):
    sign_name = LABELS[j]
    class_dir = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_dir, exist_ok=True)

    print(f"[{j+1}/{NUMBER_OF_CLASSES}] Get ready to show: '{sign_name}'  (data/raw/{j}/)")
    print("  -> Hold the sign steady, then press Q to begin. Press S to skip. Press ESC to quit all.")

    skip_class = False
    quit_all = False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("  WARNING: Bad frame from camera, retrying...")
            continue
        # FLIP to match inference orientation in app.py
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Sign: '{sign_name}' ({j}/{NUMBER_OF_CLASSES-1})",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Q = Start    S = Skip    ESC = Quit",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Move slowly: close, medium, far, left/right, up/down, slight wrist rotation",
                    (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow('SignLingo - Data Collection', frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            skip_class = True
            print(f"  -> Skipped '{sign_name}'.\n")
            break
        if key == 27:  # ESC
            quit_all = True
            print("  -> Quit by user.")
            break

    if quit_all:
        break
    if skip_class:
        continue

    counter = 0
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("  WARNING: Bad frame from camera, retrying...")
            continue
        # FLIP - must match the ready-loop above and app.py
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Capturing '{sign_name}': {counter+1}/{DATASET_SIZE}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Move slowly: close, medium, far, left/right, up/down, slight wrist rotation",
                    (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow('SignLingo - Data Collection', frame)
        key = cv2.waitKey(25) & 0xFF
        if key == 27:  # ESC — abort current class and quit all
            print("  -> Quit by user.")
            quit_all = True
            break
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1
        if counter % 100 == 0 or counter == DATASET_SIZE:
            print(f"  Captured {counter}/{DATASET_SIZE} for '{sign_name}'")

    print(f"  Done with '{sign_name}'\n")
    if quit_all:
        break

cap.release()
cv2.destroyAllWindows()
print("=== Collection complete! Next: python create_dataset.py ===")
