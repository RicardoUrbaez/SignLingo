# train_classifier.py
# Trains a Random Forest classifier on data.pickle and saves model.p.
#
# Key fixes vs original:
#   - Removed broken reshape loop (treated float elements as landmark arrays)
#   - Filters out any samples with wrong feature count before training
#   - n_estimators=200 for better accuracy
#   - Prints accuracy, full classification report, and confusion matrix

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

EXPECTED_FEATURES = 42  # 21 landmarks x 2 coords (single hand)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / 'models' / 'data.pickle'
MODEL_PATH = REPO_ROOT / 'models' / 'model.p'

print("\n=== SignLingo - Training Classifier ===")

with DATA_PATH.open('rb') as data_file:
    data_dict = pickle.load(data_file)
raw_data   = data_dict['data']
raw_labels = data_dict['labels']

print(f"Loaded {len(raw_data)} samples from {DATA_PATH}.")

# Filter out any samples that don't have exactly 42 features
filtered = [(d, l) for d, l in zip(raw_data, raw_labels) if len(d) == EXPECTED_FEATURES]
removed = len(raw_data) - len(filtered)
if removed > 0:
    print(f"Removed {removed} samples with wrong feature count (expected {EXPECTED_FEATURES}).")

if len(filtered) == 0:
    print("ERROR: No valid samples to train on. Re-run create_dataset.py.")
    exit(1)

X = np.array([d for d, _ in filtered], dtype=np.float32)
y = np.array([l for _, l in filtered])

print(f"Training on {len(X)} samples across {len(np.unique(y))} classes.")
print(f"Feature vector size per sample: {X.shape[1]}")

# Stratified split so every class appears in both train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
print("\nTraining RandomForestClassifier (n_estimators=200) ...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n=== Results ===")
print(f"Test Accuracy: {acc * 100:.2f}%")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, zero_division=0))

print("--- Confusion Matrix (rows=actual, cols=predicted) ---")
classes = sorted(np.unique(y), key=lambda x: int(x) if x.isdigit() else x)
cm = confusion_matrix(y_test, y_pred, labels=classes)
# Print with class labels for readability
header = "      " + "  ".join(f"{c:>3}" for c in classes)
print(header)
for cls, row in zip(classes, cm):
    print(f"  {cls:>3}: " + "  ".join(f"{v:>3}" for v in row))

print(f"\nSaving model to {MODEL_PATH} ...")
with MODEL_PATH.open('wb') as f:
    pickle.dump({'model': model}, f)

print("model.p saved.")
print("Next step: python app.py")
