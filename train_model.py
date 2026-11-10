
import os
import csv
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
DATA_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_sign_model.pkl")

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

X = []
y = []

print("Loading dataset...")

# Load data from all alphabet folders
for label in CLASSES:

    file_path = os.path.join(DATA_DIR, label, "data.csv")

    if not os.path.exists(file_path):
        print(f"Missing data for {label}")
        continue

    with open(file_path, "r", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            try:
                features = np.array(row, dtype=np.float32)

                if len(features) == 63 and np.all(np.isfinite(features)):
                    X.append(features)
                    y.append(label)

            except ValueError:
                continue

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\nDataset loaded successfully!")
print("Total samples:", len(X))
print("Number of features:", X.shape[1] if len(X) else 0)
print("Number of classes:", len(np.unique(y)))

if len(X) == 0 or len(np.unique(y)) < 2:
    raise ValueError("Not enough valid data to train the model.")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Train model
print("\nTraining started...")
model.fit(X_train, y_train)

print("Training completed!")

# Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=CLASSES))

# Save model
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "classes": CLASSES
    },
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)
