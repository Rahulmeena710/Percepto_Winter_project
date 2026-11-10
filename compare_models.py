
import os
import csv
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Configuration
DATA_DIR = "dataset"
MODEL_DIR = "models"

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# =====================================
# 1. LOAD DATASET
# =====================================

X = []
y = []

print("Loading dataset...")

for label in CLASSES:

    file_path = os.path.join(DATA_DIR, label, "data.csv")

    if not os.path.exists(file_path):
        print(f"Missing data: {label}")
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

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

if len(X) == 0 or set(y) != set(CLASSES):
    raise ValueError("Dataset must contain valid samples for all 26 alphabets.")

# =====================================
# 2. TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# =====================================
# 3. DEFINE CLASSIFIERS
# =====================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1
        ))
    ])
}

# =====================================
# 4. FIVE-FOLD CROSS VALIDATION
# =====================================

results = []
trained_models = {}

for name, model in models.items():

    print("\n================================")
    print("Evaluating:", name)
    print("================================")

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro",
            "f1": "f1_macro"
        },
        n_jobs=1
    )

    cv_accuracy = scores["test_accuracy"].mean()
    cv_f1 = scores["test_f1"].mean()

    print(f"5-Fold CV Accuracy: {cv_accuracy * 100:.2f}%")
    print(f"5-Fold CV F1 Score: {cv_f1:.4f}")

    # Train on complete training split
    model.fit(X_train, y_train)

    test_accuracy = model.score(X_test, y_test)

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    results.append({
        "Model": name,
        "CV Accuracy (%)": cv_accuracy * 100,
        "CV F1 Score": cv_f1,
        "Test Accuracy (%)": test_accuracy * 100
    })

    trained_models[name] = model

# =====================================
# 5. COMPARE RESULTS
# =====================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV Accuracy (%)",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

os.makedirs(MODEL_DIR, exist_ok=True)

results_df.to_csv(
    os.path.join(MODEL_DIR, "model_comparison.csv"),
    index=False
)

# =====================================
# 6. SAVE BEST MODEL
# =====================================

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

joblib.dump(
    {
        "model": best_model,
        "classes": CLASSES,
        "model_name": best_model_name
    },
    os.path.join(MODEL_DIR, "best_hand_sign_model.pkl")
)

print("\nBest model:", best_model_name)
print("Saved: models/best_hand_sign_model.pkl")
print("Comparison saved: models/model_comparison.csv")
