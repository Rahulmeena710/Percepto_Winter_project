
import os
import csv
import numpy as np

DATA_DIR = "dataset"
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
TARGET_SAMPLES = 500

print("\n========== DATASET CHECK ==========\n")

total_samples = 0
all_valid = True

for label in CLASSES:

    file_path = os.path.join(DATA_DIR, label, "data.csv")

    if not os.path.exists(file_path):
        print(f"{label}: MISSING DATA FILE")
        all_valid = False
        continue

    with open(file_path, "r", newline="") as f:
        rows = list(csv.reader(f))

    sample_count = len(rows)
    total_samples += sample_count

    valid_samples = 0

    for row in rows:
        try:
            values = np.array(row, dtype=np.float32)

            if len(values) == 63 and np.all(np.isfinite(values)):
                valid_samples += 1

        except ValueError:
            continue

    status = "OK" if (
        sample_count == TARGET_SAMPLES
        and valid_samples == sample_count
    ) else "CHECK"

    if status == "CHECK":
        all_valid = False

    print(
        f"{label}: {sample_count}/500 samples | "
        f"Valid: {valid_samples} | {status}"
    )

print("\n========== SUMMARY ==========")
print("Total samples:", total_samples)
print("Expected samples:", 26 * TARGET_SAMPLES)

if all_valid:
    print("Dataset is ready for training!")
else:
    print("Dataset needs attention. Check the entries above.")

