
import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

# Configuration
DATA_DIR = "dataset"
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
TARGET_SAMPLES = 500
SAMPLE_INTERVAL = 0.001

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def extract_features(hand):
    landmarks = []

    for landmark in hand.landmark:
        landmarks.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    landmarks = np.array(landmarks, dtype=np.float32)

    # Make wrist the origin
    landmarks = landmarks.reshape(21, 3)
    landmarks = landmarks - landmarks[0]

    # Normalize coordinates
    scale = np.max(np.abs(landmarks[:, :2]))

    if scale > 0:
        landmarks = landmarks / scale

    return landmarks.flatten()


def get_sample_count(label):
    file_path = os.path.join(DATA_DIR, label, "data.csv")

    if not os.path.exists(file_path):
        return 0

    with open(file_path, "r", newline="") as f:
        return sum(1 for _ in f)


# Start webcam
cap = cv2.VideoCapture(0)

current_index = 0
current_label = CLASSES[current_index]

count = get_sample_count(current_label)
collecting = False
last_sample_time = 0

print("Controls:")
print("N = Next alphabet")
print("S = Start collecting")
print("P = Pause collecting")
print("Q = Quit")

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not read webcam.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    features = None

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        features = extract_features(hand_landmarks)

    # Display information
    status = "COLLECTING" if collecting else "PAUSED"

    cv2.putText(
        frame,
        f"Alphabet: {current_label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {count}/{TARGET_SAMPLES}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if collecting else (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "N: Next | S: Start | P: Pause | Q: Quit",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # Collect data
    if collecting and features is not None:

        now = time.time()

        if now - last_sample_time >= SAMPLE_INTERVAL:

            folder = os.path.join(DATA_DIR, current_label)

            # Create alphabet folder if it doesn't exist
            os.makedirs(folder, exist_ok=True)

            file_path = os.path.join(folder, "data.csv")

            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(features.tolist())

            count += 1
            last_sample_time = now

            if count % 50 == 0:
                print(f"{current_label}: {count}/{TARGET_SAMPLES}")

            if count >= TARGET_SAMPLES:
                collecting = False
                print(f"Completed {current_label}!")

    cv2.imshow("Hand Sign Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Start collecting
    if key == ord("s"):
        if count < TARGET_SAMPLES:
            collecting = True
            last_sample_time = 0
            print(f"Started collecting {current_label}")
        else:
            print(f"{current_label} is already complete!")

    # Pause collecting
    elif key == ord("p"):
        collecting = False
        print("Collection paused.")

    # Move to next alphabet
    elif key == ord("n"):
        collecting = False

        if current_index < len(CLASSES) - 1:
            current_index += 1
            current_label = CLASSES[current_index]
            count = get_sample_count(current_label)

            print(f"Selected {current_label}: {count}/{TARGET_SAMPLES}")
        else:
            print("All alphabets completed!")

    # Quit
    elif key == ord("q"):
        break


cap.release()
hands.close()
cv2.destroyAllWindows()

print("Data collection stopped.")
