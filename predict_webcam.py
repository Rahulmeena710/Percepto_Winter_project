
import cv2
import mediapipe as mp
import numpy as np
import joblib

# =========================================
# 1. LOAD TRAINED MODEL
# =========================================

MODEL_PATH = "models/best_hand_sign_model.pkl"

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
classes = saved_data["classes"]

print("Model loaded successfully!")
print("Selected model:", saved_data["model_name"])
print("Recognized alphabets:", classes)


# =========================================
# 2. MEDIAPIPE SETUP
# =========================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# =========================================
# 3. FEATURE EXTRACTION
# =========================================

def extract_features(hand):

    landmarks = []

    for landmark in hand.landmark:

        landmarks.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    landmarks = np.array(landmarks, dtype=np.float32)

    # Reshape into 21 landmarks and 3 coordinates
    landmarks = landmarks.reshape(21, 3)

    # Make wrist the origin
    landmarks = landmarks - landmarks[0]

    # Normalize coordinates
    scale = np.max(np.abs(landmarks[:, :2]))

    if scale > 0:
        landmarks = landmarks / scale

    return landmarks.flatten()


# =========================================
# 4. START WEBCAM
# =========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot access webcam!")

print("\nWebcam started!")
print("Press Q to quit.")


# =========================================
# 5. REAL-TIME PREDICTION
# =========================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Failed to read webcam frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Detect hand
    results = hands.process(rgb_frame)

    prediction = "Show your hand"
    confidence = 0.0

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Extract 63 features
        features = extract_features(hand_landmarks)

        features = features.reshape(1, -1)

        # Predict probabilities
        probabilities = model.predict_proba(features)[0]

        # Get highest probability
        best_index = np.argmax(probabilities)

        predicted_label = model.classes_[best_index]

        confidence = probabilities[best_index] * 100

        # Confidence threshold
        if confidence >= 60:
            prediction = predicted_label
        else:
            prediction = "Uncertain"

    # =====================================
    # 6. DISPLAY PREDICTION
    # =====================================

    cv2.rectangle(
        frame,
        (10, 10),
        (450, 125),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (25, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (25, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Percepto Hub - ASL Recognition",
        frame
    )

    # Quit using Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================
# 7. RELEASE RESOURCES
# =========================================

cap.release()
hands.close()

cv2.destroyAllWindows()

print("Webcam closed successfully!")


