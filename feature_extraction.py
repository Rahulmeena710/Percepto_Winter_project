
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)


def extract_features(hand_landmarks):

    features = []

    # Extract all 21 landmark coordinates
    for landmark in hand_landmarks.landmark:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    # Convert list into NumPy array
    features = np.array(features).reshape(21, 3)

    # Make wrist the origin
    features = features - features[0]

    # Normalize based on hand size
    scale = np.max(np.abs(features[:, :2]))

    if scale > 0:
        features = features / scale

    # Flatten into 63-dimensional vector
    return features.flatten()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # Extract 63 features
        features = extract_features(hand)

        cv2.putText(
            frame,
            f"Features: {len(features)}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Print features when P is pressed
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print("Feature vector:")
            print(features)
            print("Shape:", features.shape)

    cv2.imshow("Feature Extraction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
