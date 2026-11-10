
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Drawing utility to draw landmarks
mp_drawing = mp.solutions.drawing_utils

# Initialize hand detection model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Access webcam
cap = cv2.VideoCapture(0)

while True:

    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Unable to access camera")
        break

    # Flip image horizontally
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Detect hand landmarks
    results = hands.process(rgb_frame)

    # Check if a hand is detected
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Display output
    cv2.imshow("Hand Landmark Detection", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
hands.close()
cv2.destroyAllWindows()
