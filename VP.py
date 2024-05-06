import cv2
import mediapipe as mp
import pyautogui
import threading

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector
hand_detector = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Get screen size
screen_width, screen_height = pyautogui.size()


# Function to move cursor
def move_cursor(index_x, index_y):
    pyautogui.moveTo(index_x, index_y)


# Function to process hand landmarks
def process_hand(frame):
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hand_detector.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract index finger and thumb landmarks
            index_x, index_y = int(
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * screen_width), \
                int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * screen_height)
            thumb_x, thumb_y = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x * screen_width), \
                int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y * screen_height)

            # Click if thumb is close to index finger
            if abs(index_y - thumb_y) < 20:
                pyautogui.click()

            # Move cursor if thumb is within range
            elif abs(index_y - thumb_y) < 100:
                threading.Thread(target=move_cursor, args=(index_x, index_y)).start()


# Main loop
while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Process hand landmarks
    process_hand(frame)

    # Display frame
    cv2.imshow('Virtual Pointer', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
