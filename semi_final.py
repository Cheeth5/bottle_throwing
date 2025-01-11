import cv2 as cv
import mediapipe as mp
import time
import joblib
import numpy as np

# Load the trained model (replace with the correct path to your .pkl model)
model_file = "trained_model.pkl"
model = joblib.load(model_file)

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up the pose model
pose_model = mp_pose.Pose(
    model_complexity=0,  # Faster model, less accurate
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global variable to store previous landmarks
prev_landmarks = None

# Function to extract pose landmarks
def extract_landmarks(results, prev_landmarks=None):
    landmarks = []

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

    # Ensure landmarks list has enough values before accessing wrist positions
    if len(landmarks) >= 132:  # Checking that we have enough landmarks (132 values: 44 landmarks with 3 values each)
        # Track the change in wrist position over time (if previous landmarks are available)
        if prev_landmarks is not None:
            prev_wrist = prev_landmarks[-9:-6]  # Assuming wrist landmarks are in the last three features (adjust accordingly)
            wrist = landmarks[-9:-6]

            # Calculate the change in wrist position
            wrist_change = [wrist[i] - prev_wrist[i] for i in range(3)]

            # Append the wrist change to the landmarks (this adds temporal data to the input)
            landmarks.extend(wrist_change)
    
    # Ensure the landmarks array has exactly 131 features
    while len(landmarks) < 131:
        landmarks.append(0.0)
    if len(landmarks) > 131:
        landmarks = landmarks[:131]
    
    return np.array(landmarks)

# Main function
def main():
    global prev_landmarks
    cap = cv.VideoCapture(0)
    prev_time = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv.resize(image, (640, 480))
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        results = pose_model.process(image_rgb)

        if results.pose_landmarks:
            # Extract landmarks only if pose landmarks are available
            landmarks = extract_landmarks(results, prev_landmarks)

            if len(landmarks) > 0:
                action = model.predict([landmarks])[0]  # Predict action based on landmarks
            # Update prev_landmarks
            prev_landmarks = landmarks

            # Draw landmarks and show the action on screen
            image.flags.writeable = True
            image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            if len(landmarks) > 0:
                cv.putText(image, f'Action: {action}', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            cv.putText(image, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Show the result
            cv.imshow('MediaPipe Pose', image)

        if cv.waitKey(5) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
