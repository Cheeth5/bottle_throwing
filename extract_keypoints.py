import mediapipe as mp
import cv2
import os
import json

# Mediapipe setup for pose keypoints extraction
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# Paths
frame_dir = "/home/cheeth/Desktop/yolov11-exhibition/frames/frame/"  # Directory with your filtered frames
output_keypoints_file = "keypoints_new.json"  # Output file to save keypoints

# Initialize list to store keypoints for each frame
keypoints = []

# Process each frame in the directory
for frame_file in sorted(os.listdir(frame_dir)):
    frame_path = os.path.join(frame_dir, frame_file)

    # Read the frame
    image = cv2.imread(frame_path)

    if image is None:
        continue  # Skip if the frame is unreadable

    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to extract keypoints
    results = pose.process(image_rgb)

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        frame_keypoints = []

        # Extract and store keypoints
        for lm in results.pose_landmarks.landmark:
            frame_keypoints.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })

        keypoints.append(frame_keypoints)

# Save the keypoints to a JSON file
with open(output_keypoints_file, "w") as f:
    json.dump(keypoints, f, indent=4)

print(f"Extracted and saved keypoints for {len(keypoints)} frames.")
