import json

def normalize_keypoints(keypoints, min_vals, max_vals):
    """
    Normalize keypoints based on min and max values.
    """
    normalized = []
    for kp in keypoints:
        normalized.append({
            "x": (kp["x"] - min_vals["x"]) / (max_vals["x"] - min_vals["x"]),
            "y": (kp["y"] - min_vals["y"]) / (max_vals["y"] - min_vals["y"]),
            "z": (kp["z"] - min_vals["z"]) / (max_vals["z"] - min_vals["z"]),
            "visibility": kp["visibility"]
        })
    return normalized

def label_and_normalize(json_file, label, output_file):
    with open(json_file, 'r') as file:
        data = json.load(file)  # data is a list of lists

    labeled_data = []

    # Flatten the keypoints to find global min/max values
    all_keypoints = [kp for frame in data for kp in frame]
    min_vals = {k: min(kp[k] for kp in all_keypoints) for k in ["x", "y", "z"]}
    max_vals = {k: max(kp[k] for kp in all_keypoints) for k in ["x", "y", "z"]}

    # Process each frame
    for frame_index, frame in enumerate(data):
        normalized_keypoints = normalize_keypoints(frame, min_vals, max_vals)
        labeled_data.append({
            "frame": frame_index + 1,  # Add a frame index for reference
            "label": label,
            "keypoints": normalized_keypoints
        })

    # Save the output
    with open(output_file, 'w') as outfile:
        json.dump(labeled_data, outfile, indent=4)

# Example usage
label_and_normalize("keypoints_new.json", "throwing", "labeled_normalized_keypoints.json")
