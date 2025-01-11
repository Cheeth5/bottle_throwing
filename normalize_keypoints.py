import json
import numpy as np

# Function to normalize keypoints
def normalize_keypoints(keypoints):
    # Filter out invalid keypoints and print the invalid ones
    valid_keypoints = [kp for kp in keypoints if isinstance(kp, dict) and 'x' in kp and 'y' in kp and 'z' in kp]
    
    # Print invalid keypoints for debugging
    if len(valid_keypoints) == 0:
        print("No valid keypoints found.")
        print("Raw keypoints data:", keypoints)  # Print raw data for inspection
    else:
        print(f"Found {len(valid_keypoints)} valid keypoints.")
    
    # Ensure we have valid keypoints before attempting to find max value
    if len(valid_keypoints) == 0:
        raise ValueError("No valid keypoints available for normalization.")

    # Extract x, y, z values and find the max value to normalize by
    max_val = np.max([kp['x'] for kp in valid_keypoints] + [kp['y'] for kp in valid_keypoints] + [kp['z'] for kp in valid_keypoints])
    
    # Normalize keypoints and add label
    normalized_keypoints = []
    for kp in valid_keypoints:
        normalized_kp = {
            'x': kp['x'] / max_val,
            'y': kp['y'] / max_val,
            'z': kp['z'] / max_val,
            'visibility': kp.get('visibility', 1.0),  # Default visibility to 1.0 if missing
            'label': 'throwing'
        }
        normalized_keypoints.append(normalized_kp)
    
    return normalized_keypoints

# Load the JSON file containing the keypoints
input_file = '/home/cheeth/Desktop/yolov11-exhibition/frames/keypoints.json'  # Corrected file path
with open(input_file, 'r') as file:
    data = json.load(file)

# Print the structure of the data to understand its format
print("Data structure:")
print(data)  # Inspect the structure to confirm the format

# Check if the data is a list or a dictionary and extract keypoints accordingly
if isinstance(data, list):
    keypoints = data  # If the data is a list of keypoints
elif isinstance(data, dict):
    keypoints = data.get('keypoints', [])  # If keypoints are nested in a dictionary
else:
    keypoints = []

# Log the keypoints to verify their structure
print("Keypoints extracted:")
print(keypoints)  # Print the keypoints to inspect their structure

# Normalize the keypoints and add labels
normalized_keypoints = normalize_keypoints(keypoints)

# Optionally, save the normalized keypoints with labels to a new JSON file
output_file = '/home/cheeth/Desktop/yolov11-exhibition/frames/normalized_keypoints.json'  # Path to save the output
with open(output_file, 'w') as file:
    json.dump(normalized_keypoints, file, indent=4)

# Print a sample of the normalized keypoints
print("Sample normalized keypoints:")
print(normalized_keypoints[:5])  # Print first 5 keypoints for inspection
