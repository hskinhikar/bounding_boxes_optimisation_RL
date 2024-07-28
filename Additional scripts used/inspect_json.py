import json

# Path to the original annotations
annotations_path = r"C:\Users\shiri\Desktop\Dataverze\Vision_experimentation\shipsnet.json"

# Load the original annotations
with open(annotations_path, 'r') as f:
    annotations = json.load(f)

# Print the keys of the JSON to understand its structure
print(annotations.keys())