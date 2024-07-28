import json
import os
import shutil

# Paths to the original dataset and annotations
original_image_dir = 'C:\\Users\\shiri\\Desktop\\Dataverze\\Vision_experimentation\\shipsnet\\shipsnet'
annotations_path = 'C:\\Users\\shiri\\Desktop\\Dataverze\\Vision_experimentation\\shipsnet.json'

# Load the original annotations
with open(annotations_path, 'r') as f:
    annotations = json.load(f)

# Inspect the keys to understand the structure
print("Keys in annotations JSON:", annotations.keys())

# Directory for the new dataset
new_image_dir = 'C:\\Users\\shiri\\Desktop\\Dataverze\\Vision_experimentation\\only_ships'
os.makedirs(new_image_dir, exist_ok=True)

# New annotations structure
new_annotations = {
    "data": [],
    "labels": [],
    "locations": [],
    "scene_ids": []
}

# Filter and copy images with ships
missing_files = []
for i, label in enumerate(annotations['labels']):
    if label == 1:
        # Copy image
        scene_id = annotations['scene_ids'][i]
        original_image_path = os.path.join(original_image_dir, scene_id + '.jpg')
        new_image_path = os.path.join(new_image_dir, scene_id + '.jpg')
        
        # Print the image path being processed
        print(f"Processing: {original_image_path}")
        
        if os.path.exists(original_image_path):
            shutil.copy(original_image_path, new_image_path)
            
            # Append to new annotations
            new_annotations['data'].append(annotations['data'][i])
            new_annotations['labels'].append(annotations['labels'][i])
            new_annotations['locations'].append(annotations['locations'][i])
            new_annotations['scene_ids'].append(annotations['scene_ids'][i])
        else:
            missing_files.append(original_image_path)
            print(f"File not found: {original_image_path}")

# Save the new annotations to a new JSON file
new_annotations_path = 'C:\\Users\\shiri\\Desktop\\Dataverze\\Vision_experimentation\\only_ships.json'
with open(new_annotations_path, 'w') as f:
    json.dump(new_annotations, f)

print("New dataset and annotations created successfully.")
if missing_files:
    print("The following files were not found:")
    for file in missing_files:
        print(file)
