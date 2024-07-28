import json
from shapely.geometry import Polygon
from shapely.validation import explain_validity

# Load the JSON file
with open(r'C:\Users\shiri\Desktop\Dataverze\Vision_experimentation\output_coco.json') as f:
    data = json.load(f)

# Function to check and fix polygons
def fix_polygon(polygon):
    poly = Polygon(polygon)
    if not poly.is_valid:
        # Simplify or correct the polygon
        poly = poly.buffer(0)
        if not poly.is_valid:
            print(f"Could not fix polygon: {explain_validity(poly)}")
            return None
    return list(poly.exterior.coords)

# Check and fix ground truth and predicted polygons
for annotation in data['annotations']:
    for key in ['ground_truth_polygon', 'predicted_polygon']:
        if key in annotation:
            polygon = annotation[key]
            fixed_polygon = fix_polygon(polygon)
            if fixed_polygon:
                annotation[key] = fixed_polygon
            else:
                print(f"Invalid polygon in annotation ID {annotation['id']} could not be fixed.")

# Save the fixed JSON file
with open(r'C:\Users\shiri\Desktop\Dataverze\Vision_experimentation\RL_attempt_3\output_coco_fixed.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Polygons have been checked and fixed where possible.")
