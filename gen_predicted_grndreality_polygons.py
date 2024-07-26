import os
import json
import numpy as np
from shapely.geometry import Polygon
from PIL import Image, ImageEnhance
import cv2
import torch
from torchvision import transforms as T, models

def load_annotations(json_path):
    with open(json_path, 'r') as file:
        annotations = json.load(file)
    return annotations

# enhance_constrast and gamma_correction are image enchancements

def enhance_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    table = np.array(table, dtype="uint8")
    return Image.fromarray(cv2.LUT(np.array(image), table))

# Get prediction from model
def get_prediction(image, model, transform, threshold):
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        prediction = model(image_tensor)[0]

    # Filter out low confidence predictions
    high_confidence_indices = [i for i, score in enumerate(prediction['scores']) if score > threshold]
    prediction = {k: v[high_confidence_indices] for k, v in prediction.items()}
    return prediction

def select_best_polygon(prediction):
    """
    Best polygons seleccted on the basis on confidence ratings
    Default polygons returned if no valid polygons are found
    """
    if 'masks' not in prediction or len(prediction['masks']) == 0:
        return None
    best_idx = np.argmax(prediction['scores'])  # Select the index of the highest confidence score
    best_mask = prediction['masks'][best_idx][0].cpu().numpy()
    contours, _ = cv2.findContours((best_mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(f"Contours found: {len(contours)}")
    if not contours:
        return None
    best_contour = contours[0]
    #print(f"Best contour points: {len(best_contour)}")
    if len(best_contour) < 2:
        print(f"Contour has insufficient points: {len(best_contour)} points, using default points.")
        return Polygon([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [6, 6], [7, 7], [8, 8]])
    best_contour = interpolate_points(best_contour, 8)  # Ensure 8 vertices
    return Polygon(best_contour.reshape(-1, 2))

def interpolate_points(points, num_points):
    """
    Ensures enough numbers of points are generated to create the initial polygon
    Default polygons returned if initially less than 2 points are generated
    """
    points = np.squeeze(points, axis=1)  # Remove the redundant dimension if present
    if len(points) < 2:
        print("Insufficient points to interpolate, using default points.")
        return np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [6, 6], [7, 7], [8, 8]])  # Return a default polygon if not enough points
    while len(points) < num_points:
        new_points = []
        for i in range(len(points)):
            new_points.append(points[i])
            if i < len(points) - 1:
                midpoint = ((points[i][0] + points[i + 1][0]) / 2, (points[i][1] + points[i + 1][1]) / 2)
                new_points.append(midpoint)
        points = new_points
    return np.array(points[:num_points])

def process_directory(directory_path, json_path, model, transform, threshold, contrast_factor, gamma):
    """
    Proces all images in the directory, applying enhancements
    Uses the model to predict and select best polygon for each image
    Loads ground truth polygons from JSON annotations
    returns a list of dictionaries containing image filenames, predicted polygons, and ground truth polygons
    """
    annotations = load_annotations(json_path)
    polygons_data = []

    for image_filename in os.listdir(directory_path):
        if image_filename.endswith('.json'):
            continue

        image_path = os.path.join(directory_path, image_filename)
        image_id = next((img['id'] for img in annotations['images'] if img['file_name'] == image_filename), None)

        if image_id is None:
            print(f"Image {image_filename} not found in annotations.")
            continue

        image = Image.open(image_path).convert("RGB")
        image = enhance_contrast(image, contrast_factor)
        image = gamma_correction(image, gamma)

        prediction = get_prediction(image, model, transform, threshold)
        if 'masks' not in prediction or len(prediction['masks']) == 0:
            print(f"No predictions for image {image_filename} with threshold {threshold}")
            continue

        best_polygon = select_best_polygon(prediction)
        if best_polygon is None:
            print(f"Could not form a valid polygon for image {image_filename}")
            continue

        ground_truth_polygons = [annotation['segmentation'][0] for annotation in annotations['annotations'] if annotation['image_id'] == image_id]
        ground_truth_polygons = [Polygon([(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]) for poly in ground_truth_polygons]

        if not ground_truth_polygons:
            continue
        ground_truth_polygon = ground_truth_polygons[0]

        polygons_data.append({
            'image': image_filename,
            'prediction': best_polygon,
            'ground_truth': ground_truth_polygon
        })

    return polygons_data

if __name__ == "__main__":
    directory_path = r"C:\Users\shiri\Desktop\Dataverze\Vision_experimentation\only_ships_images\images"
    json_path = r"C:\Users\shiri\Desktop\Dataverze\Vision_experimentation\RL_attempt_3\output_coco_valid_updated.json"
    threshold = 0  # Set a reasonable threshold
    contrast_factor = 5.0
    gamma = 1.2

    # Initialize the model and transform
    model = models.detection.maskrcnn_resnet50_fpn(weights='MaskRCNN_ResNet50_FPN_Weights.COCO_V1')
    model.eval()
    transform = T.Compose([T.ToTensor()])

    polygons_data = process_directory(directory_path, json_path, model, transform, threshold, contrast_factor, gamma)
    print(f"Extracted polygons for {len(polygons_data)} images.")
