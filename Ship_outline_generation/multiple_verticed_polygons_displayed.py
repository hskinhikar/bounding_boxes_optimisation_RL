import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision import models, transforms 
import numpy as np
from skimage import exposure, measure, restoration
import os
from shapely.geometry import Polygon, MultiPoint
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

"""
This generates polygons with 8 vertices for every image provided.
Each image undergoes enchancement so that polygons will be predicted.
The enhanced image will be shown prior to the predicted polygon.
Next stage of this is to find a way to pick only one of the multiple polygons generated.
Final stage would be to implement this into the already built RL code and train the model.

"""

model = models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()

# define the transformation

transform = transforms.Compose([
    transforms.ToTensor(), 
])

#function to load ground truth annotations from JSON
def load_annotations(json_path):
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    return annotations

# function to enhance image constrast
def enhance_contrast(image_path, factor=2.0):
    image = Image.open(image_path).convert("RGB")
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(factor)
    return enhanced_image


# Function to apply gamma correction
def gamma_correction(image, gamma=1.0):
    image_np = np.array(image)
    image_gamma = exposure.adjust_gamma(image_np, gamma)
    image_gamma = (image_gamma * 255).astype(np.uint8)
    return Image.fromarray(image_gamma)

# Function to apply sharpening
def sharpen_image(image):
    return image.filter(ImageFilter.SHARPEN)


# function to get the prediction from model
def get_prediction(image, model, transform, threshold):
    image_tensor = transform(image)
    with torch.no_grad():
        prediction = model([image_tensor])[0]

    # filter out low confidence predictions
    high_confidence_indices = [i for i, score in enumerate(prediction['scores']) if score > threshold]
    prediction = {k: v[high_confidence_indices] for k, v in prediction.items()}

    return prediction

# function to simplify polygon using Douglas-Peuker algorithm
def simplify_polygon(contour, num_vertices=8):
    polygon = Polygon(contour)
    tolerance = 0.01
    while len(polygon.exterior.coords) > num_vertices:
        simplified = polygon.simplify(tolerance, preserve_topology=False)
        if len(simplified.exterior.coords) <= num_vertices:
            polygon = simplified
        tolerance += 0.01
    coords = np.array(polygon.exterior.coords[:-1]) # excludes closing point which duplicates the first point
    return coords

# Function to convert binary masks to polygons
def mask_to_polygon(mask, num_vertices=8):
    contours = measure.find_contours(mask, 0.5)
    polygons = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        if len(contour) > num_vertices:
            contour = simplify_polygon(contour, num_vertices)
        polygons.append(contour)
    return polygons



# Function to draw polygons on the image
def draw_polygons(image_path, annotations, prediction, image_id, num_vertices=8):
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw ground truth polygons
    for annotation in annotations['annotations']:
        if annotation['image_id'] == image_id:
            poly = annotation['segmentation'][0]
            poly = [poly[i:i + 2] for i in range(0, len(poly), 2)]
            polygon = patches.Polygon(poly, closed=True, linewidth=2, edgecolor='g', facecolor='none', label = 'Ground Truth')
            ax.add_patch(polygon)

    # Draw predicted polygons
    masks = prediction['masks'].numpy()
    for mask in masks:
        mask = mask[0]
        polygons = mask_to_polygon(mask, num_vertices)
        for poly in polygons:
            polygon = patches.Polygon(poly, closed=True, linewidth=2, edgecolor='r', facecolor='none', label = 'Prediction')
            ax.add_patch(polygon)

    
    # create custom lehemt
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.show()

def process_directory(directory_path, json_path, threshold,  contrast_factor, gamma):
    # Load annotations
    annotations = load_annotations(json_path)

    # Process each image in the directory
    for image_filename in os.listdir(directory_path):
        if image_filename.endswith('.json'):
            continue

        image_path = os.path.join(directory_path, image_filename)

        # Get image ID from the file name
        image_id = next((img['id'] for img in annotations['images'] if img['file_name'] == image_filename), None)

        if image_id is None:
            print(f"Image {image_filename} not found in annotations.")
            continue
            
        image = Image.open(image_path).convert("RGB")
        # image enhancements
        image = enhance_contrast(image_path, contrast_factor)
        image = gamma_correction(image, gamma
        image = sharpen_image(image)

                # Display the enhanced image
        plt.figure()
        plt.title(f"Enhanced Image: {image_filename}")
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        # Get model prediction
        prediction = get_prediction(image, model, transform, threshold)

        if len(prediction['masks']) == 0:
            print(f"No predictions for image {image_filename} with threshold {threshold}")
            
        draw_polygons(image_path, annotations, prediction, image_id)



def main():

    image_path =  r"C:\Users\shiri\Desktop\Dataverze\Vision_experimentation\only_ships_images"
    json_path = r"C:\Users\shiri\Desktop\Dataverze\Vision_experimentation\output_coco.json"

    threshold = 0  # Minimal confidence threshold
    contrast_factor = 5.0  # Factor to increase contrast
    gamma = 1.2 # gamma factor correction
    process_directory(image_path, json_path, threshold, contrast_factor, gamma)

if __name__ == "__main__":
    main()
