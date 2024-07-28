import json
import os
import numpy as np
from PIL import Image
from labelme import utils as labelme_utils

def labelme2coco(labelme_json_dir, output_json):
    label_files = [f for f in os.listdir(labelme_json_dir) if f.endswith('.json')]
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "ship"}]
    }
    annotation_id = 1

    for image_id, label_file in enumerate(label_files):
        label_path = os.path.join(labelme_json_dir, label_file)
        with open(label_path) as f:
            data = json.load(f)

        img_path = os.path.join(labelme_json_dir, data['imagePath'])
        if not os.path.exists(img_path):
            print(f"Image path {img_path} does not exist. Skipping this file.")
            continue
        
        img = Image.open(img_path)
        width, height = img.size

        coco_output['images'].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": data['imagePath']
        })

        for shape in data['shapes']:
            points = shape['points']
            bbox = [min([p[0] for p in points]), min([p[1] for p in points]),
                    max([p[0] for p in points]) - min([p[0] for p in points]),
                    max([p[1] for p in points]) - min([p[1] for p in points])]
            segmentation = [list(np.asarray(points).flatten())]

            coco_output['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "segmentation": segmentation,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            annotation_id += 1

    with open(output_json, 'w') as f:
        json.dump(coco_output, f, indent=4)

labelme_json_dir = r'C:\Users\shiri\Desktop\Dataverze\Vision_experimentation\only_ships_images'
output_json = 'output_coco.json'
labelme2coco(labelme_json_dir, output_json)
