import os
import json

def get_image_ids_from_annotations(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return {image['file_name']: image['id'] for image in data['images']}

def main(image_directory, json_annotations_file):
    annotation_ids = get_image_ids_from_annotations(json_annotations_file)
    
    all_images = os.listdir(image_directory)
    
    missing_annotations = []
    for image_file in all_images:
        if image_file not in annotation_ids:
            missing_annotations.append(image_file)
    
    if missing_annotations:
        print(f"Images without annotations: {len(missing_annotations)}")
        for image in missing_annotations:
            print(image)
    else:
        print("All images have corresponding annotations.")

if __name__ == "__main__":
    image_directory = r"C:\Users\shiri\Desktop\Dataverze\Vision_experimentation\only_ships_images\images"
    json_annotations_file = r"C:\Users\shiri\Desktop\Dataverze\Vision_experimentation\RL_attempt_3\output_coco_valid.json"
    main(image_directory, json_annotations_file)