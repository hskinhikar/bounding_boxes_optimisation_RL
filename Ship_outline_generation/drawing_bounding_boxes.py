
"""
This script displays the initial prediction by Mask R-CNN model on individual images within the dataset.

The threshold represents the confience the model has in the predictions it generated. Although a threshold of 0.9 gives very good results,
we want to see how reinforcement learning improves this model, so from here on outwards, the Mask R-CNN will have a threshold below 0.5 
(potentially even less than 0.1, to make sure we have predictions for every dataset)

"""



import torch
import torchvision
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import cv2 
from PIL import Image
import numpy as np

#loading pretrained Faster R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

#Load and preprocess the image
image_path = r"C:\Users\shiri\Desktop\Dataverze\Vision_experimentation\shipsnet\shipsnet\0__20170106_180851_0e30__-122.33500527461648_37.728100492590634.png"
image = Image.open(image_path).convert("RGB")
image_tensor = F.to_tensor(image).unsqueeze(0)

#perform object detection
with torch.no_grad():
    predictions = model(image_tensor)

#Get bounding boxes and labels
pred_boxes = predictions[0]['boxes'].cpu().numpy()
pred_labels = predictions[0]['labels'].cpu().numpy()
pred_masks = predictions[0]['masks'].cpu().numpy()

#threshold masks
threshold = 0.9
masks = pred_masks > threshold


#draw masks on the image
image_np = cv2.imread(image_path)
for i in range(len(masks)):
    mask = masks[i,0]
    contours, _=cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(image_np, [contour], -1, (0,255,0), 2)

#display the image with the bounding boxes
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()