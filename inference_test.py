import torch
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

images_dir = '/home/karthikm/robot-perception/datasets/shapes/images/test'
labels_dir = '/home/karthikm/robot-perception/datasets/shapes/labels/test'
model_file = '/home/karthikm/robot-perception/robot-perception-project/model/best.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_file)

for file in os.listdir(images_dir):
    path = images_dir + '/' + file
    print(path)
    img = Image.open(path)
    imgs = [img]
    results = model(imgs, size=640)
    # Prediction
    print(results.xyxy[0])

    #plt.imshow(img)
