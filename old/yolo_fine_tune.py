import torch
import torchvision
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)

#imgs = ['https://ultralytics.com/images/zidane.jpg']
imgs = Image.open('../random.jpg')

results = model(imgs)

#results.show()
results.print()

#results.xyxy[0]
#results.pandas().xyxy[0]