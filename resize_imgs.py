from PIL import Image
import os
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    dh, dw, _ = img.shape

    x_min, y_min, w, h = bbox

    l = int((x_min - w / 2) * dw)
    r = int((x_min + w / 2) * dw)
    t = int((y_min - h / 2) * dh)
    b = int((y_min + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    x_min = l
    x_max = r
    y_min = t
    y_max = b

    #x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.pause(1)
    plt.close()


# Uses files placed in root dir - both images and the bounding box text

root_dir = '/home/karthikm/robot-perception/datasets/shapes/images/test/old'
new_dir = '/home/karthikm/robot-perception/datasets/shapes/images/test/new'

transform = A.Compose([A.LongestMaxSize(max_size=640)], bbox_params=A.BboxParams(format='yolo',
                                                                                 label_fields=['class_labels']))

category_id_to_name = {'1': 'cylinder','0': 'cube'}

for curr_file in os.listdir(root_dir):
    if curr_file.endswith(".jpg"):
        # Get bbox
        f_name = curr_file.split('.')[0]
        f = open(root_dir + "/" + f_name + ".txt")
        text = f.read()
        bbox_text = text.split()
        bboxes = []
        bbox = []
        labels = []
        for text in bbox_text:
            if text == '0' or text == '1':
                labels.append(text)
            else:
                bbox.append(float(text))
        # Data needs to be list of lists
        bboxes.append(bbox)


        # Read image
        img = cv2.imread(root_dir + "/" + curr_file)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transform
        print(curr_file)
        print(bboxes)
        print(labels)
        transformed = transform(image=img, bboxes=bboxes, class_labels=labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']

        #plt.imshow(transformed_image)
        #print(transformed_bboxes[0])
        #print(transformed_class_labels)
        #visualize(transformed_image, transformed_bboxes, transformed_class_labels, category_id_to_name)
        bbox_new = transformed_class_labels + list(transformed_bboxes[0])
        print(bbox_new)

        # Save image and bbox
        cv2.imwrite(new_dir + "/" + curr_file, transformed_image)
        with open(new_dir + "/" + f_name + ".txt", 'w') as filehandle:
            for item in bbox_new:
                filehandle.write(str(item))
                filehandle.write(" ")




        #print(transformed_bboxes)








