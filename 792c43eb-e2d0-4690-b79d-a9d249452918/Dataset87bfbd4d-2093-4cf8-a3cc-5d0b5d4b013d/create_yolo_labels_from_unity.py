import json

'''
This script is used to convert JSON labels from the Unity Perception package to
YOLO darknet annotation format.

The only input taken here is the json file. Its path and the picture sizes
MUST BE HARDCODED. If there are various json files just run the program as many
times changing the file name. The output will fall on the directory you are
calling this script from.
'''

pic_height = 315
pic_width = 640

with open('./captures_013.json', 'r') as file:
    big_json_file = json.load(file)

for picture in big_json_file['captures']:
    filename = picture['filename'].split('/')[-1]
    filename = filename[:-4] + '.txt'

    with open(filename, 'w') as annotation_file:
        for bbox in picture['annotations'][0]['values']:
            annotation_file.write(
                '%d %f %f %f %f\n' % (
                    bbox['label_id'] - 1,
                    (bbox['x'] + bbox['width'] /2)  / pic_width,
                    (bbox['y'] + bbox['height']/2)  / pic_height,
                    bbox['width']                   / pic_width,
                    bbox['height']                  / pic_height
                    )
                )