import zipfile as zf
files = zf.ZipFile("mobilenet_v1.zip", 'r')
files.extractall('mobilenet_v1')
files.close()

from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout,Flatten
import matplotlib.pyplot as plt
import cv2

from mobilenet_v1.mobilenet import MobileNet

from tensorflow.keras.applications.mobilenet import preprocess_input

# the parameters
IMAGE_SIZE = 224
ALPHA=0.75
EPOCHS=20

def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


# function to define dropout, hidden layers and the number of output
def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

# Using MobileNetv1
base_model=MobileNet(input_shape=(IMAGE_SIZE, IMAGE_SIZE,3), alpha = ALPHA,
                     depth_multiplier = 1, dropout = 0.001, include_top = False,
                     weights = "imagenet", classes = 4, backend=keras.backend,
                     layers=keras.layers,models=keras.models,utils=keras.utils)


import zipfile as zf
files = zf.ZipFile("3D_Shapes.zip", 'r')
files.extractall('3D_Shapes_Dataset')
files.close()

FC_LAYERS = [100, 50]
dropout = 0.5

finetune_model = build_finetune_model(base_model,
                                      dropout=dropout,
                                      fc_layers=FC_LAYERS,
                                      num_classes=4)

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory('3D_Shapes_Dataset',
                                                 target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical', shuffle=True)

finetune_model.summary()
finetune_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
step_size_train=train_generator.n//train_generator.batch_size
history = finetune_model.fit_generator(generator=train_generator,steps_per_epoch=step_size_train,epochs=EPOCHS, shuffle=True)

finetune_model.save('shape_model.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'accuracy'], loc='upper left')
plt.show()

def predict_shape(img_path):
#img_path="3D_Shapes_Dataset/cube/00000963.jpg"
    preprocessed_image = prepare_image(img_path)
    predictions_shape = finetune_model.predict(preprocessed_image)
    labels=['Cube','Cylinder','Spheroid','Sphere']
    #print("Input Image :")
    img=cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    print("Shape Detected: ", labels[predictions_shape[0].tolist().index(max(predictions_shape[0]))])

img_path= "../3D_Shapes_Dataset/cube/00000963.jpg"
predict_shape(img_path)