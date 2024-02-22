import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
import tensorflow as tf
import tensorflow as tf
import keras
from keras.layers import Conv2D,MaxPooling2D,Dropout
from tensorflow.keras import backend as K
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

tf.config.set_visible_devices([], 'GPU')
print(tf.config.list_physical_devices())

dataset_path = "Brain_Dataset"
images_path = "Brain_Dataset/data/images"
masks_path = "Brain_Dataset/data/masks"
target_shape = (256,256)

def preprocess_dataset(images_path,masks_path):
    images = []
    masks = []

    for image_file in os.listdir(images_path):
        image_path = os.path.join(images_path, image_file)
        if os.path.isfile(image_path):
            image = cv2.imread(image_path, 0) 
            image = cv2.resize(image, target_shape) 
            images.append(image)

    for mask_file in os.listdir(masks_path):
        mask_path = os.path.join(masks_path, mask_file)
        if os.path.isfile(mask_path):
            mask = cv2.imread(mask_path, 0) 
            mask = cv2.resize(mask, target_shape) 
            masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    images = images/255.0
    masks = masks/255.0

    return images,masks

images,masks = preprocess_dataset(images_path,masks_path)


def UNet(input_shape):
    inputs = Input(input_shape)
    
    # Contracting path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Expanding path
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    up1 = UpSampling2D(size=(2, 2))(conv2)
    up1 = Conv2D(1, 1, activation='sigmoid')(up1)
    
    model = Model(inputs=inputs, outputs=up1)
    return model


model = UNet(input_shape=(256, 256, 1))
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
model.fit(images, masks, epochs=10, batch_size=8, validation_split=0.2)
