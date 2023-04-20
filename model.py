from numpy import mean
from numpy import std
from tensorflow import keras
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
import keras.layers as layers
from keras.optimizers import SGD
from keras.layers import *
from keras.models import * 
from keras.preprocessing import image
from keras.utils import np_utils
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import pathlib
import os
import cv2

img_orig = []
breeds = []
y = []
train_ds = []
os.chdir("dog_dataset\\train") # change current path to dog_dataset folder
# loop through all subdirectories in the current path
for subdir in os.listdir():
    if os.path.isdir(subdir):
        breeds.append(subdir) # store subdirectory name in the breeds list
        img_subdir = [] # create a list to store images in each subdirectory
        # loop through all image files in the subdirectory
        for file in os.listdir(subdir):
            if file.endswith(".jpg"): # check if file is an image
                # read image file using OpenCV
                img = cv2.imread(os.path.join(subdir, file))
                img_subdir.append(img) # store image in img_subdir list
                train_ds.append(img)
                y.append(breeds.index(subdir))
        img_orig.append(img_subdir) # store img_subdir list in img_orig list

os.chdir("..\\valid") # change current path to parent directory

class_names = breeds
print(f"Class names: {class_names}")

img_origValid = []
val_ds = []
yv = []
# loop through all subdirectories in the current path
for subdir in os.listdir():
    if os.path.isdir(subdir):
        img_subdir = [] # create a list to store images in each subdirectory
        # loop through all image files in the subdirectory
        for file in os.listdir(subdir):
            if file.endswith(".jpg"): # check if file is an image
                # read image file using OpenCV
                img = cv2.imread(os.path.join(subdir, file))
                img_subdir.append(img) # store image in img_subdir list
                val_ds.append(img)
                yv.append(breeds.index(subdir))

        img_origValid.append(img_subdir) # store img_subdir list in img_orig list

batch_size = 700
img_height = img.shape[0]
img_width = img.shape[1]

# print(f"train_ds: {train_ds}")

# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
'''
# Andy Model

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

'''


# Marc Model

def CNN_model(num_classes, batch_size):
    model = Sequential()
    model.add(Rescaling(1./255, input_shape=(img_height, img_width, 3)))
    model.add(Conv2D(filters=32,
                    kernel_size=(3,3),
                    padding='valid',
                    data_format='channels_last',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation="softmax"))
    
    # normalize the input data
    model.add(Normalization())

    # modify the input shape to include the batch size
    model.build((None, img_height, img_width, 3))
    
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model

os.chdir("..")
os.chdir("..")

model = CNN_model(num_classes, batch_size=batch_size)
model.summary()
print(f"image shape: {train_ds[len(train_ds)-1].shape}")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints\checkpt",
                                                 save_weights_only=True,
                                                 verbose=1)
y = np_utils.to_categorical(y)
yv = np_utils.to_categorical(yv)
train_ds = np.array(train_ds)
print(train_ds.shape)
print(y.shape)
epochs=10
_ = model.fit(
    train_ds,
    y,
    epochs=epochs,
    callbacks=[cp_callback]
)

model.save('dogmodel.h5', include_optimizer=True)
