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

os.chdir("..\\valid")

class_names = breeds

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

batch_size = 32
img_height = img.shape[0]
img_width = img.shape[1]

num_classes = len(class_names)
print(f"Number of classes: {num_classes}")

os.chdir("..")
os.chdir("..")

y = np_utils.to_categorical(y)
yv = np_utils.to_categorical(yv)
train_ds = np.array(train_ds)
train_ds = train_ds.astype('float32') / 255.0
val_ds = np.array(val_ds)
val_ds = val_ds.astype('float32') / 255.0

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpointsinc\checkpt",
                                                 save_weights_only=True,
                                                 verbose=1)


IMG_SHAPE = (224,224,3)
base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                               include_top=False, 
                                               weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
  base_model,
  #tf.keras.layers.Conv2D(128, 3, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])
model._name = "InceptionV3Dogs"

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), #Adam(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(train_ds, y, epochs=10, validation_data=(val_ds, yv), callbacks=[cp_callback])

model.save('incdogmodel.h5', include_optimizer=True)