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
from keras import layers
from keras import models
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

def resnet_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x
    # add 1x1 convolution to shortcut branch to match shape with residual block
    shortcut = layers.Conv2D(filters, 1, strides=stride, name=name + '_skip_conv')(shortcut)

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False, name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def ResNet50(input_shape=(224, 224, 3), classes=num_classes):
    img_input = layers.Input(shape=input_shape)

    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = resnet_block(x, filters=64, conv_shortcut=False, name='conv2_block1')
    x = resnet_block(x, filters=64, name='conv2_block2')
    x = resnet_block(x, filters=64, name='conv2_block3')

    x = resnet_block(x, filters=128, stride=2, conv_shortcut=False, name='conv3_block1')
    x = resnet_block(x, filters=128, name='conv3_block2')
    x = resnet_block(x, filters=128, name='conv3_block3')
    x = resnet_block(x, filters=128, name='conv3_block4')

    x = resnet_block(x, filters=256, stride=2, conv_shortcut=False, name='conv4_block1')
    x = resnet_block(x, filters=256, name='conv4_block2')
    x = resnet_block(x, filters=256, name='conv4_block3')
    x = resnet_block(x, filters=256, name='conv4_block4')
    x = resnet_block(x, filters=256, name='conv4_block5')
    x = resnet_block(x, filters=256, name='conv4_block6')

    x = resnet_block(x, filters=512, stride=2, conv_shortcut=False, name='conv5_block1')
    x = resnet_block(x, filters=512, name='conv5_block2')
    x = resnet_block(x, filters=512, name='conv5_block3')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='fc1000')(x)

    inputs = img_input
    # Create model.
    model = tf.keras.Model(inputs, x, name='resnet50')

    return model

os.chdir("..")
os.chdir("..")

# Define the model
model = ResNet50(input_shape=(img_height, img_width, 3), classes=num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

y = np_utils.to_categorical(y)
yv = np_utils.to_categorical(yv)
train_ds = np.array(train_ds)
train_ds = train_ds.astype('float32') / 255.0
val_ds = np.array(val_ds)
val_ds = val_ds.astype('float32') / 255.0

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpointsres\checkpt",
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model
model.fit(train_ds, y, epochs=50, batch_size=32, validation_data=(val_ds, yv), callbacks=[cp_callback])

model.save('resdogmodel.h5', include_optimizer=True)