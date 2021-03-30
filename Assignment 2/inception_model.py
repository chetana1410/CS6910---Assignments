# -*- coding: utf-8 -*-
"""Inception Model

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O3aiFG_u8_73JB5UEUazWV-nzTiqy30_
"""

!pip install tensorflow-gpu

!nvidia-smi

from google.colab import drive
drive.mount("/content/gdrive")

!pip install wandb

# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
#import wandb
#from wandb.keras import WandbCallback
#import matplotlib.pyplot as plt

wandb.login()

IMAGE_SIZE = [224, 224]
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in inception.layers:
  layer.trainable = False

x = Flatten()(inception.output)
prediction = Dense(10, activation='softmax')(x)

# create a model object
model = Model(inputs=inception.input, outputs=prediction)

model.summary()

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45, 
    width_shift_range=.15, 
    height_shift_range=.15, 
    horizontal_flip=True, 
    zoom_range=0.5,
    validation_split=0.1)

val_datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.9)

train_data_dir= '/content/gdrive/MyDrive/inaturalist_12K/train'

train_it = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=128,
    class_mode='categorical',
    subset='training'
    ) # set as training data

val_it = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(224, 224),
    batch_size=128,
    class_mode='categorical',
    subset='validation'
    )

r = model.fit_generator(
  train_it,
  steps_per_epoch = train_it.samples // 128,
  validation_data = val_it, 
  validation_steps = val_it.samples // 128,
  epochs=10,
)

test_datagen = ImageDataGenerator(rescale=1./255.)
test_data_dir = '/content/gdrive/MyDrive/inaturalist_12K/val'
test_it = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

import numpy as np
np.random.seed(42)

true_batch=[]
true_label = []
for i in range(2000):
  a,b = test_it.next()
  true_class_indices = np.argmax(b, axis=1)
  true_batch.append(a)
  true_label.append(true_class_indices)

accuracy = 0
for i in range(2000):
  predict = model.predict(true_batch[i])
  predicted_class_indices = np.argmax(predict,axis=1)
  accuracy += int(predicted_class_indices == true_label[i])

print('correctly identified images', accuracy)
print('accuracy', accuracy * 100 / 2000)
