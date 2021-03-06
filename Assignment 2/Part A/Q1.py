# -*- coding: utf-8 -*-
"""Q1 (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bcj6-3fl2i79bj_Udu5XN60BCiuG4sYH
"""

from google.colab import drive
drive.mount("/content/gdrive")

! pip install wandb

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,BatchNormalization,Dropout,Activation ,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import CategoricalCrossentropy
import PIL
import wandb
from wandb.keras import WandbCallback
from keras.callbacks import EarlyStopping
from keras import optimizers

wandb.login()

def cnn(n,num_filters,ker_size_input,ker_size,activ,pool,num_nodes,filter_org,bn_vs_dp,bn=0,dp='no'):
  

  model = Sequential()
  model.add(Conv2D(num_filters, (ker_size_input, ker_size_input), input_shape=(224,224,3)))
  model.add(Activation(activ))
  model.add(MaxPooling2D(pool_size=(pool, pool)))

  for i in range(1,n):
  
    if filter_org=='same':
      num_filters= num_filters
    elif filter_org=='double':
      num_filters*=2
    else:
      num_filters//=2
      
        
    model.add(Conv2D(num_filters, (ker_size, ker_size)))

    if bn:
        model.add(BatchNormalization())
        
    model.add(Activation(activ))
    model.add(MaxPooling2D(pool_size=(pool, pool)))
      
  model.add(Flatten())
  model.add(Dense(num_nodes))
  
  if bn_vs_dp == 'dp':
    model.add(Dropout(dp))
  else:
    model.add(BatchNormalization())

  model.add(Activation(activ))
      
  model.add(Dense(10, activation='softmax'))

  return model

def train():
  default_hyperparams = dict(
    bn=0,
    num_filters=64,
    fliter_org='same',
    dropout=0,
    data_aug=0,
    learning_rate=0.01,
    epochs=5,
    activ="ReLU",
    optimizer = 'Adam',
    batch_size=32,
    ker_size_input=5,
    bn_vs_dp='bn',
    max_vs_avg='max'
  )

    
    
  wandb.init(config = default_hyperparams)
  config = wandb.config

  train_data_dir= '/content/gdrive/MyDrive/inaturalist_12K/train'

  batch_size=config.batch_size

  if config.data_aug:
      
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45, 
    width_shift_range=.15, 
    height_shift_range=.15, 
    horizontal_flip=True, 
    zoom_range=0.5,
    validation_split=0.1) 

  else:
      
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1) # set validation split
  
    

  train_it = train_datagen.flow_from_directory(
      train_data_dir,
      target_size=(224,224),
      batch_size=batch_size,
      class_mode='categorical',
      subset='training') # set as training data

  val_it = train_datagen.flow_from_directory(
      train_data_dir, # same directory as training data
      target_size=(224,224),
      batch_size=batch_size,
      class_mode='categorical',
      subset='validation') # set as validation data
  
  # Your model here ...
  model = cnn(5,config.num_filters,config.ker_size_input,3,config.activ,2,1024,config.filter_org,config.bn_vs_dp,config.bn,config.dropout)

  if config.optimizer == 'Adam':
    model.compile(optimizers.Adam(lr=config.learning_rate, decay=1e-6), loss=CategoricalCrossentropy(), metrics='acc')
  elif config.optimizer == 'rmsprop':
    model.compile(optimizers.RMSprop(lr=config.learning_rate, decay=1e-6), loss=CategoricalCrossentropy(), metrics='acc')
  else:
     model.compile(optimizers.SGD(lr=config.learning_rate, decay=1e-6), loss=CategoricalCrossentropy(), metrics='acc')
  
  model.fit(
  train_it,
  steps_per_epoch = train_it.samples //batch_size,
  validation_data = val_it, 
  validation_steps = val_it.samples // batch_size,
  epochs = config.epochs,
  callbacks=[WandbCallback(data_type='image',validation_data = val_it,verbose=1),EarlyStopping(patience=10,restore_best_weights=True)])
  
sweep_config = {
   #'program': train(),
    'method': 'bayes',         
    'metric': {
        'name': 'val_accuracy',     
        'goal': 'maximize'      
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-3,1e-4]
        },
        'activ': {
            'values': ['relu']
        },
        'bn': {
            'values': [0,1]
        },
        'num_filters': {
            'values': [32, 64, 128]
        },
        'filter_org': {
            'values': ['double', 'same']
        },
        'epochs': {
            'min': 5,
	    'max': 30
        },
        'dropout': {
            'values': [0, 0.2, 0.3]
        },
        'data_aug': {
            'values': [0,1]
        },
        'optimizer' : {
            'values': ['Adam','rmsprop','sgd']
        },

        'batch_size': {
            'values': [32, 64]
        },
        'ker_size_input':{
            'values':[3, 5, 7]
        },
        'bn_vs_dp':{
            'values':['bn', 'dp']
        },
        'max_vs_avg': {
            'values':['max', 'avg']
        }

        }
    }


sweep_id = wandb.sweep(sweep_config,project='asgn2 q1 1')

wandb.agent(sweep_id, function=train)
