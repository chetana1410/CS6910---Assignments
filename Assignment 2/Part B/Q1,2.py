from google.colab import drive
drive.mount("/content/gdrive")

! pip install wandb

!nvidia-smi

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, Activation, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import CategoricalCrossentropy
import PIL
import wandb
from wandb.keras import WandbCallback
from keras.callbacks import EarlyStopping
from keras import optimizers

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import SGD, Adam, RMSprop

wandb.login()

def train():
  default_hyperparams = dict(
      tr_model = 'VGG16',
      learning_rate=0.01,
      optimizer = 'Adam',
      freeze = 0.5,
      epochs = 5,
      data_aug = 0,
      batch_size = 32,
      dp=0.2
  )    
    
  wandb.init(config = default_hyperparams)
  config = wandb.config

  train_data_dir= '/content/gdrive/MyDrive/inaturalist_12K/train'

  batch_size = config.batch_size

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

  if config.tr_model == 'VGG16':
    trained_model = VGG16(input_shape = [224, 224] + [3], weights='imagenet', include_top=False)

  elif config.tr_model == 'InceptionV3':
    trained_model = InceptionV3(input_shape = [299, 299] + [3], weights='imagenet', include_top=False)
  
  elif config.tr_model == 'ResNet50':
    trained_model = ResNet50(input_shape = [224, 224] + [3], weights='imagenet', include_top=False)

  elif config.tr_model == 'Xception':
    trained_model = Xception(input_shape = [299, 299] + [3], weights='imagenet', include_top=False)

  elif config.tr_model == 'InceptionResNetV2':
    trained_model = InceptionResNetV2(input_shape = [299, 299] + [3], weights='imagenet', include_top=False)

  if config.freeze == -1:
    for layer in trained_model.layers:
      layer.trainable = False
  
  else:
    if config.tr_model == 'VGG16':
      for layer in trained_model.layers[:16]:
        layer.trainable = False

      for layer in trained_model.layers[16:]:
        layer.trainable = True

    elif config.tr_model == 'InceptionV3':
      for layer in trained_model.layers[:229]:
        layer.trainable = False

      for layer in trained_model.layers[229:]:
        layer.trainable = True

    elif config.tr_model == 'ResNet50':
      for layer in trained_model.layers[:171]:
        layer.trainable = False

      for layer in trained_model.layers[171:]:
        layer.trainable = True

    elif config.tr_model == 'Xception':
      for layer in trained_model.layers[:129]:
        layer.trainable = False

      for layer in trained_model.layers[129:]:
        layer.trainable = True

    elif config.tr_model == 'InceptionResNetV2':
      for layer in trained_model.layers[:777]:
        layer.trainable = False

      for layer in trained_model.layers[777:]:
        layer.trainable = True

  x = trained_model.output
  x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(10, activation='softmax')(x)

# this is the model we will train
  model = Model(inputs = trained_model.input, outputs = predictions)
 

  if config.optimizer == 'Adam':
    if config.learning_rate == 0:
      model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
      )  
    
    else:
      model.compile(optimizer=Adam(lr = config.learning_rate, decay = 1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

  if config.optimizer == 'rmsprop':
    if config.learning_rate == 0:
      model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
      )  
    
    else:
      model.compile(optimizer=RMSprop(lr = config.learning_rate, decay = 1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

  if config.optimizer == 'sgd':
    if config.learning_rate == 0:
      model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
      )  
    
    else:
      model.compile(optimizer=SGD(lr = config.learning_rate, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

  if config.tr_model == 'VGG16' or config.tr_model == 'ResNet50':
    train_it = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training') # set as training data

    val_it = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data


  else:
    train_it = train_datagen.flow_from_directory(
      train_data_dir,
      target_size=(299, 299),
      batch_size=batch_size,
      class_mode='categorical',
      subset='training') # set as training data

    val_it = train_datagen.flow_from_directory(
      train_data_dir, # same directory as training data
      target_size=(299, 299),
      batch_size=batch_size,
      class_mode='categorical',
      subset='validation') # set as validation data

  model.fit(
  train_it,
  steps_per_epoch = train_it.samples //batch_size,
  validation_data = val_it, 
  validation_steps = val_it.samples // batch_size,
  epochs = config.epochs,
  callbacks=[WandbCallback(data_type='image',validation_data = val_it, verbose=1), EarlyStopping(patience=10, restore_best_weights=True)])


  
sweep_config = {
   #'program': train(),
    'method': 'bayes',         
    'metric': {
        'name': 'val_accuracy',     
        'goal': 'maximize'      
    },
    'parameters': {
        'learning_rate': {
            'values': [0, 1e-3, 5*1e-4, 1e-4]   # 0 implies default learning rate set by model
        },
        'freeze': {
            'values': [-1, 5, 10]
        },
        'epochs': {
            'min': 5,
            'max': 20
        },
        'data_aug': {
            'values': [0,1]
        },
        'optimizer' : {
            'values': ['Adam', 'rmsprop', 'SGD']
        },
        'tr_model' : {
            'values': ['VGG16', 'InceptionV3', 'ResNet50', 'Xception', 'InceptionResNetV2']
        },
        'batch_size': {
            'values': [32, 64]
        }
        }
    }

sweep_id = wandb.sweep(sweep_config, project='skQ2')
wandb.agent(sweep_id, function=train)
