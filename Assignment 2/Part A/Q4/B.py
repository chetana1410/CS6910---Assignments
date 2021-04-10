#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount("/content/gdrive")


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, Activation, MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import CategoricalCrossentropy
import PIL
#import wandb
#from wandb.keras import WandbCallback
from keras.callbacks import EarlyStopping
from keras import optimizers

from keras.optimizers import SGD, Adam, RMSprop


# In[3]:


from PIL import Image
import os
from skimage import io
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


def cnn(n,num_filters,ker_size_input,ker_size,activ,pool,num_nodes,filter_org,max_vs_avg,bn=0,dp='no'):  

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

    if i>=3 and max_vs_avg == 'avg':

      model.add(AveragePooling2D(pool_size=(pool, pool)))

    else:
        
    
      model.add(MaxPooling2D(pool_size=(pool, pool)))
      
  model.add(Flatten())
  model.add(Dense(num_nodes))
  
  if dp:
    model.add(Dropout(dp))
  else:
    model.add(BatchNormalization())

  model.add(Activation(activ))
      
  model.add(Dense(10, activation='softmax'))

  return model


# In[5]:


# active - relu, batch size = 64, bn = 0, data_aug = 0, dropout = 0.2, epoch = 10, double, ker_size = 7, lr = 1e-4, avg, num_filt = 64, Adam
#cnn(n,num_filters,ker_size_input,ker_size,activ,pool,num_nodes,filter_org,max_vs_avg,bn=0,dp='no')
num_filters = 64
ker_size_input = 7
filter_org = 'double'
max_vs_avg = 'avg'
bn = 0
data_aug = 0
dp = 0.2
batch_size = 64
iters = 10


# In[6]:


if data_aug:
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
    validation_split=0.1) 


# In[7]:


train_data_dir= '/content/gdrive/MyDrive/inaturalist_12K/train'

# active - relu, batch size = 64, bn = 1, data_aug = 0, dropout = 0.3, epoch = 5, double, ker_size = 3, lr = 1e-4, avg, num_filt = 32, Adam

train_it = train_datagen.flow_from_directory(
      train_data_dir,
      target_size=(224,224),
      batch_size = batch_size,
      class_mode='categorical',
      subset='training') # set as training data

val_it = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(224,224),
    batch_size = batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data


# In[8]:


model = cnn(5, num_filters, ker_size_input, 3, 'relu', 2, 1024, filter_org, max_vs_avg, bn, dp)
model.compile(optimizers.Adam(lr= 1e-4, decay=1e-6), loss=CategoricalCrossentropy(), metrics='acc')
model.fit(
  train_it,
  steps_per_epoch = train_it.samples //batch_size,
  validation_data = val_it, 
  validation_steps = val_it.samples // batch_size,
  epochs = iters,
)


# In[9]:


train_loss, train_accuracy = model.evaluate(train_it)
val_loss, val_accuracy = model.evaluate(val_it)
print('train_loss = ', train_loss, 'train_accuracy = ', train_accuracy)
print('val_loss = ', val_loss, 'val_accuracy = ', val_accuracy)


# In[41]:


# In[10]:


test_datagen = ImageDataGenerator(rescale=1./255.)
test_data_dir = '/content/gdrive/MyDrive/inaturalist_12K/val'
test_it = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224,224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)


# In[11]:


test_loss, test_accuracy = model.evaluate(test_it)
print('test_loss = ', test_loss, ', test_accuracy = ', test_accuracy)


# In[13]:


visited_class = [0 for i in range(10)]
X_batch = []
Y_batch = []
while True:  
  x_batch, y_batch = test_it.next()
  img_class = np.argmax(y_batch)
  if visited_class[img_class] < 3:
    visited_class[img_class] += 1
    X_batch.append(x_batch)
    Y_batch.append(y_batch)
  
  if visited_class.count(3) == 10:
    break


# In[14]:


class_name = {0 : 'Amphibia', 1 : 'Animilia', 2 : 'Arachnida', 3 : 'Aves', 4 : 'Fungi', 5 : 'Insecta', 6 : 'Mammalia', 7 : 'Mollusca', 8 : 'Plantae', 9 : 'Reptilia'}


# In[15]:


Y_pred = []
for i in X_batch:
  Y_pred.append(model.predict(i))


# In[19]:


from PIL import Image
import os
from skimage import io
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


def show_grid(image_list, nrows, ncols, label_list=None, pred_label = None, show_labels=False, savename=None, figsize=(15,10), showaxis='off'):
  if type(image_list) is not list:
      if(image_list.shape[-1]==1):
          image_list = [image_list[i,:,:,0] for i in range(image_list.shape[0])]
      elif(image_list.shape[-1]==3):
          image_list = [image_list[i,:,:,:] for i in range(image_list.shape[0])]
  fig = plt.figure(None, figsize,frameon=False)
  grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
                    axes_pad = 0.4,  # pad between axes in inch.
                    share_all = True,
                    )
  for i in range(nrows * ncols):
      ax = grid[i]
      ax.imshow(image_list[i], cmap='Greys_r')  # The AxesGrid object work as a list of axes.
      ax.axis('off')
      if show_labels:
          ax.set_title('True : ' + str(class_name[np.argmax(Y_batch[i])]) + ', Predicted : ' + str(class_name[np.argmax(Y_pred[i])]), fontsize = 8,fontweight='bold',pad=8)
  if savename != None:
      plt.savefig(savename,bbox_inches='tight')


# In[21]:


XXbatch = np.array(X_batch).reshape((len(X_batch), 224, 224, 3))
XXbatch.shape


# In[40]:


show_grid(XXbatch, 10, 3, label_list= Y_batch, pred_label = Y_pred, show_labels=True, figsize=(40,20))


# In[ ]:




