from google.colab import drive
drive.mount("/content/gdrive")


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
from PIL import Image
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop

import os
from skimage import io
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
import math
get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.applications.resnet import  preprocess_input, decode_predictions
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

#Q4 a

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


num_filters = 64
ker_size_input = 7
filter_org = 'double'
max_vs_avg = 'avg'
bn = 0
data_aug = 0
dp = 0.2
batch_size = 64
iters = 10



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


train_data_dir= '/content/gdrive/MyDrive/inaturalist_12K/train'

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



model = cnn(5, num_filters, ker_size_input, 3, 'relu', 2, 1024, filter_org, max_vs_avg, bn, dp)
model.compile(optimizers.Adam(lr= 1e-4, decay=1e-6), loss=CategoricalCrossentropy(), metrics='acc')
model.fit(
  train_it,
  steps_per_epoch = train_it.samples //batch_size,
  validation_data = val_it, 
  validation_steps = val_it.samples // batch_size,
  epochs = iters,
)



train_loss, train_accuracy = model.evaluate(train_it)
val_loss, val_accuracy = model.evaluate(val_it)
print('train_loss = ', train_loss, 'train_accuracy = ', train_accuracy)
print('val_loss = ', val_loss, 'val_accuracy = ', val_accuracy)


test_datagen = ImageDataGenerator(rescale=1./255.)
test_data_dir = '/content/gdrive/MyDrive/inaturalist_12K/val'
test_it = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224,224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)


test_loss, test_accuracy = model.evaluate(test_it)
print('test_loss = ', test_loss, ', test_accuracy = ', test_accuracy)


#Q4 b


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


class_name = {0 : 'Amphibia', 1 : 'Animilia', 2 : 'Arachnida', 3 : 'Aves', 4 : 'Fungi', 5 : 'Insecta', 6 : 'Mammalia', 7 : 'Mollusca', 8 : 'Plantae', 9 : 'Reptilia'}


Y_pred = []
for i in X_batch:
  Y_pred.append(model.predict(i))



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

XXbatch = np.array(X_batch).reshape((len(X_batch), 224, 224, 3))
XXbatch.shape


show_grid(XXbatch, 10, 3, label_list= Y_batch, pred_label = Y_pred, show_labels=True, figsize=(35,35))



# Q4c

img_PIL = Image.open(r'/content/gdrive/MyDrive/inaturalist_12K/val/Aves/049650eac0f12d7a16c785a0f1e06e0f.jpg')
img = img_PIL.resize((224, 224))
img= keras.preprocessing.image.img_to_array(img)

filters, biases = model.layers[0].get_weights()

def forward_pass(img,f,b) :
    
    h_f, w_f, d_f = f.shape
    h_out, w_out = img.shape[0]-f.shape[0]+1,img.shape[1]-f.shape[1]+1
    
    
    output = np.zeros((218,218))
    
    for i in range(h_out):
        for j in range(w_out):
            h_start, w_start = i, j
            h_end, w_end = h_start + h_f, w_start + w_f
            output[i, j] = np.sum(img[h_start:h_end, w_start:w_end,:] *f )

    return output + b



plt.rcParams["figure.figsize"] = (30,30)
for i in range(64):
  f = filters[:, :, :, i]
  b=biases[i]
  plt.subplot(8,8, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.title(f'Filter : {i+1}',fontsize=15,fontweight='bold')
  plt.imshow(forward_pass(img,f,b))

  
H, W = 224, 224

def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    return x
  
  
 # Q5   Guided Backpropagation

def deprocess_image(x):
    
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
  
# Original image
plt.imshow(load_image('/content/gdrive/MyDrive/inaturalist_12K/val/Aves/049650eac0f12d7a16c785a0f1e06e0f.jpg', preprocess=False))
plt.axis("off")

# process example input
preprocessed_input = load_image('/content/gdrive/MyDrive/inaturalist_12K/val/Aves/049650eac0f12d7a16c785a0f1e06e0f.jpg')

@tf.custom_gradient
def guidedRelu(x):
  def grad(dy):
    return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
  return tf.nn.relu(x), grad


gb_model = Model(
    inputs = [model.inputs],
    outputs = [model.get_layer("conv2d_4").output]
)
output_shape = model.get_layer("conv2d_4").output.shape[1:]
layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer,'activation')]
for layer in layer_dict:
  if layer.activation == tf.keras.activations.relu:
    layer.activation = guidedRelu
  
  
fig , ax = plt.subplots(10,1,figsize=(30,30))
for i in range(10):
  random_neuron_idx=[0]+[np.random.randint(0,d_max-1) for d_max in output_shape]

  mask = np.zeros((1,*output_shape))
  mask[random_neuron_idx[0],random_neuron_idx[1],random_neuron_idx[2],random_neuron_idx[3]]=1
  with tf.GradientTape() as tape:
    inputs = tf.cast(preprocessed_input, tf.float32)
    tape.watch(inputs)
    outputs = gb_model(inputs)*mask

  grads = tape.gradient(outputs,inputs)[0]
  ax[i].imshow(np.flip(deprocess_image(np.array(grads)),-1))
  ax[i].set_xticks([])
  ax[i].set_yticks([])
  ax[i].set_title(f'Neuron_index :{random_neuron_idx}',fontsize=10,fontweight='bold', pad=10)


