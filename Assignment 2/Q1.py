#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,BatchNormalization,Dropout
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


def cnn(ker_size,activ,pool,num_nodes):

    model = Sequential()
    model.add(Conv2D(num_filters[0], (ker_size[0], ker_size[0]), input_shape=input_shape))
    model.add(Activation(activ[0]))
    model.add(MaxPooling2D(pool_size=(pool[0], pool[0])))
    for i in range(1,n):
        model.add(Conv2D(num_filters[i], (ker_size[i], ker_size[i])))
        #model.add(BatchNormalization())
        model.add(Activation(activ[i]))
        model.add(MaxPooling2D(pool_size=(pool[i], pool[i])))
    model.Flatten()
    model.add(Dense(num_nodes, activation=activ[n+1]))
   
    #model.add(keras.layers.Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


  

    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    dtype=None,
)
# In[ ]:


# load datasets

# create generator
datagen = ImageDataGenerator()
# prepare an iterators for each dataset
train_it = datagen.flow_from_directory('data/train/', class_mode='categorical')
#val_it = datagen.flow_from_directory('data/validation/', class_mode='binary')
test_it = datagen.flow_from_directory('data/test/', class_mode='categorical')
# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

