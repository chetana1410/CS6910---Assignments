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

