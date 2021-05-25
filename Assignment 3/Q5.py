# -*- coding: utf-8 -*-
"""Attention_SK_2405S.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SLUg08mTMpZ55KHUVwO23d8vKTkoMwhf
"""

from google.colab import drive
drive.mount('/content/drive/')

import tarfile
my_tar = tarfile.open('/content/drive/MyDrive/dakshina_dataset_v1.0.tar')
my_tar.extractall('./RNN folder') # specify which folder to extract to
my_tar.close()

!pip install wandb

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from keras.models import Model
from keras.layers import Input, LSTM, Dense, RNN, GRU, SimpleRNN
import math
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import os
import io
import time

wandb.login(key = 'd1ceb7eee4b7974dd2a8169c606b3c0d311c3276')

train = pd.read_csv('/content/RNN folder/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv',delimiter="\t",header=None,names = ['hindi', 'english', 'number'])
val = pd.read_csv('/content/RNN folder/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv',delimiter="\t",header=None,names = ['hindi', 'english', 'number'])
test = pd.read_csv('/content/RNN folder/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv',delimiter="\t",header=None,names = ['hindi', 'english', 'number'])

train.head()
val.head()
test.head()

train.dropna(inplace = True)

df = pd.concat([train, val, test])
df['english'] = df['english'].apply(lambda x:'<start> '+ x.replace("", " ")[1: -1] + " <end>")
df['hindi'] = df['hindi'].apply(lambda x: "<start> " + x.replace("", " ")[1: -1] + " <end>")
df.drop(columns=['number'],inplace=True)
df

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer

def load_dataset(data):
  # creating cleaned input, output pairs
  targ_lang, inp_lang = list(data['hindi']),list(data['english'])

  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(df)
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

input_tensor_train   = input_tensor[:train.shape[0]]
target_tensor_train  = target_tensor[:train.shape[0]]
input_tensor_val     = input_tensor[train.shape[0]:train.shape[0]+val.shape[0]]
target_tensor_val    = target_tensor[train.shape[0]:train.shape[0]+val.shape[0]]
input_tensor_test    = input_tensor[train.shape[0]+val.shape[0]:]
target_tensor_test   = target_tensor[train.shape[0]+val.shape[0]:]

print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val),len(input_tensor_test ),len(target_tensor_test ))

def convert(lang, tensor):
  for t in tensor:
    if t != 0:
      print(f'{t} ----> {lang.index_word[t]}')

print("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print()
print("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])

BUFFER_SIZE = len(input_tensor_train)

vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, batch_sz, latent_dims, dropout = 0, cell_type = 'LSTM'):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.encoder_states = []
    self.Outputs = []
    self.latent_dims = latent_dims
    self.cell_type = cell_type
    self.enc_layers = [3] * len(latent_dims)
    for j in range(len(latent_dims))[::-1]:
      if self.cell_type == 'LSTM':
        self.enc_layers[j] = tf.keras.layers.LSTM(latent_dims[j], return_state = True, return_sequences = True, recurrent_initializer='glorot_uniform', dropout = dropout)   
      
      elif self.cell_type == 'GRU':
        self.enc_layers[j] = tf.keras.layers.GRU(latent_dims[j], return_state = True, return_sequences = True,recurrent_initializer='glorot_uniform', dropout = dropout)

      elif self.cell_type == 'RNN':
        self.enc_layers[j] = tf.keras.layers.SimpleRNN(latent_dims[j], return_state = True, return_sequences = True, recurrent_initializer='glorot_uniform', dropout = dropout)

  def call(self, x):
    x = self.embedding(x)
    for j in range(len(self.latent_dims))[::-1]:
      if self.cell_type == 'LSTM':
        outp, h, c = self.enc_layers[j](x)
        self.encoder_states += [h, c]
        self.Outputs += [outp]
      
      elif self.cell_type == 'GRU':
        outp, h = self.enc_layers[j](x)
        self.encoder_states += [h]
        self.Outputs += [outp]

      elif self.cell_type == 'RNN':
        outp, h = self.enc_layers[j](x)
        self.encoder_states += [h]
        self.Outputs += [outp]

    output, state = self.Outputs, self.encoder_states
    return output, state

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)

    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, batch_sz, latent_dims, dropout = 0, cell_type = 'LSTM'):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.latent_dims = latent_dims
    self.cell_type = cell_type
    self.dec_layers = [3] * len(latent_dims)
    for j in range(len(latent_dims)):
      if self.cell_type == 'LSTM':
        self.dec_layers[j] = tf.keras.layers.LSTM(latent_dims[len(latent_dims) - j - 1], return_state = True, return_sequences = True, recurrent_initializer='glorot_uniform', dropout = dropout)   
      
      elif self.cell_type == 'GRU':
        self.dec_layers[j] = tf.keras.layers.GRU(latent_dims[len(latent_dims) - j - 1], return_state = True, return_sequences = True,recurrent_initializer='glorot_uniform', dropout = dropout)

      elif self.cell_type == 'RNN':
        self.dec_layers[j] = tf.keras.layers.SimpleRNN(latent_dims[len(latent_dims) - j - 1], return_state = True, return_sequences = True, recurrent_initializer='glorot_uniform', dropout = dropout)

    
    self.fc = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(latent_dims[0])

  def call(self, x, enc_output, encoder_states):
    context_vector, attention_weights = self.attention(encoder_states[-1], enc_output)

    x = self.embedding(x)

    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    decoder_states = []
    outputs = x

    output_layers = []
    for j in range(len(self.latent_dims)):
      if self.cell_type == 'LSTM':
        output_layers.append(
            self.dec_layers[j]
        )
        outputs, dh, dc = output_layers[-1](outputs)
        decoder_states += [dh, dc]

      elif self.cell_type == 'GRU':
        output_layers.append(
            self.dec_layers[j]
        )
        outputs, dh = output_layers[-1](outputs)
        decoder_states += [dh]

      elif self.cell_type == 'RNN':
        output_layers.append(
            self.dec_layers[j]
        )
        outputs, dh = output_layers[-1](outputs)
        decoder_states += [dh]

    output = tf.reshape(outputs, (-1, outputs.shape[2]))
    x = self.fc(output)

    return x, decoder_states, attention_weights

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


