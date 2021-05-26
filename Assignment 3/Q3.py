# -*- coding: utf-8 -*-
"""Assignment_3_vat_part1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1maWyLksy-5cKYDZ-Rc36UR5En5N-4yBk
"""

from google.colab import drive
drive.mount('/content/drive/')

import tarfile
my_tar = tarfile.open('/content/drive/MyDrive/dakshina_dataset_v1.0.tar')
my_tar.extractall('./RNN folder') # specify which folder to extract to
my_tar.close()

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from keras.models import Model
from keras.layers import Input, LSTM, Dense, RNN, GRU, SimpleRNN
from keras.layers import TimeDistributed
import math

train = pd.read_csv("/content/RNN folder/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv",delimiter="\t",header=None,names = ['hindi', 'word', 'number'])
val = pd.read_csv('/content/RNN folder/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv',delimiter="\t",header=None,names = ['hindi', 'word', 'number'])
test = pd.read_csv('/content/RNN folder/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv',delimiter="\t",header=None,names = ['hindi', 'word', 'number'])

train.head()
val.head()
test.head()

train.dropna(inplace = True)

df = pd.concat([train, val, test])
df['word'] = df['word'].apply(lambda x: x + "\n")
df['hindi'] = df['hindi'].apply(lambda x: "\t" + x + "\n")
df

def get_unique_char(dataset):
  my_set = set()
  for i in dataset:
    for char in i:
      my_set.add(char)
  return my_set

def get_req_vec(tr_word, tr_hindi):
  tr_input_char = get_unique_char(tr_word)
  tr_target_char = get_unique_char(tr_hindi)
  num_encoder_tokens = len(tr_input_char)
  num_decoder_tokens = len(tr_target_char)
  max_encoder_seq_length = max([len(txt) for txt in tr_word])
  max_decoder_seq_length = max([len(txt) for txt in tr_hindi])
  return [num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length]

def generate_dataset(input_texts, target_texts, params):
  input_characters = sorted(list(get_unique_char(input_texts)))
  target_characters = sorted(list(get_unique_char(target_texts)))

  global input_token_index 
  global target_token_index 
  global encoder_input_data

  input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
  target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

  num_encoder_tokens = params[0]
  num_decoder_tokens = params[1]
  max_encoder_seq_length = params[2] 
  max_decoder_seq_length = params[3]

  encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
  )
  decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
  )
  decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
  )

  for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index["\n"]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index["\n"]] = 1.0
    decoder_target_data[i, t:, target_token_index["\n"]] = 1.0

  return [encoder_input_data, decoder_input_data], decoder_target_data

params = get_req_vec(list(df.word), list(df.hindi))
data_X, data_Y = generate_dataset(list(df.word), list(df.hindi), params)

train_X = [data_X[0][:train.shape[0]], data_X[1][:train.shape[0]]]
train_Y = data_Y[:train.shape[0]]
val_X = [data_X[0][train.shape[0]: train.shape[0] + val.shape[0]], data_X[1][train.shape[0]: train.shape[0] + val.shape[0]]]
val_Y = data_Y[train.shape[0]: train.shape[0] + val.shape[0]]

test_X = [data_X[0][train.shape[0] + val.shape[0]:], data_X[1][train.shape[0] + val.shape[0]:]]
test_Y = data_Y[train.shape[0] + val.shape[0]:]
test_Y.shape

num_encoder_tokens = params[0]
num_decoder_tokens = params[1]
max_encoder_seq_length = params[2] 
max_decoder_seq_length = params[3]

def generate_latent_dim(hidden_layer_size, num_encode_layers):
  if num_encode_layers == 1: return [hidden_layer_size]
  elif num_encode_layers == 2:
    if hidden_layer_size == 16:
      return [16, 32]
    else:
      return [hidden_layer_size, hidden_layer_size // 2]
  else:
    if hidden_layer_size < 64:
      return [hidden_layer_size, hidden_layer_size * 2, hidden_layer_size * 4]
    else:
      return [hidden_layer_size, hidden_layer_size // 2, hidden_layer_size // 4]

def build_seq2seq_model(num_encode_layers, num_decode_layers, hidden_layer_size, dropout, beam_size, cell_type = 'LSTM'):
  global encoder_inputs 
  global encoder_states 
  global decoder_inputs 
  global latent_dims 
  global output_layers
  global time_layer

  latent_dims = [hidden_layer_size] * num_encode_layers

  encoder_inputs = Input(shape=(None, num_encoder_tokens),  name='encoder_inputs')

  outputs = encoder_inputs
  encoder_states = []
  for j in range(len(latent_dims))[::-1]:
    if cell_type == 'LSTM':
      outputs, h, c = LSTM(latent_dims[j], return_state = True, return_sequences = True, dropout = dropout)(outputs)
      encoder_states += [h, c]
    
    elif cell_type == 'GRU':
      outputs, h = GRU(latent_dims[j], return_state = True, return_sequences = True, dropout = dropout)(outputs)
      encoder_states += [h]
    elif cell_type == 'RNN':
      outputs, h = SimpleRNN(latent_dims[j], return_state = True, return_sequences = True, dropout = dropout)(outputs)
      encoder_states += [h]

  decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')

  outputs = decoder_inputs
  output_layers = []
  for j in range(len(latent_dims)):
    if cell_type == 'LSTM':
      output_layers.append(
          LSTM(latent_dims[len(latent_dims) - j - 1], return_sequences=True, return_state=True, dropout = dropout)
      )
      outputs, dh, dc = output_layers[-1](outputs, initial_state=encoder_states[2*j : 2*(j + 1)])

    elif cell_type == 'GRU':
      output_layers.append(
          GRU(latent_dims[len(latent_dims) - j - 1], return_sequences=True, return_state=True, dropout = dropout)
      )
      outputs, dh = output_layers[-1](outputs, initial_state=encoder_states[j])

    elif cell_type == 'RNN':
      output_layers.append(
          SimpleRNN(latent_dims[len(latent_dims) - j - 1], return_sequences=True, return_state=True, dropout = dropout)
      )
      outputs, dh = output_layers[-1](outputs, initial_state=encoder_states[j])

  final_layer = Dense(num_decoder_tokens, activation='softmax')
  time_layer = TimeDistributed(final_layer)
  decoder_outputs = time_layer(outputs)

  model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

  return model

reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq, cell_type = 'LSTM'):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index["\t"]] = 1.0
    stop_condition = False
    decoded_sentence = []  
    while not stop_condition:
        if cell_type == 'LSTM':
            to_split = decoder_model.predict([target_seq] + states_value)
        else:
            to_split = decoder_model.predict([target_seq] + [states_value])

        output_tokens, states_value = to_split[0], to_split[1:]

        sampled_token_index = np.argmax(output_tokens[0, 0])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(sampled_char)

        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

    return "".join(decoded_sentence)

def get_potential(string, probab, char, char_prob):
  string = string + [char]
  probab += math.log(char_prob)
  return [string, probab]

def beam_search(input_seq, beam_size, cell_type = 'LSTM'):
  potential_seq = [[['\t'], 0]]
  explored_seq = []
  
  states = [encoder_model.predict(input_seq)] * beam_size 

  for _ in range(max_decoder_seq_length):
    seq_list = []
    for i, seq in enumerate(potential_seq):
      target_seq = np.zeros((1, 1, num_decoder_tokens))
      last_character = seq[0][-1]
      target_seq[0, 0, target_token_index[last_character[0]]] = 1.

      if cell_type == 'LSTM':  
        to_split = decoder_model.predict([target_seq] + states[i])
      else:
        to_split = decoder_model.predict([target_seq] + [states[i]])

      output_tokens, states[i] = to_split[0], to_split[1:]
      probabs, positions = tf.nn.top_k(output_tokens[0, 0], beam_size)

      for k, index in enumerate(positions):
        sequence = get_potential(seq[0], seq[1], reverse_target_char_index[index.numpy()], probabs[k].numpy())
        last_ch = sequence[0][-1]
        if last_ch[0] == '\n':
          explored_seq.append(sequence)
        else:
          seq_list.append(sequence)

    seq_list = sorted(seq_list, key = lambda x : -x[1] / (len(x[0]) ** 0.8))
    potential_seq = seq_list[:beam_size]

  explored_seq = explored_seq + potential_seq
  explored_seq = sorted(explored_seq, key = lambda x : -x[1] / (len(x[0]) ** 0.8))

  return explored_seq[: beam_size]

def custom_metric(word1, word2):
  word1 = [char for char in word1]
  word2 = [char for char in word2]
  
  return 1 if word1 == word2 else 0

def similarity(word1, word2):
  word1 = [char for char in word1]
  word2 = [char for char in word2]
  
  n, m = len(word1), len(word2)
  dp = [[0 for i in range(m + 1)] for i in range(n + 1)]
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0:
        dp[i][j] = j
      elif j == 0:
        dp[i][j] = i
      elif word1[i - 1] == word2[j - 1]:
        dp[i][j] = dp[i - 1][j - 1]         
      else:
        dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])    
  
  return 1 - dp[n][m]/m

def generate_indices():
  return np.random.randint(low=train.shape[0], high = train.shape[0] + val.shape[0], size=150)

model = build_seq2seq_model(2,2,256, 0.2, 3, 'GRU')
model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ["accuracy"])
model.fit(
    train_X,
    train_Y,
    batch_size = 128,
    epochs = 17,
    validation_data = (val_X, val_Y),
    )

encoder_model = Model(encoder_inputs, encoder_states)
d_outputs = decoder_inputs

decoder_states_inputs = []
decoder_states = []

cell_type='GRU'
search_type='beam'
beam_size=3
score = 0
acc = 0

for j in range(len(latent_dims))[::-1]: 
  if cell_type == 'LSTM':
      current_state_inputs = [Input(shape=(latent_dims[j],)) for _ in range(2)]
  else:
      current_state_inputs = [Input(shape=(latent_dims[j],)) for _ in range(1)]

  temp = output_layers[i](d_outputs, initial_state=current_state_inputs)

  d_outputs, cur_states = temp[0], temp[1:]

  decoder_states += cur_states
  decoder_states_inputs += current_state_inputs

decoder_outputs = time_layer(d_outputs)

decoder_model = tf.keras.Model(
  [decoder_inputs] + decoder_states_inputs,
  [decoder_outputs] + decoder_states) 
    
iter = 0
if search_type=='greedy':      
    for seq_index in generate_indices():
        iter += 1
        input_seq1 = encoder_input_data[seq_index : seq_index + 1]
        decoded_sentence = decode_sequence(input_seq1, cell_type)
        score += custom_metric(list(df['hindi'])[seq_index][1:], decoded_sentence)  
        acc += similarity(list(df['hindi'])[seq_index][1:], decoded_sentence)    
    score /= iter
    acc /= iter
else:
    for seq_index in generate_indices():
        iter += 1
        input_seq1 = encoder_input_data[seq_index : seq_index + 1]
        decoded_sentence = beam_search(input_seq1, beam_size, cell_type)
        ans = [''.join(k[0]) for k in decoded_sentence]
        ll = []
        bl = []        
        for i in ans:
            ll.append(custom_metric(list(df['hindi'])[seq_index], i))  
            bl.append(similarity(list(df['hindi'])[seq_index], i)) 
        score += max(ll)
        acc += max(bl)
    score /= iter
    acc /= iter

def generate_indices1():
  return [i for i in range(train.shape[0] + val.shape[0] ,train.shape[0] + val.shape[0]+test.shape[0] )]

iter = 0
p=[]
q=[]
for seq_index in generate_indices1():
    iter += 1
    input_seq1 = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = beam_search(input_seq1, beam_size, cell_type)
    ans = [''.join(k[0]) for k in decoded_sentence]
    q.append(ans)
    ll = []
    bl = []      
    
    for i in ans:
        ll.append(custom_metric(list(df['hindi'])[seq_index], i))  
        bl.append(similarity(list(df['hindi'])[seq_index], i)) 
    score += max(ll)
    acc += max(bl) 
    p.append(ans[ll.index(max(ll))])
    
score /= iter
acc /= iter

acc

score

p=pd.DataFrame(p)
p[0] = p[0].apply(lambda x : x[1:-1])

q=pd.DataFrame(q)
q[0] = q[0].apply(lambda x : x[1:-1])
q[1] = q[1].apply(lambda x : x[1:-1])
q[2] = q[2].apply(lambda x : x[1:-1])

p.to_csv('p.csv')
q.to_csv('q.csv')
p.to_csv('p.txt')
q.to_csv('q.txt')
