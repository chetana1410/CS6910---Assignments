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
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import os
import io
import time

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

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

def train_step(encoder,decoder,inp, targ, batch_size, optimizer):
  loss = 0
  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp)

    dec_hidden = enc_hidden
    enc_output = enc_output[-1]
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * batch_size, 1)

    iter = 1
    for t in range(1, targ.shape[1]):
      predictions, dec_hidden, _ = decoder(dec_input, enc_output, dec_hidden)

      loss += loss_function(targ[:, t], predictions)

      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)
  
  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

def evaluate(sentence):
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  sentence = '<start> '+sentence.replace("", " ")[1: -1]+ ' <end>'

  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''
  
  enc_out, enc_hidden = encoder(inputs)
  enc_out = enc_out[-1]
  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):

    predictions, dec_hidden, attention_weights = decoder(dec_input, enc_out, dec_hidden)

    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot

def generate_indices():
  return np.random.randint(low=0, high =  val.shape[0], size=150)

epochs = 10
dropout = 0.2
num_encoder_layers = 1
num_decoder_layers = 1
batch_size = 32
hl_size = 128
cell_type = 'GRU'
optimizer = 'adam'
embedding_dim = 128

latent_dims = [hl_size] 

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(batch_size, drop_remainder=True)
global steps_per_epoch
steps_per_epoch = len(input_tensor_train)//batch_size

global encoder
encoder = Encoder(vocab_inp_size, embedding_dim, batch_size, latent_dims, dropout, cell_type)
global decoder
decoder = Decoder(vocab_tar_size, embedding_dim,batch_size, latent_dims, dropout, cell_type)

global optimizer
if optimizer=='adam':
  optimizer = tf.keras.optimizers.Adam()



EPOCHS = epochs
score=0
for epoch in range(EPOCHS):
  start = time.time()

  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(encoder,decoder,inp, targ, batch_size, optimizer)
    total_loss += batch_loss

    if batch % 100 == 0:
      print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')

  print(f'Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f}')
  print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')

iter=0
for seq_index in generate_indices()[:5]:
  iter+=1
  result,_ , _ = evaluate(val['english'].iloc[seq_index])
  result = ''.join(result[:-1])
  result = result.split(' ')
  result = ''.join(result[:-1])
  
  if result==val['hindi'].iloc[seq_index]:
    score += 1
score/=iter
print(score)



# Q5d

from pathlib import Path
from matplotlib.font_manager import FontProperties
h= Path('/content/Lohit-Devanagari.ttf')
hindi_font = FontProperties(fname=h)
rnd = np.random.randint(low=0, high = test.shape[0], size=12)
fig , ax = plt.subplots(4,3,figsize=(30,30))
j=0
for i in rnd:
  
  sentence=test['english'].iloc[i]
  result, sentence, attention_plot = evaluate(sentence)
  attention_plot = attention_plot[:len(result.split(' ')),:len(sentence.split(' '))]
  sentence = sentence.split(' ') 
  predicted_sentence=result.split(' ')
  ax[j//3][j%3].matshow(attention_plot, cmap='viridis')

  ax[j//3][j%3].set_xticklabels([''] + sentence, fontsize=20, rotation=90)
  ax[j//3][j%3].set_yticklabels([''] + predicted_sentence, fontsize=35,fontproperties=hindi_font)

  ax[j//3][j%3].xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax[j//3][j%3].yaxis.set_major_locator(ticker.MultipleLocator(1))
  ax[j//3][j%3].set_title('ATTENTION PLOT '+str(j+1),fontsize=20,fontweight='bold', pad=50)
  j+=1

fig.tight_layout(pad=10)
plt.show()



  
#Q6

from IPython.display import HTML as html_print
from IPython.display import display

# get html element
def cstr(s, color='black'):
	if s == ' ':
		return "<text style=color:#000;padding-left:10px;background-color:{}> </text>".format(color, s)
	else:
		return "<text style=color:#000;background-color:{}>{} </text>".format(color, s)

# get appropriate color for value
def get_clr(value):
	colors = [
          '#e1ddec', 'd6cfe6', '#ccc3de', 'bdb2d4', '#b4aacf', 
          'ae9fc9', '#a393c0', '#927fb5', '#836dab', '#765b9e', 
          '#664c95', '#5b3c8c', '59388f', '#502984', '#472476', 
          '#431d66', '#38195c', '#2f114d', '#250640', '#1e0034'
]
	value = int((value * 100) / 5)
	return colors[value]

def visualize_connectivity(sentence):
  html_format_str = '''
  <div class="container" style="
    font-family: Arial, Helvetica, sans-serif;
    border: 1px solid rgba(0, 0, 0, 0.514);
    border-radius: 8px;
    margin: 20px auto;
    padding: 20px 10px;
    background-color: rgba(250, 249, 249, 0.5);
    width: 1000px;
  ">
    <h2 style="
      text-align: center;
      margin-top: 10px;
      margin-bottom: 25px;
    ">Visualizing Attention for Predictions</h1>
    <div class="wrapper">
      <div class="left" style="
        width: 60%;
        padding: 10px;    
        margin: 0 auto;
        display: flex;
        justify-content: space-evenly;   
      ">
        <p style="margin: 8px 0">Input: <b>{}</b></p>
    '''.format(sentence)
  result, sentence1, attention_scores = evaluate(sentence)

  

  result = result.split(' ')
  predicted = ''.join(result[:result.index('<end>')])

  html_format_str += '''
        <p style="margin: 8px 0">Prediction: <b>{}</b></p>
      </div>
      <div class="right" style="
        width: 60%;
        padding: 0;    
        margin: 0 auto;      
      ">
        <table class="attention" style="
          background-color: white;
          text-align: center;
          width: 100%;
          border-collapse: collapse;
          border: 2px solid black;
        ">
          <tbody>
            <tr>
              <td style="border: 1px solid rgb(0, 0, 0, 0.75);padding: 10px 0;"><b>Character at each index</b></td>
              <td style="border: 1px solid rgb(0, 0, 0, 0.75);padding: 10px 0;"><b>Attention Visualization</b></td>
            </tr>
    '''.format(predicted)

  for i in range(len(predicted)):
    res = ''

    for j in range(len(sentence)):    
      res += cstr(sentence[j], get_clr(attention_scores[i, j + 1]))

    html_format_str += '''
            <tr>
              <td style="border: 1px solid rgb(0, 0, 0, 0.75);padding: 10px 0;"><b>Character at index {}: {}</b></td>
              <td style="border: 1px solid rgb(0, 0, 0, 0.75);padding: 10px 0;">{}</td>
            </tr>
    '''.format(i, predicted[i], res)
  
  html_format_str += '''
          </tbody>
        </table>
      </div>
    </div>
  </div>
  '''
 
  display(html_print(html_format_str))
	
visualize_connectivity('sunil')
