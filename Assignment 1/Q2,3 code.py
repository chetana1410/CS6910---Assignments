# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p7U3tHtk8QXGenQZCQS48wK1nTVKx8Rh
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# ! pip install wandb --upgrade

import wandb
import numpy as np
import math
from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical 
from sklearn.model_selection import train_test_split

wandb.login()

class feedforward_neural_network:
  def __init__(self, image_input, output, hidden_info, activation_info = ['sigmoid'], weight_decay = 0.001, weight_initialization = 'random'):
    self.n_inputs = image_input         # input size
    self.n_class = output               # number of classes
    self.n_hidden_layers = len(hidden_info)
    self.activation = [''] + activation_info
    self.weight_decay = weight_decay
       
    # hidden info is a list of n integers where ith value represent number of neurons in hidden layer i.
    self.network = [self.n_inputs] + hidden_info + [self.n_class]
    
    # dict to store the randomly initialized weights and biases for each hidden layer in the network.
    self.W = {}
    self.B = {}
    for i in range(self.n_hidden_layers+1):
      np.random.seed(0);
      if weight_initialization == 'random':        
        self.W[i+1] = np.random.randn(self.network[i], self.network[i+1])
        self.B[i+1] = np.zeros((1, self.network[i+1]))
      else:
        self.W[i+1] = np.random.normal(scale = math.sqrt(2 / np.sum((self.network[i], self.network[i+1]))), size = (self.network[i], self.network[i+1]))
        self.B[i+1] = np.random.normal(scale = math.sqrt(2 / np.sum((1, self.network[i+1]))), size = (1, self.network[i+1]))   
                
  def create_mini_batches(self,X,Y,batch_size):    
    
    mini_batches=[]
    data=np.hstack((X,Y))
    np.random.shuffle(data)

    n_batches=X.shape[0]//batch_size
    
    mini_batches=[data[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
    mini_batches=[(i[:,:-10],i[:,-10:]) for i in mini_batches]
        
    if X.shape[0]%batch_size!=0:
      mini_batch=data[(n_batches)*batch_size:]
      mini_batches.append((mini_batch[:,:-10],mini_batch[:,-10:]))        
    return mini_batches

  def cross_entropy(self,Y_pred, Y_true, epsilon=1e-12):
        
    Y_pred = np.clip(Y_pred, epsilon, 1. - epsilon)
    ce = -np.sum(Y_true*np.log2(Y_pred+1e-9))/Y_pred.shape[0]
    return ce

  def RMSE(self, Y_true, Y_pred):
    return np.sqrt(np.sum((Y_true-Y_pred)**2)/len(Y_true))

  def accuracy(self,Y_true, Y_pred):
        
    Y_true = [np.argmax(i) for i in Y_true]
    Y_pred = [np.argmax(i) for i in Y_pred]
    diff= np.array(Y_true) - np.array(Y_pred)
    return np.count_nonzero(diff==0)*100/len(diff)
       
  def sigmoid(self, x):
    x = np.clip(x, -50, 50)
    return 1.0/(1.0 + np.exp(-x))
   
  def tan_h(self, x):
    x = np.clip(x, -50, 50)
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
   
  def ReLU(self, x):          
    return np.maximum(x,0)    
   
  def softmax_y(self, x):
    x = np.clip(x, -50, 50)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
   
  def forward_propagation(self, x, W, B):
    self.A = {}
    self.H = {}
    self.H[0] = x.reshape(1, -1)
    for i in range(self.n_hidden_layers):
      self.A[i+1] = np.matmul(self.H[i], W[i+1]) + B[i+1]
      if self.activation[i+1] == 'sigmoid':
        self.H[i+1] = self.sigmoid(self.A[i+1])
      elif self.activation[i+1] == 'tanh':
        self.H[i+1] = self.tan_h(self.A[i+1])
      elif self.activation[i+1] == 'ReLU':
        self.H[i+1] = self.ReLU(self.A[i+1])
          
    self.A[self.n_hidden_layers+1] = np.matmul(self.H[self.n_hidden_layers], W[self.n_hidden_layers+1]) +  B[self.n_hidden_layers+1]
    return self.A[self.n_hidden_layers+1]
     
  def grad_sigmoid(self, x):
    return self.sigmoid(x)*(1 - self.sigmoid(x))
   
  def grad_tanh(self, x):
    return 1 - (self.tan_h(x))**2
   
  def grad_ReLU(self, x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
   
  def back_propagation(self,x,y,W,B,loss_fn):
    self.forward_propagation(x, W, B)
    self.dW = {}
    self.dB = {}
    self.dH = {}
    self.dA = {}
    L = self.n_hidden_layers + 1
    if loss_fn == 'cross_entropy':
      self.dA[L] = (self.softmax_y(self.A[L]) - y)
    else:
      y_pred = self.softmax_y(self.A[L])
      self.dA[L] = ((y_pred - y) - (y_pred - y) * y_pred) * y_pred
    for k in range(L, 0, -1):
      self.dW[k] = self.H[k-1].T @ self.dA[k]
      self.dW[k] += (self.weight_decay * W[k])
      self.dB[k] = self.dA[k]
      self.dB[k] += (self.weight_decay * B[k]) 
      
      self.dH[k-1] = self.dA[k] @ W[k].T
      if self.activation[k - 1] == 'sigmoid':
        self.dA[k-1] = np.multiply(self.dH[k-1], self.grad_sigmoid(self.H[k-1]))
      elif self.activation[k - 1] == 'tanh':
        self.dA[k-1] = np.multiply(self.dH[k-1], self.grad_tanh(self.H[k-1]))
      if self.activation[k - 1] == 'ReLU':
        self.dA[k-1] = np.multiply(self.dH[k-1], self.grad_ReLU(self.H[k-1]))
               
           
  def fit(self, X, Y, x_val,y_val, epochs, optimizer, batch_size, learning_rate, loss_fn, metric, verbose, callback_fn):
    M_W_updates = {}
    M_B_updates = {}
    V_W_updates = {}
    V_B_updates = {}
    W_look_ahead = {}
    B_look_ahead = {}
    for i in range(self.n_hidden_layers + 1):
      M_W_updates[i + 1] = np.zeros((self.network[i], self.network[i+1]))
      M_B_updates[i + 1] = np.zeros((1, self.network[i+1]))
      V_W_updates[i + 1] = np.zeros((self.network[i], self.network[i+1]))
      V_B_updates[i + 1] = np.zeros((1, self.network[i+1]))
      W_look_ahead[i+1] = self.W[i+1]-M_W_updates[i + 1]
      B_look_ahead[i+1] = self.B[i+1]-M_B_updates[i + 1]
          
          
    for iters in range(epochs):
      dW = {}
      dB = {}
      for i in range(self.n_hidden_layers + 1):
        dW[i+1] = np.zeros((self.network[i], self.network[i+1]))
        dB[i+1] = np.zeros((1, self.network[i+1]))

      mini_batches = self.create_mini_batches(X, Y, batch_size)
      
      for batch in mini_batches:
        X_batch,Y_batch=batch[0] , batch[1]
        sz = len(X_batch)  
        for x, y in zip(X_batch, Y_batch): 
            
          if optimizer == 'sgd' or optimizer == 'momentum' or optimizer == 'rmsprop' or optimizer == 'adam':
            self.back_propagation(x, y,self.W,self.B, loss_fn)
            for i in range(self.n_hidden_layers + 1):
              dW[i+1] += self.dW[i+1]/sz
              dB[i+1] += self.dB[i+1]/sz

          elif optimizer == 'nesterov' or optimizer == 'nadam':
            self.back_propagation(x, y, W_look_ahead, B_look_ahead, loss_fn)
            for i in range(self.n_hidden_layers + 1):
              dW[i+1] += self.dW[i+1]/sz
              dB[i+1] += self.dB[i+1]/sz
     
        
        if optimizer == 'sgd':
          for i in range(self.n_hidden_layers + 1):
            self.W[i+1] -= learning_rate * dW[i+1]
            self.B[i+1] -= learning_rate * dB[i+1]

        elif optimizer == 'nesterov':
          gamma = 0.9
          for i in range(self.n_hidden_layers + 1):
            M_W_updates[i+1] = gamma * M_W_updates[i + 1] + learning_rate * dW[i+1]
            M_B_updates[i+1] = gamma * M_B_updates[i + 1] + learning_rate * dB[i+1]
            W_look_ahead[i+1] = self.W[i+1] - M_W_updates[i + 1]
            B_look_ahead[i+1] = self.B[i+1] - M_B_updates[i + 1]
            self.W[i+1] -= M_W_updates[i+1]
            self.B[i+1] -= M_B_updates[i+1]

        elif optimizer == 'momentum':
          gamma = 0.9
          for i in range(self.n_hidden_layers + 1):
            M_W_updates[i+1] = gamma * M_W_updates[i + 1] + learning_rate * dW[i+1]
            M_B_updates[i+1] = gamma * M_B_updates[i + 1] + learning_rate * dB[i+1]                    
            self.W[i+1] -= M_W_updates[i+1]
            self.B[i+1] -= M_B_updates[i+1]

        elif optimizer == 'rmsprop':
          epsilon = 1e-8
          beta = 0.9
          for i in range(self.n_hidden_layers + 1):
            V_W_updates[i+1] = beta * V_W_updates[i + 1] + (1-beta) * dW[i+1]*dW[i+1]
            V_B_updates[i+1] = beta * V_B_updates[i + 1] + (1-beta) * dB[i+1]*dB[i+1]                  
            self.W[i+1] -= learning_rate * dW[i+1]/np.sqrt(V_W_updates[i+1]+epsilon)
            self.B[i+1] -= learning_rate * dB[i+1]/np.sqrt(V_B_updates[i+1]+epsilon)

        elif optimizer == 'adam':
          epsilon = 1e-8
          beta1 = 0.9
          beta2 = 0.999
          
          for i in range(self.n_hidden_layers + 1):
            M_W_updates[i+1] = beta1 * M_W_updates[i + 1] + (1-beta1) * dW[i+1]
            M_B_updates[i+1] = beta1 * M_B_updates[i + 1] + (1-beta1) * dB[i+1]  

            V_W_updates[i+1] = beta2 * V_W_updates[i + 1] + (1-beta2) * dW[i+1]*dW[i+1]
            V_B_updates[i+1] = beta2 * V_B_updates[i + 1] + (1-beta2) * dB[i+1]*dB[i+1]

            M_W_hat = M_W_updates[i+1]/(1-math.pow(beta1,iters+1))
            M_B_hat = M_B_updates[i+1]/(1-math.pow(beta1,iters+1))

            V_W_hat = V_W_updates[i+1]/(1-math.pow(beta2,iters+1))
            V_B_hat = V_B_updates[i+1]/(1-math.pow(beta2,iters+1))

            self.W[i+1] -= learning_rate * M_W_hat/np.sqrt(V_W_hat + epsilon)
            self.B[i+1] -= learning_rate * M_B_hat/np.sqrt(V_B_hat + epsilon)
                    
        elif optimizer == 'nadam':
          epsilon = 1e-8
          beta1 = 0.9
          beta2 = 0.999
          for i in range(self.n_hidden_layers + 1):
            M_W_updates[i+1] = beta1 * M_W_updates[i + 1] + (1-beta1) * dW[i+1]
            M_B_updates[i+1] = beta1 * M_B_updates[i + 1] + (1-beta1) * dB[i+1]  

            V_W_updates[i+1] = beta2 * V_W_updates[i + 1] + (1-beta2) * dW[i+1]*dW[i+1]
            V_B_updates[i+1] = beta2 * V_B_updates[i + 1] + (1-beta2) * dB[i+1]*dB[i+1]

            M_W_hat = M_W_updates[i+1]/(1 - beta1**(iters+1))
            M_B_hat = M_B_updates[i+1]/(1 - beta1**(iters+1))

            V_W_hat = V_W_updates[i+1]/(1 - beta2**(iters+1))
            V_B_hat = V_B_updates[i+1]/(1 - beta2**(iters+1))
            
            W_look_ahead[i+1] = self.W[i+1] - M_W_updates[i + 1]
            B_look_ahead[i+1] = self.B[i+1] - M_B_updates[i + 1]
            
            self.W[i+1] -= learning_rate * beta1 /(1 - beta1**(iters+1)) * M_W_hat/np.sqrt(V_W_hat + epsilon)
            self.W[i+1] -= learning_rate * (1 - beta1)/(1 - beta1**(iters+1)) * dW[i+1]/np.sqrt(V_W_hat + epsilon)
            self.B[i+1] -= learning_rate * beta1 /(1 - beta1**(iters+1)) * M_B_hat/np.sqrt(V_B_hat + epsilon)
            self.B[i+1] -= learning_rate * (1 - beta1)/(1 - beta1**(iters+1)) * dB[i+1]/np.sqrt(V_B_hat + epsilon)
  
                          
      train_eval_metrics={}
      val_eval_metrics={}
      
      train_preds = self.predict(X)
      train_eval_metrics['cross_entropy']=self.cross_entropy(train_preds,Y)
      train_eval_metrics['rmse']=self.RMSE(train_preds,Y)
      train_eval_metrics['accuracy'] = self.accuracy(train_preds,Y)
      
      val_preds = self.predict(x_val)
      val_eval_metrics['cross_entropy']=self.cross_entropy(val_preds,y_val)
      val_eval_metrics['rmse']=self.RMSE(val_preds,y_val)
      val_eval_metrics['accuracy'] = self.accuracy(val_preds,y_val)
      
      if iters % verbose == 0: 
        print('Epoch[{}]  ;  Training accuracy =  {}     ;  Val accuracy =  {}'.format(iters,train_eval_metrics['accuracy'],val_eval_metrics['accuracy']))

      callback_fn(iters,train_eval_metrics,val_eval_metrics)
                      
                        
  def predict(self,x):
    Y_pred_prob=[self.softmax_y(self.forward_propagation(i,self.W,self.B)[0]) for i in x]
    return Y_pred_prob

def train():
  default_hyperparams = dict(
      hidden_layer_size=32,
      num_hidden_layers=3,
      lr=0.01,
      epochs=5,
      l2=0,
      optimizer="adam",
      batch_size=64,
      weight_init="random",
      activation="ReLU",
  )

  metrics_list = ["mse", "accuracy"]


  def callback(epoch, train_eval_metrics, val_eval_metrics):
      wandb.log({"train_cross_entropy_loss": train_eval_metrics['cross_entropy'], "train_rmse": train_eval_metrics['rmse'],"train_accuracy": train_eval_metrics['accuracy'],
                "val_cross_entropy_loss": val_eval_metrics['cross_entropy'], "val_rmse": val_eval_metrics['rmse'],"val_accuracy": val_eval_metrics['accuracy']})

        # Pass your defaults to wandb.init
  wandb.init(config = default_hyperparams)
  config = wandb.config
  # Your model here ...
  np.random.seed(0)
  (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

  train_x = np.reshape(train_x, (-1, 784))/255.0
  test_x = np.reshape(test_x, (-1, 784))/255.0
  train_y = to_categorical(train_y)
  train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=0, shuffle=True)
  model = feedforward_neural_network(28*28,10,[config.hidden_layer_size]*config.num_hidden_layers,[config.activation]*config.num_hidden_layers,config.l2,config.weight_init)

  model.fit(train_x, train_y, val_x, val_y, epochs=config.epochs,optimizer=config.optimizer, verbose=10,
            batch_size=config.batch_size,metric='cross_entropy',learning_rate=config.lr,callback_fn=callback,loss_fn='cross_entropy)

sweep_config = {
   #'program': train(),
    'method': 'random',         
    'metric': {
        'name': 'val_accuracy',     
        'goal': 'maximize'      
    },
    'parameters': {
        'activation': {
            'distribution' : 'categorical',
            'values': ['ReLU', 'tanh']
        },
        'lr': {
            'distribution': 'categorical',
            'values': [1e-2, 1e-4 ,1e-3]
        },
        'batch_size': {
            'distribution' : 'categorical',
            'values': [32, 64, 128]
        },
        'hidden_layer_size': {
            'distribution' : 'categorical',
            'values': [32, 64, 128]
        },
        'epochs': {
            'distribution' : 'categorical',
            'values': [5, 10, 15]
        },
        'num_hidden_layers': {
            'distribution' : 'categorical',
            'values': [3, 4, 5]
        },
        'optimizer': {
            'distribution' : 'categorical',
            'values': ['adam', 'nadam', 'rmsprop']
        },
        'l2': {
            'distribution' : 'categorical',
            'values': [0,0.005, 0.0005]
        },
        'weight_init': {
            'distribution' : 'categorical',
            'values': ['xavier', 'random']
        }

        }
    }

sweep_id = wandb.sweep(sweep_config,project='asgn1')

wandb.agent(sweep_id, function=train)
