# CS6910 Assignment 3
This repository contains the following files and folders -
* `predictions_vanilla` - This folder contains predictions of the best model without attention.
* `predictions_attention` -  This folder contains predictions of the best model with attention.
* `Q1.py` - This contains code Build a RNN based seq2seq model which contains the following layers: 
  (i) input layer for character embeddings 
  (ii) one encoder RNN which sequentially encodes the input character sequence (Latin) 
  (iii) one decoder RNN which takes the last state of the encoder as input and produces one output character at a time.

   * The code is also flexible such that the dimension of the hidden states of the encoders and decoders, the cell (RNN, LSTM, GRU) and the number of layers in the encoder and decoder, dropout, batch size can be changed.
   * The decoded sequence is predicted using two methods:
       * greedy
       * beam search
   * There are two metrics to calculate accuracy:
       * `Custom Metric` : It checks whether the predicted sequence is exactly equal to true sequence or not.
       * `Similarity`: It estimates accuracy of predicted sequence at character level.
