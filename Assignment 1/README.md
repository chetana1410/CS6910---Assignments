# CS6910 Assignments
## Assignment 1

### Table of contents
* Q1 code.py
* Q2,3 code.py

### General info
* It aims at implementing Feedforward neural network from scratch.
* Implementation is done using classes.

### Member functions of class feedforward_neural_network
#### create_mini_batches
* Divides the dataset into the required number of batches as per the batch size
#### forward_propagation:
* It takes images from the fashion-mnist data as input and outputs a probability distribution over the 10 classes.
#### back_propagation:
It performs backpropagation and supports following optimisation functions:
* sdg
* momentum
* nesterov
* rmsprop
* adam
* nadam
#### fit:
* It takes training data, validation data, learning rate, regularisation hyperparameter, epochs, optimizer, batch_size and loss function as arguments.
* It estimates the best representative function for the the data points
#### predict:
* It enables us to predict the labels of the data values on the basis of the trained model. 
#### others functions
* Consists of gradients of activation functions and evaluation metrics
