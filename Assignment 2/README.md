# CS6910 Assignment 2
This repository contains the following files -
* `Part A` - This folder contains the following: 
  * `Q1.py` - This contains code for the CNN model consisting of 5 convolution layers. It further consists of the code that uses the sweep feature in wandb to find the best hyperparameter configuration.
  * `Q4, 5.py` - This file contains codes for the following:
    *  Plotting a 10 x 3 grid containing sample images from the test data and predictions made by the best model
    *  Visualizing all the filters in the first layer of the best model for a random image from the test set.
    *  Guided back propagation on any 10 neurons in the CONV5 layer and plotting the images which excite this neuron.

* `Part B` - This folder contains the following:
  * `Q1,2.py` - This contains the code for loading pre- trained models (VGG16, ResNet50, InceptionV3, Xception and InceptionResNetV2) and fine-tuning it using the naturalist data.
  It further has the code to use the sweep feature in wandb to find the best hyperparameter configuration for each model mentioned above. 
