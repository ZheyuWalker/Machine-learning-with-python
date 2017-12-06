#!/usr/bin/env python
#-*- coding:utf-8 -*-

'''
This file will implement a single hidden layer neural network(2 layers NN) for binary classification.
The hidden layer can use different activation functions like sigmoid, tanh, relu or leacky relu etc.
Since this NN is used for binary classification, so output layer will use sigmoid function as activation function
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def initialize_parameters(n_x, n_h, n_y):
  '''
  Arguments:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
   
  Returns:
    params -- python dictionary containing network parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
  '''
