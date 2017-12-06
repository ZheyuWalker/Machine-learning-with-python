#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np

### some normal activation functions for logistic regression and forward propagation in simple NN

def sigmoid(Z):
  A = 1 / (1 + np.exp(-Z))
  return A
  
def tanh(X):
  A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
  return A
  
def relu(X):
  A = np.maximum(0, Z)
  assert(A.shpae == Z.shape)
  return A
  
def leaky_relu(X):
  A = np.maximum(0.01*Z, Z)
  assert(A.shape == Z.shpae)
  return A

