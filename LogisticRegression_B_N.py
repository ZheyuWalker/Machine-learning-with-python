#!/usr/bin/env python
#-*- coding:utf-8 -*-


### Logistic Regression Algorithm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def sigmoid(Z):
  A = 1 / (1/ np.exp(-Z))
  return A

def fit(X_train, Y_train, iteration_num = 1000, learning_rate = 0.01):
  '''
  Arguments:
  X_train -- input data, a numpy array of shape (n_x, m)
  Y_train -- label of input data , a numpy array of shape (1, m)
  iteration_num -- number of iterations of the optimization loop
  learning_rate -- learning rate of the gradient descent update rule
  
  Returns:
  W -- weights vector, a numpy array of shape (1, n_x)
  b -- bias, a real number
  costs -- list of all the costs computed during the optimization
  '''

  m = X_train.shape[1]
  costs = []
  
  # initialize the weigths and bias with 0
  W = np.zeros((1, X_train.shape[0]))
  b = 0
  
  Z = np.dot(W, X) + b
  A = sigmoid(Z)
  
  for i in range(iteration_num):
    dZ = A - Y
    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ)
    W = W - learning_rate * dW
    b = b - learning_rate * db
    Z = np.dot(W, X) + b
    A = sigmoid(Z)
    cost = -1/m * (np.dot(A, Y.T) - np.dot(1-A, (1-Y).T))
    if i % 100 == 0:
      costs.append(cost)
  
  return W, b, costs
  
def predict(X_test, W, b):
  '''
  Arguments:
  X_test: test input data, a numpy array of shape(n_x, m)
  W: optimized weights, a numpy array of shape (1, n_x)
  b: optimized bias, a real number
  
  Returns:
  Yhat: prediction of test input data X, a numpy array of shape (1, m)
  '''
  A = sigmoid(np.dot(W, X_test) + b)
  Yhat = np.round(A+1) -1
  
  return Yhat

'''
X_train, Y_train, X_test is needed
Y_test is needed to compute the accuracy of prediction
'''
i = 1500
a = 0.005
W, b, costs = fit(X_train, Y_train, iteration_num = i, learning_rate = a)
Yhat = predict(X_test, W, b)
accuracy = np.mean(np.abs(Yhat - Y))

costs = np.squeeze(costs)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate = ", a)
plt.show()
  
