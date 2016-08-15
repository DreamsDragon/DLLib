# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 19:18:50 2016

@author: sreekar
"""

# This file contains all the activations functions used by
# the software to squach the input 
#import theano.tensor as T
import numpy as np
from collections import *

# Defining the possible non-linearity functions for the activations of neurons that are most commonly used are written here and can be imported later on while building models

# ReLu --- Rectified Linear Unit
def relu(x,deriv = False):
    if not deriv:
        return np.asfarray(np.maximum(0,x))
    if deriv:
        der = np.asfarray(np.maximum(0,x))
        der[der > 0] = 1
        return der
    
# Leaky ReLu --- Leaky Rectified Linear Unit
def Prelu(x,deriv = False,parameter=0.1):
    if not deriv:
        return np.asfarray(np.maximum(0,np.multiply(parameter,x)))
    else:
        der = np.asfarray(np.maximum(0,np.multiply(parameter,x)))
        der[der > 0] = 1
        der[der < 0] = parameter
        return der
# ELU    
def elu(x,deriv = False):
    x = np.asfarray(x)
    if not deriv:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i,j] >= 0:
                    x[i,j] = x[i,j]
                else:
                    temp = np.exp(x[i,j]) - 1
                    x[i,j] = temp
        return x
    else:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i,j] >= 0:
                    x[i,j] = 1
                else:
                    temp = np.exp(x[i,j])
                    x[i,j] = temp
        return x

# sigmoid 
def sigmoid(x,deriv = False):
    if not deriv:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        sig = 1.0 / (1.0 + np.exp(-x))
        return (sig)*(1-sig)
    
# tanh
def tanh(x,deriv = False):
    if not deriv:
        return np.tanh(x)
    else:
        1 - (np.tanh(x)**2)

# Softmax
def softmax(x,deriv = False):
    sig = np.exp(x - np.max(x))
    return np.divide(sig,np.sum(sig))
    
    
# Gradient Test Code
def grad(cost,params):
    params = [params]
    outputs = []
    outputs.append(cost)
    
    grad_dict = OrderedDict()
    known_grads = OrderedDict()
    grad_cost = np.ones_like(cost,dtype = 'float')
    grad_dict[cost] = grad_cost
    
    for var in known_grads:
        grad_dict[var] = known_grads[var]

'''        
a = np.array([[-1,2,3],[4,-5,6],[7,8,9]])
print T.nnet.elu(a).eval()
print elu(a)
print np.maximum(0,a)
'''


