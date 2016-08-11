# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 08:52:00 2016

@author: sreekar
"""

import numpy as np

class trainer():
    def __init__(self,layers):
        self.layers = layers   #The dictionary which contains all the layer objects
        
        
    def backprop(self,error,target,output):
        grads = {}
        if error == 'MS':
            delta_out = np.multiply((output - target),self.layers[len(self.layers)].nonlinearity(np.dot(self.layers[len(self.layers)].W,self.layers[len(self.layers)].input) + self.layers[self.layers].b,deriv = True))
        dWo = np.dot(self.layers['out'].input,delta_out.T)
        dbo = delta_out
        grads['out'] = (dWo,dbo)
        delta_prev = delta_out
        for i in reversed(self.layers.keys()):
            i = i-1
            delta = np.multiply(np.dot(delta_prev.T,self.layers[i].W),self.layers[i].nonlinearity(np.dot(self.layers[i].W,self.layers[i].input) + self.layers[i].b,deriv = True))
            dW = np.dot(delta,self.layers[i].input)
            db = delta
            grads[i] = (dW,db)
            delta_prev = delta
        return grads

    
 
    
    
    
        
    