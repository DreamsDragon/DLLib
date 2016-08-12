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
        no_of_layers = len(self.layers)
        if error == 'MS':
            delta_out = np.multiply((output - target),self.layers[no_of_layers].get_out(True))
        dWo = np.dot(self.layers[no_of_layers].input,delta_out.T)
        dbo = delta_out
        grads[no_of_layers] = (dWo,dbo)
        delta_prev = delta_out
        for i in reversed(self.layers.keys()):
            if i == no_of_layers:
                continue

            delta = np.multiply(np.dot(delta_prev.T,self.layers[i+1].W),self.layers[i].get_out(True).T)

            dW = np.dot(delta.T,self.layers[i].input.T)

            db = delta

            print db.shape
            print self.layers[i].input.shape
            grads[i] = (dW,db)

            delta_prev = delta
        return grads

    
 
    
    
    
        
    