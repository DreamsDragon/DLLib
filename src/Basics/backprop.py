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
        dWo = np.dot(delta_out,self.layers[no_of_layers].input.T)
        dbo = delta_out
        grads[no_of_layers] = (dWo,dbo)
        delta_prev = delta_out
        for i in reversed(self.layers.keys()):
            if i == no_of_layers:
                continue

            delta = np.multiply(np.dot(self.layers[i+1].W.T,delta_prev),self.layers[i].get_out(True))

            dW = np.dot(delta,self.layers[i].input.T)

            db = delta


            grads[i] = (dW,db)

            delta_prev = delta
        return grads

    
 
    
    
    
        
    