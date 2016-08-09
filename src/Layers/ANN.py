# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:03:44 2016

@author: sreekar
"""

# Layers 
class ANN(layer):
    def __init__(self,input_layer,num_units,W = initialize_W(),b =initialize_b(),nonlinearity):
        self.nonlinearity  = nonlinearity
        self.num_units = num_units
        self.b = 
        activation = np.dot(self.W,input_layer)
        