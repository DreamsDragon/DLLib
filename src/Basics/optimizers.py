# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:16:01 2016

@author: sreekar
"""
import collections
# Objective Functions

def sgd(gradients,parameters,learning_rate = 0.05):
    updates = dict()
    for p,grad in zip(parameters,gradients):
        updates[p] = (parameters[p][0] - learning_rate*gradients[grad][0],parameters[p][1] - learning_rate*gradients[grad][1])
    return updates
    

        
        

        
