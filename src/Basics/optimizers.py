# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:16:01 2016

@author: sreekar
"""
import collections
# Objective Functions

def sgd(gradients,parameters,learning_rate = 005):
    updates = collections.OrderedDict()
    for p,grad in zip(parameters,gradients):
        updates[p] = p - learning_rate*grad
    return updates
    

        
        

        
