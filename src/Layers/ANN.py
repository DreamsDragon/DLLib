# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:03:44 2016

@author: sreekar
"""
import numpy as np

# Artifical Neural Network Layer
# W is the weights of the arcs connecting to the layer
# B is the bias of the nodes in the layer
# Non Linearity is the activation function
# Num_units is the number of neurons in the layer

class ANN():
    # This is a class for a layer of perceptrons 
    # The number of neurons and initial weights and biases are given at the time of initialisation
    # An activation function object is also passed as an argument
    def __init__(self,prev_layer,num_units,nonlinearity,Initialise_W,Initialise_b,batch_size=1):
        self.nonlinearity  = nonlinearity # Activation function to be used
        self.num_units = num_units # Number of perceptrons in the layer
        self.b = Initialise_b((num_units,batch_size),'b') # Initial Bias 
        self.input = np.zeros_like(prev_layer.num_units) # Input vector
        self.W = Initialise_W((num_units,prev_layer.num_units)) # Initial Weights
        self.prev_layer = prev_layer
        self.out_val = 0
        self.out_val_der = 0
        self.type = 'ANN'
        
    #This function activates the neurons
    #That is the neuron evaluates W.X + b
    def activate(self):


        self.get_in()  
        wxb = np.dot(self.W,self.input) + self.b
        self.out_val = self.nonlinearity(wxb)
        self.out_val_der = self.nonlinearity(wxb,True) 
    #Gives the final output of the layer after running it through
    #the activation layer
    def get_out(self,deriv = False):
    	return (self.out_val,self.out_val_der)
    
    #Get the current weights and biases of the layer as a tuple
    def get_params(self):
    	return (self.W,self.b)

    #Change the activation function used by the function
    def change_activation_fnc(self,new_fnc):
    	self.nonlinearity = new_fnc

    def get_in(self):
        self.input = self.prev_layer.get_out()[0]
        
    def set_w_b(self,inp):
        self.W = inp[0]
        self.b = inp[1]

    def get_save_array(self):
        return (self.type,self.num_units,self.W,self.b)
