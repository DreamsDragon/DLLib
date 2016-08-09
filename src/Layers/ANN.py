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
    def __init__(self,num_units,nonlinearity,Initialise_W,Initialise_b):
        self.nonlinearity  = nonlinearity # Activation function to be used
        self.num_units = num_units # Number of perceptrons in the layer
        self.b = Initialise_b(num_units) # Initial Bias 
    	self.input = 0 # Input vector
    	self.W = Initialise_W(num_units) # Initial Weights

    #This function activates the neurons
    #That is the neuron evaluates W.X + b
    def activate(self):    
    	return np.dot(self.W,self.input)+self.b
    
    #Gives the final output of the layer after running it through
    #the activation layer
    def get_out(self):
    	return self.nonlinearity(self.activate())
    
    #Get the current weights and biases of the layer as a tuple
    def get_params(self):
    	return (self.W,self.b)

    #Set the current input value of the layer
    def give_in(self,input_layer):
    	self.input = input_layer

    #Change the activation function used by the function
    def change_activation_fnc(self,new_fnc):
    	self.nonlinearity = new_fnc
