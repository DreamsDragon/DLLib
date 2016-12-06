"""

The file contains the code for ANN

Author : Karunakar Gadireddy , Sreekar Kamireddy

Dated : 18 December 2016

"""

import numpy as np

class ANN():
	"Artifical Neural Network Layer"

	def __init__(self,nb_neurons,act_fnc,prev,init_w):
		self.nb_neurons = nb_neurons # Number of neurons in this hidden layer
		self.prev = prev # Previous Layer
		self.nb_ins = prev.nb_neurons+1 # Number of inputs to this layer
		self.w = init_w((self.nb_ins,nb_neurons)) # A random matrix of size nb_inputs * nb_neurons
		self.input = 0 # Inputs of this layer
		self.act_fnc = act_fnc # Activation function of this layer
		self.out = 0 # Output of this layer (f(wx))
		self.out_der = 0 # Derivative of the output of this layer (f'(wx))
		self.type = "ANN" # Type of the layer


	def activate(self):
		self.input = np.ndarray.tolist(self.prev.get_out()[0]) # Getting the output of the previous layer

		self.input.append([1]) # converting it to a list, appending 1 to it 
		self.input = np.array(self.input) # and converting it back to  np array
		wx = np.dot(self.w.T,self.input) # Matrix multiplication of W (input * nb_neutron) and input

		self.out = self.act_fnc(wx) # Activated output

		self.out_der = self.act_fnc(wx,True) # Derivative of activated output

	def get_out(self):
		return (self.out,self.out_der) # Return activated output and derivative of activated output

	def update_w(self,new):
		self.w = new