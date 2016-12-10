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
		self.init_w = init_w # Method Used to initialise the weights
		self.w = init_w((self.nb_ins,nb_neurons)) # A random matrix of size nb_inputs * nb_neurons
		self.input = 0 # Inputs of this layer
		self.act_fnc = act_fnc # Activation function of this layer
		self.out = 0 # Output of this layer (f(wx))
		self.out_der = 0 # Derivative of the output of this layer (f'(wx))
		self.type = "ANN" # Type of the layer


	def activate(self):
		prev_out = self.prev.get_out()[0]

		prev_out = self.stretch(prev_out)

		if self.nb_ins != self.prev.nb_neurons+1: # This if loop should only be satisfied when connected to an ANN and only done so once PLEASE!!!  Make sure of this
			self.nb_ins = self.prev.nb_neurons+1 
			self.w = self.init_w((self.nb_ins,self.nb_neurons)) 


		self.input = np.ndarray.tolist(prev_out) # Getting the output of the previous layer


		self.input.append([1]) # converting it to a list, appending 1 to it 
		self.input = np.array(self.input) # and converting it back to  np array
		wx = np.dot(self.w.T,self.input) # Matrix multiplication of W (input * nb_neutron) and input

		self.out = self.act_fnc(wx) # Activated output

		self.out_der = self.act_fnc(wx,True) # Derivative of activated output

	def get_out(self):
		return (self.out,self.out_der) # Return activated output and derivative of activated output

	def update_w(self,new):
		self.w = new

	def stretch(self,y):
		new_prev_out =[]
		if len(y.shape) >2:
			new_row = []
			for i in range(y.shape[0]):
				for j in range(y.shape[1]):
					for k in range(y.shape[2]):
						new_row.append(y[i][j][k])
			new_prev_out.append(np.array(new_row))

			return np.array(new_prev_out).T

		else:
			return y
