#This file contain the input Layer

#Input layers are common for different neural nets

#The number of neurons in the input layer depends on
#the size of the input data 

import numpy as np 

class Input_Layer():

	def __init__(self,values):
		self.num_units = values.shape[0]
		self.values = values
		self.batch_size = values.shape[1]
	def get_out(self):
		return self.values

	def set_input(self,new_input):
		self.values = new_input