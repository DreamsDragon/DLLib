#This file contain the input Layer

#Input layers are common for different neural nets

#The number of neurons in the input layer depends on
#the size of the input data 

import numpy as np 

class Input_Layer():

	def __init__(self,values):
		self.size = values.shape
		self.values = value

	def get_out(self):
		return self.values