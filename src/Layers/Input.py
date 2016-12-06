"""

The file contains the code for Input Layer of a Neural Network

Author : Karunakar Gadireddy , Sreekar Kamireddy

Dated : 18 December 2016

"""


class Input():
	"Class for input layer of a Neural Network"

	def __init__(self,values):
		self.out = values
		self.nb_neurons = values.shape[0]
		self.type = "Input"

	def get_out(self):
		return (self.out,0)

	def set_in(self,x):
		self.out = x
		self.nb_neurons = x.shape[0]