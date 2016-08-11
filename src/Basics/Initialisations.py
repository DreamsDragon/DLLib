#Author - DreamsDragon

# This file contains all the functions which are used
# to initialies the weights and biases

import numpy as np 

# Randomly initialises to values between 0 and 1
def random_initialisation(size):
	print size
	return np.random.rand(size[0],size[1])
