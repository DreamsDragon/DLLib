#Author - DreamsDragon

# This file contains all the functions which are used
# to initialies the weights and biases

import numpy as np 


# Randomly initialises to values between 0 and 1
def random_initialisation(size,rang= (-1,1)):
	#np.random.seed(0)
	return np.random.uniform(rang[0],rang[1],size)
		
def random_normal_init(size,W_b = 'W'):
	#np.random.seed(0)
	if W_b == 'b':
		bias = []
		rnd = np.random.normal(size = size[0])
		for i in range(size[1]):
			bias.append(rnd)
		return np.array(bias).T
	else:
		return np.random.normal(size = (size[0],size[1]))