import pickle
from ANN import *
from Input import *


def saving_params(filename,layers):

	pickle.dump(layers,open(filename+'.p','wb'))


def load_net(filename): 
	# Function to load pickle file
	ept_dic = {}
	net = pickle.load(open(filename,"rb"))

	return net

