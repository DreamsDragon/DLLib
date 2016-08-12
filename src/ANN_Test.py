from Layers.ANN import *
from Basics.Initialisations import *
from Basics.ActivationFunction import *
from Layers.Input import *
from Basics.optimizers import *
from Basics.backprop import *

import numpy as np
import pickle


import gzip

epoch = 10
batch = 64

np.random.seed(42)


# Loading the data

train,val,test = pickle.load(gzip.open('/home/dreamsd/dev/SE306/mnist.pkl.gz'))
x_train , y_train = train
x_val , y_val = val
x_test , y_test = test


print x_train.shape
def batch_generation(x,y,N):
    while True:
        idx = np.random.choice(len(y),N)
        yield x[idx].astype('float32'),y[idx].astype('int32')


# Create the network
layer_in = Input_Layer(np.zeros((784,batch)))
Layer = ANN(layer_in,800,relu,random_initialisation,random_initialisation)
layer_out = ANN(Layer,62,relu,random_initialisation,random_initialisation)




n_train_batches = len(x_train//batch)
n_val_batches = len(x_val//batch)

train_batches = batch_generation(x_train,y_train,batch)
val_batches = batch_generation(x_val,y_val,batch)

trainer_1 = trainer({1:Layer,2:layer_out})

# Running to train the network
for i in range(epoch):
	train_accuracy = 0
	val_accuracy = 0
	for i in range(n_train_batches):
		x,y = next(train_batches)
		layer_in.set_input(x.T)
		Layer.activate()
		layer_out.activate()
		output = layer_out.get_out()
		error = 0.5*((y-output)**2)
		epoch_accuracy = 0
		trainer_1.backprop('MS',y,output)





