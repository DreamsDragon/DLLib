from Layers.ANN import *
from Basics.Initialisations import *
from Basics.ActivationFunction import *
from Layers.Input import *
from Basics.optimizers import *
from Basics.backprop import *

import numpy as np
import pickle

'''
f = open("/home/dreamsd/dev/SE306/Data.txt",'r')

x_in = []
y_in = []

for lines in f:
    x,y = lines.split(' ')
    x_in.append([float(x)])
    y_in.append([float(y)])

x_in = np.array(x_in)
y_in = np.array(y_in)


input_layer = Input_Layer(np.zeros((1,1)))
hidden_layer_1 = ANN(input_layer,3,Prelu,random_initialisation,random_initialisation)
output_layer = ANN(hidden_layer_1,1,Prelu,random_initialisation,random_initialisation)


layers ={0:input_layer,1:hidden_layer_1,2:output_layer}
anntrainer = trainer(layers)
anntrainer.give_x_y(x_in,y_in,x_in,y_in,x_in,y_in)


print "Input"

print input_layer.num_units

print "Hidden"

print "Num units are ",hidden_layer.num_units
print "Input is ",hidden_layer.input 
print "W is ",hidden_layer.W 
print "B is ",hidden_layer.b 
print "Output is",hidden_layer.out_val 

print "Output"

print "Num units are ",output_layer.num_units
print "Input is ",output_layer.input 
print "W is ",output_layer.W 
print "B is ",output_layer.b 
print "Output is",output_layer.out_val 


check_val = np.array([x_in[len(x_in)-1]])
anntrainer.start_training()

print "Acutal is ",y_in[len(x_in)-1]
print anntrainer.get_result(check_val)
'''
# Loading the data

def batch_generation(x,y,N):
    while True:
        idx = np.random.choice(len(y),N)
        yield x[idx].astype('float32'),y[idx].astype('int32')
        
def vectorise(x,s):
    out = []

    for i in x:
        y = []
        for z in range(s):
            y.append(0)
        y[i] = i
        out.append(y)
    return np.array(out)


epoch = 10
batch = 64
np.random.seed(42)



train,val,test = pickle.load(gzip.open('/home/dreamsd/dev/SE306/mnist.pkl.gz'))
x_train , y_train = train
x_val , y_val = val
x_test , y_test = test

y_train = vectorise(y_train,10)


# Create the network
layer_in = Input_Layer(np.zeros((784,batch)))
Layer = ANN(layer_in,800,relu,random_initialisation,random_initialisation)
layer_out = ANN(Layer,10,relu,random_initialisation,random_initialisation)



n_train_batches = len(x_train)//batch

n_val_batches = len(x_val)//batch

#train_batches = batch_generation(x_train,y_train,batch)
#val_batches = batch_generation(x_val,y_val,batch)

trainer_1 = trainer({0:layer_in,1:Layer,2:layer_out})

trainer_1.give_x_y(x_train,y_train,x_val,y_val,x_test,y_test)

trainer_1.start_training()
# Running to train the network



