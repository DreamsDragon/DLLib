from Basics.Initialisations import *
from Basics.ActivationFunction import *
from Basics.optimizers import *
from Basics.backprop import *
from Basics.Pre_Process import *
from Layers.ALL_Layers import *
from Layers.Save_Load import *
from Basics.Parallel_DE import *
import numpy as np
import pickle


import random

'''
def vectorise(x,s):
    out = []

    for i in x:
        y = []
        for z in range(s):
            y.append(0)
        y[i] = i
        out.append(y)
    return np.array(out)

train,val,test = pickle.load(gzip.open('/home/dreamsd/dev/SE306/Data Set/MNIST/mnist.pkl.gz'))
x_train,y_train = train
x_val,y_val = val
x_test,y_test = test
''
'f = open("Data_3.txt",'r')

x_train = []
y_train = []

for lines in f:
    x,y = lines.split(' ')
    x_train.append([float(x)])
    y_train.append([float(y)])

x_train = np.array(x_train)
y_train = np.array(y_train)

input_layer = Input_Layer(np.zeros((1,1)))
hidden_layer_1 = ANN(input_layer,5,relu,random_normal_init,random_normal_init)
hidden_layer_2 = ANN(hidden_layer_1,5,Prelu,random_initialisation,random_initialisation)
output_layer = ANN(hidden_layer_2,1,Prelu,random_normal_init,random_normal_init)

layers  = {0:input_layer,1:hidden_layer_1,2:hidden_layer_2,3:output_layer}
anntrainer = trainer(layers,50)

#y_train = vectorise(y_train,10)
anntrainer.give_x_y(x_train,y_train,x_train,y_train,x_train,y_train)
anntrainer.start_training()
#para_differential_evln(layers,(x_in,y_in),1000,200)
#differential_evln(layers,(x_in,y_in),1000,200)
#saving_params('DE_200pop_1000gen',layers)

while True:
    inp = raw_input("Enter a number btw 0 and "+str(10000)+": ")
    if inp == "break":
        break
    else:
        inp = int(inp)
    for i in range(len(layers)):
        if i == 0:
            layers[i].set_input(np.array(inp))
            continue

        layers[i].activate()

    print "The value of output for ",inp," is ",layers[len(layers)-1].get_out()[0]
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
batch_size = 1
np.random.seed(42)


x,y = xl_cnv('/home/dreamsd/dev/SE306/Data Set/Inputs to MEC Students/mpp_dataset_v1_13-inputs.xls')
#x = forawrd_sig_convert(x)
y = (2*((y- min(y)))/(max(y)-min(y)))-1

x_train,x_val,x_test = break_data(x,(60,30,10))
y_train,y_val,y_test = break_data(y,(60,30,10))

print max(y_test)

no_of_ins = len(x_train[0])
# Create the network
layer_in = Input_Layer(np.zeros((no_of_ins,batch_size)))
Layer_1 = ANN(layer_in,20,sigmoid,random_normal_init,random_normal_init,batch_size)
Layer_2 = ANN(Layer_1,20,sigmoid,random_normal_init,random_normal_init,batch_size)

layer_out = ANN(Layer_2,1,tanh,random_normal_init,random_normal_init,batch_size)





n_train_batches = len(x_train)//batch_size

n_val_batches = len(x_val)//batch_size

#train_batches = batch_generation(x_train,y_train,batch)
#val_batches = batch_generation(x_val,y_val,batch)
layers = {0:layer_in,1:Layer_1,2:Layer_2,3:layer_out}
differential_evln(layers,(x_train,y_train),1000,200,1)
saving_params('DE_200pop_1000gen_indusdat',layers)

#anntrainer = trainer(layers,100)
#anntrainer.give_x_y(x_train,y_train,x_val,y_val,x_test,y_test)
#anntrainer.start_training()
valid_error = 0
for inp in range(len(x_val)):
        for i in range(len(layers)):
            if i == 0:
                layers[i].set_input(np.array(x_val[inp]))
                continue

            layers[i].activate()
        valid_error+=abs(layers[len(layers)-1].get_out()[0]-y_val)
        print "The value of output for ",x_val[inp]," is ",layers[len(layers)-1].get_out()[0]," Actual answer is ",y_val[inp]
'''
while True:
    inp = raw_input("Enter a number btw 0 and "+str(len(x_train))+": ")
    if inp == "break":
        break
    else:
        inp = int(inp)
    for i in range(len(layers)):
        if i == 0:
            layers[i].set_input(np.array(x_train[inp]))
            continue

        layers[i].activate()

    print "The value of output for ",x_train[inp]," is ",layers[len(layers)-1].get_out()[0]," Actual answer is ",y_train[inp]
#anntrainer.give_x_y(x_train,y_train,x_train,y_train,x_train,y_train)
#anntrainer.start_training()
'''