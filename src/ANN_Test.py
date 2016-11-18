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
import time

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
x_train,y_train = trainer
x_val,y_val = val
x_test,y_test = test
'''

f = open("Sine_data.txt",'r')

total_x = []
total_y = []

for lines in f:
    x,y = lines.split(' ')
    total_x.append([float(x)])
    total_y.append([float(y)])

total_y = np.array(total_y)    
total_y = (((total_y- min(total_y)))/(max(total_y)-min(total_y)))
#print max(total_y)
#print min(total_y)
print len(total_y)
x_train,x_val,x_test = break_data(total_x,(70,30,0))
y_train,y_val,y_test = break_data(total_y,(70,30,0))


input_layer = Input_Layer(np.zeros((1,1)))
hidden_layer_1 = ANN(input_layer,10,sigmoid,random_normal_init,random_normal_init)
#hidden_layer_2 = ANN(hidden_layer_1,3,Prelu,random_normal_init,random_normal_init)
output_layer = ANN(hidden_layer_1,1,tanh,random_normal_init,random_normal_init)

layers  = {0:input_layer,1:hidden_layer_1,2:output_layer}
anntrainer = trainer(layers,100,1)

#y_train = vectorise(y_train,10)
print y_train
anntrainer.give_x_y(x_train,y_train,x_val,y_val,x_train,y_train)
anntrainer.set_show_graph()
anntrainer.start_training()
#para_differential_evln(layers,(x_in,y_in),1000,200)
#differential_evln(layers,(x_train,y_train),1000,200)
#saving_params('DE_200pop_1000gen_sine',layers)
while True:
    inp = raw_input("Enter a number btw 0 and "+str(len(x_val))+": ")
    if inp == "break":
        break
    else:
        inp = float(inp)
    for i in range(len(layers)):
        if i == 0:
            layers[i].set_input(np.array([x_val[inp]]))
            continue

        layers[i].activate()

    print "The value of output for ",x_val[inp]," is ",layers[len(layers)-1].get_out()[0]," actual value is ",y_val[inp]



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
#x 0000+++++= forawrd_sig_convert(x)
#y = (2*((y- min(y)))/(max(y)-min(y)))-1

x_train,x_val,x_test = break_data(x,(60,30,10))
y_train,y_val,y_test = break_data(y,(60,30,10))

print max(y_test)

no_of_ins = len(x_train[0])
# Create the network
layer_in = Input_Layer(np.zeros((no_of_ins,batch_size)))
Layer_1 = ANN(layer_in,40,Prelu,random_normal_init,random_normal_init,batch_size)
Layer_2 = ANN(Layer_1,30,Prelu,random_normal_init,random_normal_init,batch_size)
Layer_3 = ANN(Layer_2,20,Prelu,random_normal_init,random_normal_init,batch_size)
Layer_4 = ANN(Layer_3,10,Prelu,random_normal_init,random_normal_init,batch_size)
#Layer_5 = ANN(Layer_4,10,Prelu,random_normal_init,random_normal_init,batch_size)

layer_out = ANN(Layer_4,1,Prelu,random_normal_init,random_normal_init,batch_size)





n_train_batches = len(x_train)//batch_size

n_val_batches = len(x_val)//batch_size

#train_batches = batch_generation(x_train,y_train,batch)
#val_batches = batch_generation(x_val,y_val,batch)
layers = {0:layer_in,1:Layer_1,2:Layer_2,3:Layer_3,4:Layer_4,5:layer_out}
#differential_evln(layers,(x_train,y_train),100,60,1)
#saving_params('DE_200pop_100gen_indusdat_relu',layers)
#layers = load_net("DE_200pop_1000gen_indusdat.p")
anntrainer = trainer(layers,100)
anntrainer.give_x_y(x_train,y_train,x_val,y_val,x_test,y_test)
anntrainer.start_training()

start_time = time.time()
for i in range(len(layers)):
            if i == 0:
                layers[i].set_input(np.array(x_val[0]))
                continue

            layers[i].activate()

print "Time Taken is ",(time.time()-start_time)

start_time = time.time()
for j in range(200):
    for i in range(len(x_val)):
        get_indi_err(layers,x_val[i],y_val[i])
        print "Time taken is ",(time.time()-start_time),j

valid_error = 0
epi = 0.1
correct_count = 0
for inp in range(len(x_val)):
        for i in range(len(layers)):
            if i == 0:
                layers[i].set_input(np.array([x_val[inp]]).T)
                continue

            layers[i].activate()
        if abs(layers[len(layers)-1].get_out()[0]-y_train[0]) <=epi:
            correct_count+=1
        valid_error+=abs(layers[len(layers)-1].get_out()[0]-y_val[0])**2
        print "The value of output for ",x_val[inp]," is ",layers[len(layers)-1].get_out()[0]," Actual answer is ",y_val[inp]

print "The total validation error is ",valid_error , "Total accuracy is ",correct_count
    
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
