# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 08:52:00 2016

@author: DreamsD
"""

import numpy as np
import pickle
import gzip
import random
from optimizers import *
from Graphing import *
def check_accuracy(y,out):
    class_y = 0
    acc = 0
    for i in range(len(y)):
        for j in range(len(y[i])):
            if y[i][j]!=0:
                if  np.argmax(out.T[i]) == y[i][j]:
                    acc+=1
                break
    return acc
def batch_generation(x,y,N):
    while True:
        idx = np.random.choice(len(y),N)
        yield x[idx].astype('float32'),y[idx].astype('int32')

def get_next(x,y,batch_size,iter,nb_batches):

    if iter == nb_batches - 1:
        start = batch_size*iter
        end = len(x)
    else:
        start = batch_size*iter
        end = batch_size*(iter+1)

    a = np.ndarray.tolist(x)
    b = np.ndarray.tolist(y)

    return (np.array(a[start:end]),np.array(b[start:end]))

class trainer():
    """

    To use this class initialise it with a dictionary of layers
    Additional options to initialise with the the number of epochs and the batch size are also provided

    Create an object of this class and give it the input values needed using the give_x_y() method

    Call the start_training() method to start training the ANN 

    """
    def __init__(self,layers,epoch = 100 ,batch = 1):
        self.layers = layers   #The dictionary which contains all the layer objects
        self.num_layers = len(layers)# Number of layers in the ANN
        self.parameters = {} # Weights and biases of each layer

        #The weights and biases of the layers are added to the above library
        for h in range(len(self.layers)):
            if h == 0 :
                continue
            self.parameters[h] = (self.layers[h].W,self.layers[h].b)
        self.x_train = 0 # Training input values
        self.y_train = 0 # Training output values
        self.x_val = 0 # Validation input values
        self.y_val = 0 # Validation output values
        self.x_test = 0 # Testing input value
        self.y_test = 0 # Testing output value
        self.n_train_batches = 0
        self.n_val_batches = 0
        self.epoch = epoch # Number of epochs
        self.batch = batch # Batch size
        self.show_graph = False # To check if drawing a graph is needed
        self.error_grapher = None # Grapher of the trainer
        self.data_grapher = None

    def set_show_graph(self,set_val = True):
        self.show_graph = set_val

    def give_x_y(self,x_train,y_train,x_val,y_val,x_test,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.n_train_batches = len(self.x_train)//self.batch
        self.n_val_batches = len(self.x_val)//self.batch
        np.random.shuffle(x_train)
        np.random.shuffle(y_train)
        
        #self.n_train_batches = len(x_train)//self.batch

        #elf.n_val_batches = len(x_val)//self.batch
    def cal_validation_error(self,x_val,y_val):
        val_error = 0
        for i in range(len(x_val)):
            x = np.array([x_val[i]])
            y = np.array([y_val[i]])
            for lyr in range(len(self.layers)):
                if lyr == 0:
                    self.layers[lyr].set_input(x.T)
                    continue
                self.layers[lyr].activate()
                #print self.layers[lyr].out_val
                #if lyr ==2:
                    #print self.layers[lyr].input
            # Feed forward ends
            output = self.layers[self.num_layers-1].get_out()[0]
            error = 0.5*((y.T-output)**2)            
            err = np.mean(error)
            val_error+=err
        return val_error

    def start_training(self,learning_rate = 0.5,division = 30):
        train_batches = batch_generation(self.x_train,self.y_train,self.batch)
        val_batches = batch_generation(self.x_val,self.y_val,self.batch)
        lrate = learning_rate
        div = division
        if self.show_graph == True:
           self.error_grapher = grapher()
           self.error_grapher.add_line("Training Error")
           self.error_grapher.add_line("Validation Error")
        for epc in range(self.epoch):
            train_error = 0
            val_accuracy = 0
            for i in range(self.n_train_batches):
                sel = i
                #sel = random.randint(0,len(self.x_train)-1)
                x,y = get_next(self.x_train,self.y_train,self.batch,i,self.n_train_batches)#np.array([self.x_train[sel]]),np.array([self.y_train[sel]]) # To change the input size and the output size for batches modify these variables and weights and biases accordinly

                #x,y = next(train_batches)
                # Feed forward loop
                for lyr in range(len(self.layers)):
                    if lyr == 0:
                        self.layers[lyr].set_input(x.T)
                        continue
                    self.layers[lyr].activate()
                    #print self.layers[lyr].out_val
                    #if lyr ==2:
                        #print self.layers[lyr].input
                # Feed forward ends
                output = self.layers[self.num_layers-1].get_out()[0]
                #if i == 1:
                    #print output.T
                #print y.T,output,epc
                error = 0.5*((y.T-output)**2)
                #print output

                epoch_accuracy = 0

                grads = self.backprop('MS',y,output)
                if epc!=0 and epc%div == 0:
                    lrate = lrate/10
                upd = sgd(grads,self.parameters,lrate)
                for lyr in range(len(self.layers)):
                    if lyr == 0:
                        continue
                    self.layers[lyr].set_w_b(upd[lyr])
                err = np.mean(error)
                train_error += err
                val_accuracy+=check_accuracy(y,output)
            val_error = self.cal_validation_error(self.x_val,self.y_val)
            if self.show_graph == True:
                self.error_grapher.update_pyl_plot((epc+1,train_error),'Training Error')
                self.error_grapher.update_pyl_plot((epc+1,val_error),'Validation Error')
            print "epoch is ",epc," error is : ",train_error," Validation Error is ",val_error

    def get_result(self,input_vec):
        for lyr in range(len(self.layers)):
            if lyr == 0:
                self.layers[lyr].set_input(input_vec.T)
                continue
            self.layers[lyr].activate() 
        return self.layers[self.num_layers-1].get_out()[0]
    def set_epoch(self,epoch):
        self.epoch = epoch

    def set_batch(self,batch):
        self.batch = batch

    def backprop(self,error,target,output):
        grads = {}
        no_of_layers = len(self.layers)-1
        if error == 'MS':
            delta_out = np.multiply((output - target.T),self.layers[no_of_layers].get_out()[1])
        dWo = np.dot(delta_out,self.layers[no_of_layers].input.T)
        dbo = delta_out
        grads[no_of_layers] = (dWo,dbo)
        delta_prev = delta_out
        for i in reversed(self.layers.keys()):
            if i == no_of_layers or i == 0:
                continue

            delta = np.multiply(np.dot(self.layers[i+1].W.T,delta_prev),self.layers[i].get_out()[1])

            dW = np.dot(delta[:np.newaxis],self.layers[i-1].get_out()[0].T)

            db = delta


            grads[i] = (dW,db)

            delta_prev = delta

        return grads

    
 
    
    def backprop_book(self,error,target,output):
        grads = {}
        for i in reversed(self.layers.keys()):  
            if (i == 0):
                continue

            dw = (output - target)*self.layers[i].get_out()[1]*self.layers[i].get_out()[0]
            
            grads[i] = (dw,dw)
        return grads


def sto_backprop(error,target,output,layers):   
    grads = {}
    no_of_layers = len(layers)
    delta_out = 0
    out_layer = 0
    dWodel = 0
    prev_w = 0
    prev_delta = 0

    if error == 'MS':
        diff = output-target

        out_layer = layers[no_of_layers-1]

        delta_out = np.multiply(diff,out_layer.get_out()[1])

        dWo = np.dot(delta_out,out_layer.input.T)

        dbo = delta_out

        grads[no_of_layers-1] = (dWo,dbo)

        prev_delta = delta_out

        prev_w = out_layer.W

    for i in reversed(layers.keys()):
        if i == no_of_layers-1 or i == 0:
            continue

        dWodel = np.dot(prev_w.T,prev_delta)

        delta = np.multiply(layers[i].get_out()[1],dWodel)

        dW = np.multiply(delta,layers[i-1].get_out()[0].T)

        db = delta

        grads[i] = (dW,db)

        prev_delta = delta

        prev_w = layers[i].W


            #print "Delta is ",delta
    return grads