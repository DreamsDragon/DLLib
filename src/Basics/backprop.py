# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 08:52:00 2016

@author: sreekar
"""

import numpy as np
import pickle
import gzip
from optimizers import *

def batch_generation(x,y,N):
    while True:
        idx = np.random.choice(len(y),N)
        yield x[idx].astype('float32'),y[idx].astype('int32')

class trainer():
    def __init__(self,layers,epoch = 10,batch = 64):
        self.layers = layers   #The dictionary which contains all the layer objects
        self.num_layers = len(layers)
        self.parameters = {}
        for i in range(len(layers)):
            if i == 0 :
                continue
            self.parameters[i] = (self.layers[i].W,self.layers[i].b)
        self.x_train = 0
        self.y_train = 0
        self.x_val = 0
        self.y_val = 0
        self.x_test = 0
        self.y_test = 0
        self.n_train_batches = 0
        self.n_val_batches = 0
        self.epoch = epoch
        self.batch = batch

    def give_x_y(self,x_train,y_train,x_val,y_val,x_test,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        
        self.n_train_batches = len(x_train)//self.batch

        self.n_val_batches = len(x_val)//self.batch

    def start_training(self):
        train_batches = batch_generation(self.x_train,self.y_train,self.batch)
        val_batches = batch_generation(self.x_val,self.y_val,self.batch)

        for epc in range(self.epoch):
            train_error = 0
            val_accuracy = 0
            for i in range(self.n_train_batches):
                x,y = next(train_batches)
                for lyr in range(len(self.layers)):
                    if lyr == 0:
                        self.layers[lyr].set_input(x.T)
                        continue
                    self.layers[lyr].activate()
                output = self.layers[self.num_layers-1].get_out()
                error = 0.5*((y-output)**2)
                epoch_accuracy = 0
                grads = self.backprop('MS',y,output)
                upd = sgd(grads,self.parameters)
                for lyr in range(len(self.layers)):
                    if lyr == 0:
                        continue
                    self.layers[lyr].set_w_b(upd[lyr])
                err = np.mean(error)
            train_error += err
            print "epoch is ",epc," error is : ",train_error/self.n_train_batches

    def set_epoch(self,epoch):
        self.epoch = epoch
    def set_batch(self,batch):
        self.batch = batch

    def backprop(self,error,target,output):
        grads = {}
        no_of_layers = len(self.layers)-1
        if error == 'MS':
            delta_out = np.multiply((output - target),self.layers[no_of_layers].get_out(True))
        dWo = np.dot(delta_out,self.layers[no_of_layers].input.T)
        dbo = delta_out
        grads[no_of_layers] = (dWo,dbo)
        delta_prev = delta_out
        for i in reversed(self.layers.keys()):
            if i == no_of_layers or i == 0:
                continue

            delta = np.multiply(np.dot(self.layers[i+1].W.T,delta_prev),self.layers[i].get_out(True))

            dW = np.dot(delta,self.layers[i].input.T)

            db = delta


            grads[i] = (dW,db)

            delta_prev = delta
        return grads

    
 
    
    
    
        
    