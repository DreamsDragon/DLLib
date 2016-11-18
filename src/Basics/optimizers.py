# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:16:01 2016

@author: sreekar
"""
import collections
import random
import numpy as np
from Initialisations import *
import time
# Objective Functions

def sgd(gradients,parameters,learning_rate = 0.05):
    updates = dict()
    for p,grad in zip(parameters,gradients):
    	if p == 0:
    		continue

        updates[p] = (parameters[p][0] - learning_rate*gradients[grad][0],parameters[p][1] - learning_rate*gradients[grad][1])
        parameters[p] = updates[p]
    return updates
    
def sgd_book(gradients,parameters,learning_rate  = 0.05):
	updates = {}
	
	for k in range(len(gradients)):
		i = k+1


		updates[i] = (parameters[i][0] - learning_rate*gradients[i][0],parameters[i][1] - learning_rate*gradients[i][1])
	return updates

def init_var(fnc,pop_size,layers):
	vals = []
	no_of_lyrs = len(layers)
	for i in range(pop_size):
		pop = []
		for j in range(1,no_of_lyrs):
			no_of_cols = layers[j].prev_layer.num_units+1
			no_of_rows = layers[j].num_units+1
			app = fnc((no_of_rows,no_of_cols))
			pop.append(app)
		vals.append(pop)
	return np.array(vals)
	#Every app is a weight matrix in which each row represents the weights of all the connections to that row and the
	#last column is the bias of that neuron

def gene_mutant_vector(var,gen_no,r_fac = 0.5,ffactor =0.3):
	total_gene_pool = []
	pop_size = len(var)
	#for i in range(pop_size):
		#total_gene_pool.append(var[i])

	for i in range(pop_size):

		mutant_vector = []
		(k1,k2,k3) = random.sample(range(pop_size),3)
		for j in range(len(var[i])):
			mutant_gene = var[i][j] + r_fac*(var[k1][j] - var[k2][j]) + ffactor*(var[k2][j] - var[k3][j])
			mutant_vector.append(mutant_gene)
		total_gene_pool.append(mutant_vector)

	return (total_gene_pool)

#Training dataset should be a tuple with inputs as 1st element and outputs as the 2nd element
def ftn_fnc(var,layers,dataset,div = 10):
	train_in = dataset[0]
	train_out = dataset[1]
	ftn_vals = []
	no_of_lyrs = len(layers)
	no_of_ins = len(train_in)/div
	pop_size = len(var)
	#start_time = time.time()
	
	for i in range(pop_size):
		#Setting W_b_values
		for j in range(no_of_lyrs-1):
			(W,b) = get_W_b_mat(var[i][j])
			layers[j+1].set_w_b((W,b))
		#Calc_fnc
		ftn_val = 0
		for inpt in range(no_of_ins):
			ftn_val+=get_indi_err(layers,train_in[inpt],train_out[inpt])
			'''no_of_layers = len(layers)

			for p in range(no_of_layers):
				if p == 0:
					layers[p].set_input(train_in[inpt])
					continue
				layers[p].activate()

			layer_out = layers[no_of_layers-1].get_out()[0]
			ftn_val+=abs(layer_out[0] - train_out[inpt][0]) '''
			#print "Time taken is ",(time.time()-start_time),i
		ftn_vals.append(ftn_val)

	return ftn_vals

def shuffle(dataset):
	x = []
	for i in range(len(dataset[0])):
		x.append((dataset[0][i],dataset[1][i]))
	np.random.shuffle(np.array(x))
	data = ([],[])
	for i in range(len(dataset[0])):
		data[0].append(x[i][0])
		data[1].append(x[i][1])
	return data

#Most time consuming function try to change if possible
def get_indi_err(layers,in_val,out):
	no_of_layers = len(layers)
	
	for i in range(no_of_layers):
		if i == 0:
			layers[i].set_input(in_val)
			continue
		layers[i].activate()

	layer_out = layers[no_of_layers-1].get_out()[0]

	return abs(out[0]-layer_out[0])#Check this part if more than one output is causing an issue

def my_sort(var,ftn_vals):
	pop_size = len(var)/2
	for i in range(len(var)):
		for j in range(len(var)):
			if ftn_vals[i]<ftn_vals[j]:
				(ftn_vals[i],ftn_vals[j]) = (ftn_vals[j],ftn_vals[i])
				(var[i],var[j]) = (var[j],var[i])
	return (var[0:pop_size],ftn_vals[0:pop_size])


def get_W_b_mat(line):
	no_of_rows = len(line)-1
	W = []
	b = []

	for i in range(no_of_rows):
		row = []
		no_of_cols = len(line[i])-1
		for j in range(no_of_cols):
			row.append(line[i][j])
		W.append(row)
		b.append(line[i][len(line[i])-1])
	return(W,b)	

def differential_evln(layers,dataset,no_of_generations=500,pop_size=200,div = 10):
   	var = []

   	var = init_var(random_initialisation,pop_size,layers)
   	old_ftn_vals = ftn_fnc(var,layers,dataset,div)
   	for i in range(no_of_generations):	
   		mut_var  = gene_mutant_vector(var,i)
   		if i%100 == 0:
   			dataset = shuffle(dataset)
   		new_ftn_vals = ftn_fnc(mut_var,layers,dataset,div)
   		ftn_vals = old_ftn_vals+new_ftn_vals
   		var = np.concatenate((var,mut_var))
   		var,old_ftn_vals = my_sort(var,ftn_vals)

   		print "Generation Number is ",(i+1)," Error is ",min(ftn_vals)
   	W_b_pairs_op(var[ftn_vals.index(min(ftn_vals))],len(layers)-1,layers)

def W_b_pairs_op(array,size,layers):
	for i in range(size):
		(W,b)=get_W_b_mat(array[i])
		layers[i+1].set_w_b((W,b))

