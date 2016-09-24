import random
import numpy as np
from Initialisations import *

from joblib import Parallel, delayed
import multiprocessing


num_cores = multiprocessing.cpu_count()

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
	for i in range(pop_size):
		total_gene_pool.append(var[i])

	for i in range(pop_size):

		mutant_vector = []
		(k1,k2,k3) = random.sample(range(pop_size),3)
		for j in range(len(var[i])):
			mutant_gene = var[i][j] + r_fac*(var[k1][j] - var[k2][j]) + ffactor*(var[k2][j] - var[k3][j])
			mutant_vector.append(mutant_gene)
		total_gene_pool.append(mutant_vector)

	return (total_gene_pool)

def check(a,b):
	return a*b
#Training dataset should be a tuple with inputs as 1st element and outputs as the 2nd element
def ftn_fnc(var,layers,dataset):
	train_in = dataset[0]
	train_out = dataset[1]
	ftn_vals = []
	no_of_lyrs = len(layers)
	no_of_ins = len(train_in)
	pop_size = len(var)
	ftn_vals = Parallel(n_jobs = num_cores-1)(delayed(ftn_for_pop)(layers,var,i,train_in,train_out) for i in range(pop_size))
	
	return ftn_vals
		
def ftn_for_pop(layers,var,i,train_in,train_out):
		no_of_lyrs = len(layers)
		no_of_ins = len(train_in)
		for j in range(no_of_lyrs-1):
			(W,b) = get_W_b_mat(var[i][j])
			layers[j+1].set_w_b((W,b))
		#Calc_fnc
		ftn_val = 0
		for inpt in range(no_of_ins):
			ftn_val+=get_indi_err(layers,train_in[inpt],train_out[inpt])
		return ftn_val

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
	return var[0:pop_size]


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

def para_differential_evln(layers,dataset,no_of_generations=500,pop_size=200):
   	var = []

   	var = init_var(random_initialisation,pop_size,layers)
   	for i in range(no_of_generations):
   		var  = gene_mutant_vector(var,i)
   		if i%100 == 0:
   			dataset = shuffle(dataset)
   		ftn_vals = ftn_fnc(var,layers,dataset)
   		var = my_sort(var,ftn_vals)
   		print "Generation Number is ",(i+1)," Error is ",min(ftn_vals)
   	W_b_pairs_op(var[ftn_vals.index(min(ftn_vals))],len(layers)-1,layers)

def W_b_pairs_op(array,size,layers):
	for i in range(size):
		(W,b)=get_W_b_mat(array[i])
		layers[i+1].set_w_b((W,b))

