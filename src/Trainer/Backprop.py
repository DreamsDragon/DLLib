import numpy as np 


def backprop(net,t,o):
	deltas = {}
	prev_delta  = 0

	for i in range(len(net)-1,0,-1):
		if i == len(net)-1:
			out_delta = [1.0]*net[i].nb_neurons
			#Output Layer
			for k in range(net[i].nb_neurons):
				out_delta[k] = (o[k]-t[k])*net[i].get_out()[1][k]

			prev_delta = out_delta
			deltas[i] = out_delta

		else:
			#For Hidden Layers other than the layer infront of input
			hidden_delta = [1.0]*net[i].nb_neurons
			for j in range(net[i].nb_neurons):
				error = 0
				if (j == net[i].nb_ins -1):
					#Bias case
					pass
				for k in range(net[i+1].nb_neurons):
					error += prev_delta[k]*net[i+1].w[j][k]

				hidden_delta[j] = net[i].get_out()[1][j]*error

			prev_delta = hidden_delta
			deltas[i] = prev_delta



	return deltas