import numpy as np 



def sgd(net,deltas,lrate = 0.05):
	new_ws = {}
	for i in net:
		if (i == 0):
			continue

		w = net[i].w
		for j in range(len(w)):
			for k in range(len(w[j])):
				w[j][k] += lrate * deltas[i][k] * net[i].input[j]

		new_ws[i]= w
	return new_ws
