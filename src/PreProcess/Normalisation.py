import numpy as np


def get_min_max_med(data):
	total_size = data.shape[0]
	rows = data.shape[1]
	cols = data.shape[2]
	mins = []
	maxs = []
	meds = []
	for j in range(rows):
		var = []	
		for i in range(total_size):
			var.append(data[i][j][0])

		var = np.array(var)
		mins.append(np.min(var))
		maxs.append(np.max(var))
		meds.append(np.median(var))

	return mins,maxs,meds


def min_max_normalisation(data):
	mins,maxs,medians = get_min_max_med(data)

	rows = data.shape[1]
	total_size = data.shape[0]
	cols = data.shape[2]

	new_data = data
	for i in range(total_size):
		for j in range(rows):
			for k in range(cols):
					new_data[i][j][k] = (new_data[i][j][k] - mins[j])/(maxs[j] - mins[j])
	return new_data

def min_max_denormalisation(data,extremes):
	mins,maxs,medians = extremes
	rows = data.shape[1]
	total_size = data.shape[0]
	cols = data.shape[2]

	new_data = data
	for i in range(total_size):
		for j in range(rows):
			for k in range(cols):
				new_data[i][j][k] = ((new_data[i][j][k]*(maxs[j] - mins[j]))+ mins[j])

	return new_data
