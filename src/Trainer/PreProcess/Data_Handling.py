import numpy as np


def break_data(data,pers):

	size = len(data)
	split_per_1 = int(size*(pers[0]/100.0))
	split_per_2 = int(size*(pers[1]/100.0))
	split_per_3 = size - split_per_1 - split_per_2
	split_data_1 = np.array(data[:split_per_1])
	split_data_2 = np.array(data[split_per_1:split_per_1+split_per_2])
	split_data_3 = np.array(data[split_per_1+split_per_2:split_per_1+split_per_2+split_per_3])

	return (split_data_1,split_data_2,split_data_3)