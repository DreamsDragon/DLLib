"""

The file contains the code for CNN

Author : Karunakar Gadireddy , Sreekar Kamireddy

Dated : 10 December 2016

"""


"""

Trick to go around backprop for pooling layer, take the weights of the next layer and pass it down as its own

Since the weights are not used anywhere in this layer this should be fine

"""


import numpy as np 


class Pooling():

	def __init__(self,filter_size,nb_filters,Pooling_type,prev,padding = True):

		self.window_size = filter_size
		self.nb_windows = nb_filters
		self.stride = self.window_size[0]
		self.prev = prev
		self.pad = padding
		self.pool_type = Pooling_type
		self.type = "Pooling"
		self.input = []
		self.out = []

	def get_x(self,x,y,ins):
		z = []
		for i in range(self.window_size[0]):
			x_row = []
			for j in range(self.window_size[1]):
				x_row.append(self.input[ins][x+i][y+j])
			z.append(np.array(x_row))

		return np.array(z)

	def padding(self):
		
		nb_pad_rows = self.window_size[0]-1
		nb_pad_cols = self.window_size[1]-1

		new_ins = []

		for ins in range(len(self.input)):
			old_in = self.input[ins].tolist()

			for i in range(len(old_in)):
				for k in range(nb_pad_cols):
					old_in[i].append(0)

			old_in = np.array(old_in)
			for i in range(nb_pad_rows):
				z = np.zeros((1,len(old_in[0])))
				old_in = np.vstack((old_in,z))

			new_ins.append(np.array(old_in))

		self.input = np.array(new_ins)


	def activate(self):

		self.input = self.prev.get_out()[0]

		self.out = []
		
		if self.pad == True:
			row_end = self.input.shape[1]
			col_end = self.input.shape[2]
			self.padding()

		else:
			row_end = self.input.shape[1]-self.window_size[0]+1
			col_end = self.input.shape[2]-self.window_size[1]+1
		

		for wnds in range(self.nb_windows):
			wnd_out = []
			for i in range(0,row_end,self.window_size[0]):
				wnd_out_row = []
				last_j =-1
				for j in range(0,col_end,self.window_size[1]):
					for chls in range(len(self.input)):
						x = self.get_x(i,j,chls)
						c = self.pool(x)
						if chls == 0:
							wnd_out_row.append(c)
							last_j+=1
						else:
							wnd_out_row[last_j]+=c

				wnd_out.append(np.array(wnd_out_row))

			self.out.append(np.array(wnd_out))

		self.out = np.array(self.out)

	def pool(self,x):

		if self.pool_type == "Max":
			max_now = x[0][0]
			for i in range(len(x)):
				for j in range(len(x[0])):
					if x[i][j]>max_now:
						max_now = x[i][j]
			return max_now

		if self.pool_type == "Avg":
			sum_now = 0
			total = len(x)*len(x[0])
			for i in range(len(x)):
				for j in range(len(x[0])):
					sum_now += x[i][j]
			return sum_now/total

	def get_out(self):
		return (self.out,None)
