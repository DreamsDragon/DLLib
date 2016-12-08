"""

The file contains the code for CNN

Author : Karunakar Gadireddy , Sreekar Kamireddy

Dated : 8 December 2016

"""

import numpy as np 





class CNN():

	def __init__(self,filter_size,nb_filters,stride,activation_fnc,prev,init_w,padding = True):

		self.filter_size = filter_size # Filter size 
		self.stride  = stride #Stride
		self.prev = prev #Previous Layer
		self.w = [] # Weights

		for i in range(nb_filters):
			self.w.append(init_w((filter_size[0],filter_size[1])))
		self.activation_fnc = activation_fnc
		self.input = 0 #Input to this layer
		self.out = [] #Output of the layer
		self.out_der = [] #Derivative of the output of this layer
		self.pad = padding # Boolean to see if padding is required
		self.nb_filters = nb_filters # Number of filters in this layer

	def padding(self):
		
		nb_pad_rows = self.filter_size[0]-1
		nb_pad_cols = self.filter_size[1]-1

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


	def convolve(self,x,w):
		c = np.multiply(x,w)
		sums = 0
		for i in range(len(c)):
			for j in range(len(c[i])):
				sums+=c[i][j]

		return sums

	def get_x(self,x,y,ins):
		z = []
		for i in range(self.filter_size[0]):
			x_row = []
			for j in range(self.filter_size[1]):
				x_row.append(self.input[ins][x+i][y+j])
			z.append(np.array(x_row))

		return np.array(z)

	def activate(self):
		self.input = self.prev.get_out()[0]

		if self.pad == True:
			row_end = self.input.shape[1]
			col_end = self.input.shape[2]
			self.padding()

		else:
			row_end = self.input.shape[1]-self.filter_size[0]+1
			col_end = self.input.shape[2]-self.filter_size[1]+1
		

		for wnds in range(self.nb_filters):
			wnd_out = []
			wnd_der_out = []
			for i in range(0,row_end,self.stride):
				wnd_out_row = []
				wnd_der_out_row = []
				for j in range(0,col_end,self.stride):

					for chls in range(len(self.input)):
						x = self.get_x(i,j,chls)
						c = self.convolve(x,self.w[wnds])
						if chls == 0:
							wnd_out_row.append(c)
							wnd_der_out_row.append(c)
						else:
							wnd_out_row[j]+=c
							wnd_der_out_row[j]+=c

				wnd_out.append(self.activation_fnc(np.array(wnd_out_row)))
				wnd_der_out.append(self.activation_fnc(np.array(wnd_der_out_row),True))
			self.out.append(np.array(wnd_out))
			self.out_der.append(np.array(wnd_der_out))
		self.out = np.array(self.out)
		self.out_der = np.array(self.out_der)


			



	def get_out(self):
		return (self.out,self.out_der)

	def update_w(self,new_w):
		pass
