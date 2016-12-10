from PIL import Image
import numpy as np


class Cnv_Input():
	""" Convolution Neural Network Input"""

	def __init__(self,filename):
		self.filename = filename
		self.type = "CNN_Input"
		
	def give_in_file(self,new_filename):
		self.filename = new_filename

	def get_out(self):
		if isinstance(self.filename,str):
			return (self.get_img_data(),None)
		
		else:
			return (self.filename,None)
			


	def get_img_data(self):
		im = Image.open(self.filename)
		pix = im.load()
		channels = []

		row,col = im.size

		for i in range(len(pix[0,0])):
			ch = []

			for j in range(row):
				ch_row = []
				for k in range(col):
					ch_row.append(pix[j,k][i])

				ch.append(ch_row)

			channels.append(np.array(ch))

		return np.array(channels)