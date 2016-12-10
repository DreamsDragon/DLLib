"""

The file contains the code for Trainer for Neural Networks

Author : Karunakar Gadireddy , Sreekar Kamireddy

Dated : 6 December 2016

"""

from PreProcess.Xcl_read import *
from PreProcess.Normalisation import *
from PreProcess.Data_Handling import *

class DataManager():

	def __init__(self,file_name,no_of_ins,number_of_outs,precents = (60,30,10),normalization = None):
		self.file_name = file_name
		self.nb_ins = no_of_ins
		self.nb_outs = number_of_outs
		self.percents = precents
		self.normal = normalization


	def read_file(self):
		if ".txt" in self.file_name:
			return self.read_txt()

		if ".xls" or ".xlsx" in self.file_name:
			return self.read_xl()

	def read_xl(self):
		x,y  = xl_cnv(self.file_name,self.nb_ins,self.nb_outs)

		if self.normal!=None:
			x = self.normalise(x,self.normal)
			y = self.normalise(y,self.normal)

		x_train,x_val,x_test = break_data(x,self.percents)
		y_train,y_val,y_test = break_data(y,self.percents)
	
		x_ret = (x_train,x_val,x_test)
		y_ret = (y_train,y_val,y_test)


		return (x_ret,y_ret)

	def read_txt(self):
		x = []
		y = []

		f = open(self.file_name)

		for lines in f:
			splt = lines.split(" ")
			x_row = []
			y_row = []
			for i in range(self.nb_ins):
				x_row.append(float(splt[i]))
			for j in range(self.nb_ins,self.nb_outs):
				y_row.append(float(splt[j]))

			x.append(np.array(x_row))
			y.append(np.array(y_row))
		


		x = np.array(x)
		y = np.array(y)

		if self.normal!=None:
			x = self.normalise(x,self.normal)
			y = self.normalise(y,self.normal)
		
		x_train,x_val,x_test = break_data(x,self.percents)
		y_train,y_val,y_test = break_data(y,self.percents)
	
		x_ret = (x_train,x_val,x_test)
		y_ret = (y_train,y_val,y_test)


		return (x_ret,y_ret)
			

	def normalise(self,x,type = None):
		if type == "Min Max":
			return min_max_normalisation(x)



