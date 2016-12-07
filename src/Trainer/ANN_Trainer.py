"""

The file contains the code for Trainer for Neural Networks

Author : Karunakar Gadireddy , Sreekar Kamireddy

Dated : 18 November 2016

"""

from Backprop import *
from Update import *
from Grapher.Grapher import *
from PreProcess.Xcl_write import *


class ANN_Trainer():
	"""Trainer Class"""

	def __init__(self,layers,nb_epc,name,normalise = False):
		self.net = layers #Neural Network
		self.x_train = 0 #Training input values
		self.y_train = 0 #Target Training Values
		self.x_val = 0 # Validation input values
		self.y_val = 0 # Validation target values
		self.x_test = 0 # Test input values
		self.y_test = 0 # Test output values
		self.nb_epc = nb_epc # Number of epochs
		self.show_graph = False # Boolean to show or not show graphs
		self.error_grapher = 0 #Grapher to plot the error values
		self.graph_save = False #Boolean to save the graph or no
		self.graph_names = 0 # Graph names to save as
		self.name = name # Name of ANN

		self.pred_test = 0 #Predicted Test Data

	def show_graphs(self):	
		self.show_graph = True

	def forward_prop(self,x):
		for i in self.net:
			if self.net[i].type == "Input":
				self.net[i].set_in(x)
				continue
			self.net[i].activate()

		y = self.net[len(self.net)-1].get_out()[0]
		return y

	def set_train_data(self,x,y):
		self.x_train = x
		self.y_train = y

	def set_val_data(self,x,y):
		self.x_val = x
		self.y_val = y

	def set_test_data(self,x,y):
		self.x_test = x
		self.y_test = y

	def update_ws (self,new):
		for i in self.net:
			if (i == 0):
				continue
			self.net[i].update_w(new[i])

	def get_val_error(self):
		error = 0
		for i in range(len(self.x_val)):
			error+= abs(self.y_val[i] - self.forward_prop(self.x_val[i]))
		return error

	def get_test_error(self):
		error = 0
		test_outs = []
		test_ins = []
		original_outs = []
		indi_errors = []
		accuracy= 0
		for i in range(len(self.x_test)):
			out = self.forward_prop(self.x_test[i])
			original_out  = self.y_test[i]

			indi_error = abs(original_out - out)

			if indi_error<0.5:
				accuracy+=1.0
			error+= indi_error
			indi_errors.append(indi_error)
			test_outs.append(out)
			if len(self.x_test[i]) == 1:
				test_ins.append(self.x_test[i])
			else:
				test_ins.append(i)
			original_outs.append(original_out)

		print "Total Test Error is ",error," And the Accuray is ",accuracy/len(self.y_test)
		self.pred_test = np.array(test_outs)

		if self.graph_save == True:
			fig = plt.figure()
			ax1 = fig.add_subplot(111)
			ax1.scatter(test_ins,test_outs,label = "Predicted Values",c = 'b')
			ax1.scatter(test_ins,original_outs,label = "Actual Values",c = 'g')
			plt.legend()
			plt.savefig(self.name+"_test_data.png")

			fig2 = plt.figure()
			plt.scatter(test_outs,original_outs)
			plt.xlabel("Predicted Values")
			plt.ylabel("Actual Values")
			plt.legend()
			plt.savefig(self.name+"_r_plot.png")


			fig3 = plt.figure()
			plt.scatter(range(0,len(self.y_test)),indi_errors)
			plt.xlabel("Inputs")
			plt.ylabel("Errors")
			plt.legend()
			plt.savefig(self.name+"_individual_error_plot.png")

	def create_graphs(self):
		#Section to initialise the grapher for plotting the error
		self.error_grapher = grapher("Epochs","Error","Validation and Training Error over Epochs")
		self.error_grapher.add_line("Training Error")
		self.error_grapher.add_line("Validation Error")
		#Section ends

	def train(self):
		if self.show_graph == True:
			self.create_graphs()
		lrate = 0.05
		prev_error = 0
		epi = 0.01
		for epc in range(self.nb_epc):
			error = 0
			for i in range(len(self.x_train)):
				out = self.forward_prop(self.x_train[i])									
				error+= abs(sum(out - self.y_train[i]))
				deltas = backprop(self.net,out,self.y_train[i])
				new_w = sgd(self.net,deltas,lrate)
				self.update_ws(new_w)
			if abs(prev_error - error)<epi:
				lrate = lrate/2
			prev_error = error
			val_error = self.get_val_error()
			print "Error in epc",epc+1,"is",error," and Validation error is ",val_error
			if self.show_graph == True:
			    self.error_grapher.update_pyl_plot((epc+1,error/len(self.x_train)),'Training Error')
			    self.error_grapher.update_pyl_plot((epc+1,val_error/len(self.x_val)),'Validation Error')
		if self.graph_save == True:
			self.save_all(self.graph_names)

	def save_graphs(self,names):
		self.graph_save = True
		self.graph_names = names


	def save_all(self,names = None):
		if (names == None):
			names = []
			names[0] = self.name+"_error"
		self.error_grapher.save(names[0])		


	def save_output(self):
		xl_wrt(self.name+"_Data.xls",self.x_test,self.y_test,self.pred_test)
