import matplotlib
matplotlib.use("Qt4Agg")
import pylab as pyl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



class grapher():
	"This object can be used to generate various graphs in the software"
	def __init__(self,x_axis= None,y_axis = None,title = None,sort = False,scatter = False):
		self.graph = None
		self.order = sort
		self.lines = {}
		self.nbl = 0
		self.x_label = x_axis
		self.y_label = y_axis
		self.title = title
		self.create_plt_graph()

	def create_plt_graph(self):
		fig = plt.figure()
		if (self.x_label!= None):
			plt.xlabel(self.x_label)
		if (self.y_label!=None):
			plt.ylabel(self.y_label)
		if (self.title != None):
			plt.title(self.title)
		plt.ion()
		self.graph = fig.add_subplot(111)
		plt.show(block = False)

	def add_line(self,name = None):
		if name == None:
			name = self.nbl
		new, = self.graph.plot([],[],label = name)
		self.lines[name] = new
		self.nbl+=1
		plt.legend()

	def set_line_data(self,lnb,x = None ,y = None):
		if x!= None:
			self.lines[lnb].set_xdata(x)
		if y!=None:
			self.lines[lnb].set_ydata(y)
		self.graph.relim()
		self.graph.autoscale_view()
		plt.draw()

	def update_pyl_plot(self,vals,lnb =0):	#Extremly slow need ot find a better way to do the updation
		if self.graph == None:
			self.create_plt_graph()
			self.add_line(lnb)
			

		x_vals = self.lines[lnb].get_xdata().tolist() # Better way to do this
		y_vals = self.lines[lnb].get_ydata().tolist() # Better way to do this
		x_vals.append(vals[0])
		y_vals.append(vals[1])
		if self.order == True:
			sort(x_vals)
			sort(y_vals)

		x_vals = np.array(x_vals)
		y_vals = np.array(y_vals)

		self.set_line_data(lnb,x_vals,y_vals)
		


		plt.pause(0.00001)

	def save(self,name):
		plt.savefig(name+".png")