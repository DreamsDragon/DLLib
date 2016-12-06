import numpy as np
import random

from Basics.Initialisations import *
from Basics.ActivationFunction import *
from Layers.ANN import *
from Layers.Input import *
from Trainer.ANN_Trainer import *
from PreProcess.Xcl_read import *
from PreProcess.Normalisation import *
from PreProcess.Data_Handling import *


name = "mpp_dataset_v1_13-inputs"

'''
f = open(name+".txt",'r')

x_train = []
y_train = []

for line in f:
	x,y = line.split(" ")
	x_train.append([[float(x)]])
	y_train.append([[float(y)]])

x = np.array(x_train)
y = np.array(y_train)
'''
x_train,y_train  = xl_cnv(name+".xls")
y = y_train
x = x_train
#y_train = (2*((y_train- min(y_train)))/(max(y_train)-min(y_train)))-1
'''
x = []
y = []

for i in range(1000):
	a = random.uniform(0,10)
	b = random.uniform(0,10)
	c = random.uniform(0,10)

	x.append([[a],[b],[c]])
	y.append([[(a**2)+(b**2)-(c**2)]])
x = np.array(x)
y = np.array(y)
'''

x = min_max_normalisation(x)
y = min_max_normalisation(y)


x_train,x_val,x_test = break_data(x,(60,30,10))
y_train,y_val,y_test = break_data(y,(60,30,10))

#x_train,y_train  = xl_cnv("Function Data.xlsx")
#print x_train,y_train

#y_train = (2*((y_train- min(y_train)))/(max(y_train)-min(y_train)))-1

in_layer = Input(x_train[0])
hidden_layer = ANN(20,sigmoid,in_layer,random_initialisation)
hidden_layer_2 = ANN(20,sigmoid,hidden_layer,random_initialisation)
hidden_layer_3 = ANN(20,sigmoid,hidden_layer_2,random_initialisation)

out_layer = ANN(1,tanh,hidden_layer_3,random_initialisation)

names = [name+"_error"]
net = {0:in_layer,1:hidden_layer,2:hidden_layer_2,3:hidden_layer_3,4:out_layer}

trainer = ANN_Trainer(net,300,name)
trainer.set_train_data(x_train,y_train)
trainer.set_val_data(x_val,y_val)	
trainer.set_test_data(x_test,y_test)
trainer.show_graphs()
trainer.save_graphs(names)
trainer.train()	 
trainer.get_test_error()

'''

test_vals = []
test_ins =[]
er = 0
ac = 0
for n in range(len(x_test)):
	test_ins.append(y[n])
	test_vals.append(trainer.forward_prop(x[n]))
	er+= abs(y[n] - test_vals[n])
	if (abs(y[n] - test_vals[n])<0.05):
		ac+=1
	#print "Actual Value is ",y[n]," Predicted Value is ",out_layer.get_out()[0]
print "Test Error :",er,"Accuracy is ",ac

#test_ins = [i for i in range(len(x_test))]
plt.figure()
plt.scatter(test_ins,test_vals)
plt.savefig("Line Class_data.png")'''