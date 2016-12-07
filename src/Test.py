import numpy as np
import random

from Basics.Initialisations import *
from Basics.ActivationFunction import *
from Layers.ANN import *
from Layers.Input import *
from Trainer.ANN_TrainerC import *
from Trainer.DataManager import *




name = "mpp_dataset_v1_13-inputs"
epcs = 800

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

DM = DataManager(name+".xls",13,1,normalization = "Min Max")

x_data,y_data = DM.read_file() # Index 0 is train data , 1 is validation data , 2 is test data



in_layer = Input(x_data[0][0])
hidden_layer = ANN(40,sigmoid,in_layer,random_initialisation)
hidden_layer_2 = ANN(40,sigmoid,hidden_layer,random_initialisation)

out_layer = ANN(1,tanh,hidden_layer_2,random_initialisation)

names = [name+"_error"]
net = {0:in_layer,1:hidden_layer,2:hidden_layer_2,3:out_layer}

trainer = ANN_Trainer(net,epcs,name)
trainer.set_train_data(x_data[0],y_data[0])
trainer.set_val_data(x_data[1],y_data[1])	
trainer.set_test_data(x_data[2],y_data[2])
trainer.show_graphs()
trainer.save_graphs(names)
trainer.train()	 
trainer.get_test_error()

trainer.save_output()
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