from Basics.Initialisations import *
from Basics.ActivationFunction import *

from Layers.Convolution_Input import *
from Layers.CNN import *
from Layers.Pooling import *
from Layers.ANN import *

import numpy as np


a = np.arange(81).reshape(9,9)

a = np.array([a])


in_layer = Cnv_Input("Test_Img.png")

hd1 = CNN((3,3),2,1,relu,in_layer,random_initialisation)
p1 = Pooling((3,3),2,"Max",hd1)
hd2 = CNN((3,3),2,1,relu,p1,random_initialisation)
fc1 = ANN(1,tanh,hd2,random_initialisation)



hd1.activate()
p1.activate()
hd2.activate()
fc1.activate()

hd1.activate()
p1.activate()
hd2.activate()
fc1.activate()

print hd1.get_out()[0].shape
print hd2.get_out()[0].shape
print p1.get_out()[0].shape
print fc1.get_out()[0].shape