from Basics.Initialisations import *
from Basics.ActivationFunction import *

from Layers.Convolution_Input import *
from Layers.CNN import *

import numpy as np


a = np.arange(81).reshape(9,9)

a = np.array([a])


in_layer = Cnv_Input("Test_Img.png")

hd1 = CNN((3,3),2,1,relu,in_layer,random_initialisation)
hd2 = CNN((3,3),2,1,relu,hd1,random_initialisation)

hd1.activate()
hd2.activate()
