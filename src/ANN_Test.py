from Layers.ANN import *
from Basics.Initialisations import *
from Basics.ActivationFunction import *
from Layers.Input import *

inp = np.random.rand(10,10)
print inp

layer_in = Input_Layer(inp)
Layer = ANN(layer_in,10,relu,random_initialisation,random_initialisation)

print Layer.activate()

print Layer.get_out()