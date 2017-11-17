import numpy


from mlp.layers import MLP, Linear, Sigmoid, Softmax, Relu, Tanh, Maxout
from mlp.conv import ConvLinear
from mlp.optimisers import SGDOptimiser  # import the optimiser
from mlp.dataset import MNISTDataProvider  # import data provider
from mlp.costs import MSECost, CECost  # import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed
import matplotlib.pyplot as plt
from mlp.utils import test_conv_linear_fprop , test_conv_linear_bprop, test_conv_linear_pgrads

cost = CECost()
model = MLP(cost=cost)
rng = numpy.random.RandomState([2015, 10, 10])

convLayer = ConvLinear(num_inp_feat_maps=192,num_out_feat_maps=2,rng=rng)


model.add_layer(convLayer)

print test_conv_linear_fprop(convLayer)

print test_conv_linear_bprop(convLayer)

print test_conv_linear_pgrads(convLayer)