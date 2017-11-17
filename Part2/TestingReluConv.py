import numpy
import logging
from mlp.layers import MLP, Linear, Sigmoid, Softmax, Relu, Tanh, Maxout  # import required layer types
from mlp.optimisers import SGDOptimiser  # import the optimiser
from mlp.dataset import MNISTDataProvider  # import data provider
from mlp.costs import MSECost, CECost  # import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed
import matplotlib.pyplot as plt
from mlp.conv import ConvLinear, ConvMaxPool2D, ConvSigmoid, ConvRelu
from mlp.utils import DumpData

logging.basicConfig()
logger = logging.getLogger('My Logger')
logger.setLevel(logging.INFO)

logger.info("write something here")

rng = numpy.random.RandomState([2015, 10, 10])

maxBatches = 1000
batchSize = 10
numberOfEpochs = 10
learningRate = 0.2

numberOfFeatureMaps = 1
linearLayerIdim = numberOfFeatureMaps * 12 * 12

cost = CECost()
model = MLP(cost=cost)
convLayer = ConvRelu(num_inp_feat_maps=1, num_out_feat_maps=numberOfFeatureMaps)
maxPoolLayer = ConvMaxPool2D(batch_size=batchSize, num_feat_maps=numberOfFeatureMaps, conv_shape=(24, 24))
linearLayer = Sigmoid(idim=linearLayerIdim, odim=linearLayerIdim)
softMax = Softmax(idim=linearLayerIdim, odim=10)

model.add_layer(convLayer)
model.add_layer(maxPoolLayer)
model.add_layer(linearLayer)
model.add_layer(softMax)

lr_scheduler = LearningRateFixed(learning_rate=learningRate, max_epochs=numberOfEpochs)
optimiser = SGDOptimiser(lr_scheduler=lr_scheduler)

train_dp = MNISTDataProvider(dset='train', batch_size=batchSize, max_num_batches=maxBatches, randomize=False,
                             conv_reshape=True)
valid_dp = MNISTDataProvider(dset='valid', batch_size=batchSize, max_num_batches=maxBatches, randomize=False,
                             conv_reshape=True)

a, v = optimiser.train(model=model, train_iterator=train_dp, valid_iterator=valid_dp)

dumpFileNameModel = '../ModelDumps/Part2Task6_ConvRelu.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part2Task6_ConvRelu.pkl'
DumpData(dumpFileNameModel, model)
DumpData(dumpFileNameOptimiser, optimiser)