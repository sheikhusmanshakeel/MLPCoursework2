import numpy
import logging
from mlp.layers import MLP, Linear, Sigmoid, Softmax, Relu, Tanh, Maxout  # import required layer types
from mlp.optimisers import SGDOptimiser  # import the optimiser
from mlp.dataset import MNISTDataProvider  # import data provider
from mlp.costs import MSECost, CECost  # import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed, LearningRateExponential, DropoutAnnealing, DropoutFixed
import matplotlib.pyplot as plt
from mlp.conv import ConvLinear, ConvMaxPool2D, ConvSigmoid, ConvRelu
from mlp.utils import DumpData

logging.basicConfig()
logger = logging.getLogger('My Logger')
logger.setLevel(logging.INFO)

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
logger.info("write something here")

rng = numpy.random.RandomState([2015, 10, 10])

maxBatches = 1000
batchSize = 50
numberOfEpochs = 30
learningRateValue = 0.5
baseLearningRateValue = 15
l1Weights = 0.001
l2Weights = 0.001

numberOfFeatureMaps = 1
numberOfFeatureMapsSecondLayer = 5
linearLayerIdim = numberOfFeatureMapsSecondLayer * 4 * 4

cost = CECost()
model = MLP(cost=cost)
convLayer = ConvSigmoid(num_inp_feat_maps=1, num_out_feat_maps=numberOfFeatureMaps)
maxPoolLayer = ConvMaxPool2D(batch_size=batchSize, num_feat_maps=numberOfFeatureMaps, conv_shape=(24, 24))
convLayer2 = ConvSigmoid(num_inp_feat_maps=numberOfFeatureMaps, num_out_feat_maps=numberOfFeatureMapsSecondLayer)
maxPoolLayer2 = ConvMaxPool2D(batch_size=batchSize, num_feat_maps=numberOfFeatureMapsSecondLayer, conv_shape=(8, 8))
tanh = Tanh(idim=linearLayerIdim, odim=linearLayerIdim)
softMax = Softmax(idim=linearLayerIdim, odim=10)

model.add_layer(convLayer)
model.add_layer(maxPoolLayer)
model.add_layer(convLayer2)
model.add_layer(maxPoolLayer2)
model.add_layer(tanh)
model.add_layer(softMax)

# learningRateScheduler = LearningRateExponential(learning_rate=learningRateValue, max_epochs=numberOfEpochs,
#                                                 batchSize=batchSize,baseLearningRate=baseLearningRateValue)
learningRateScheduler = LearningRateFixed(learning_rate=learningRateValue, max_epochs=numberOfEpochs)
optimiser = SGDOptimiser(lr_scheduler=learningRateScheduler, dp_scheduler=None, l1_weight=l1Weights,
                         l2_weight=l2Weights)

train_dp = MNISTDataProvider(dset='train_expanded', batch_size=batchSize, max_num_batches=maxBatches, randomize=False,
                             conv_reshape=True)
valid_dp = MNISTDataProvider(dset='valid', batch_size=batchSize, max_num_batches=maxBatches, randomize=False,
                             conv_reshape=True)

a, v = optimiser.train(model=model, train_iterator=train_dp, valid_iterator=valid_dp)

dumpFileNameModel = '../ModelDumps/Part2Task6_ConvTwoLayer.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part2Task6_ConvTwoLayer.pkl'
DumpData(dumpFileNameModel, model)
DumpData(dumpFileNameOptimiser, optimiser)
