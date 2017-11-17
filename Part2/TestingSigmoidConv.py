import numpy
import logging
from mlp.layers import MLP, Linear, Sigmoid, Softmax, Relu, Tanh, Maxout  # import required layer types
from mlp.optimisers import SGDOptimiser  # import the optimiser
from mlp.dataset import MNISTDataProvider  # import data provider
from mlp.costs import MSECost, CECost  # import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed,LearningRateExponential,DropoutAnnealing,DropoutFixed
import matplotlib.pyplot as plt
from mlp.conv import ConvLinear, ConvMaxPool2D, ConvSigmoid
from mlp.utils import DumpData

logging.basicConfig()
logger = logging.getLogger('My Logger')
logger.setLevel(logging.INFO)

logger.info("write something here")

rng = numpy.random.RandomState([2015, 10, 10])

maxBatches = 10000
batchSize = 20
numberOfEpochs = 10
learningRateValue = 0.5
baseLearningRateValue = 15
l1Weights =0.0001
l2Weights = 0.0001

numberOfFeatureMaps = 1
linearLayerIdim = numberOfFeatureMaps * 12 * 12

cost = CECost()
model = MLP(cost=cost)
convLayer = ConvSigmoid(num_inp_feat_maps=1, num_out_feat_maps=numberOfFeatureMaps)
maxPoolLayer = ConvMaxPool2D(batch_size=batchSize, num_feat_maps=numberOfFeatureMaps, conv_shape=(24, 24))
linearLayer = Tanh(idim=linearLayerIdim, odim=linearLayerIdim)
softMax = Softmax(idim=linearLayerIdim, odim=10)

model.add_layer(convLayer)
model.add_layer(maxPoolLayer)
model.add_layer(linearLayer)
model.add_layer(softMax)

learningRateScheduler = LearningRateExponential(learning_rate=learningRateValue, max_epochs=numberOfEpochs,
                                                batchSize=batchSize,baseLearningRate=baseLearningRateValue)
optimiser = SGDOptimiser(lr_scheduler=learningRateScheduler,dp_scheduler=None,l1_weight=l1Weights,l2_weight=l2Weights)

train_dp = MNISTDataProvider(dset='train_expanded', batch_size=batchSize, max_num_batches=maxBatches, randomize=False,
                              conv_reshape=True)
valid_dp = MNISTDataProvider(dset='valid', batch_size=batchSize, max_num_batches=maxBatches, randomize=False,
                             conv_reshape=True)

a, v = optimiser.train(model=model, train_iterator=train_dp, valid_iterator=valid_dp)


dumpFileNameModel = '../ModelDumps/Part2Task6_ConvSigmoid.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part2Task6_ConvSigmoid.pkl'
DumpData(dumpFileNameModel, model)
DumpData(dumpFileNameOptimiser, optimiser)


