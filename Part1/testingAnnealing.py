import cPickle
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy
import os
import scipy.ndimage as scipyImage
from matplotlib import pyplot

logging.basicConfig()
logger = logging.getLogger('My Logger')
logger.setLevel(logging.INFO)

logger.info("write something here")

from mlp.layers import MLP, Linear, Sigmoid, Softmax, Relu, Tanh, Maxout  # import required layer types
from mlp.optimisers import SGDOptimiser  # import the optimiser
from mlp.dataset import MNISTDataProvider  # import data provider
from mlp.costs import MSECost, CECost  # import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed, DropoutAnnealing, LearningRateExponential, LearningRateReciprocal, \
    LearningRateNewBob, DropoutFixed
from mlp.utils import DumpData


rng = numpy.random.RandomState([2015, 10, 10])

maxBatches = 10000
batchSize = 10
numberOfEpochs = 15
learningRateValue = 0.5
dropoutBaseRate = 0.5
l1_weight = 0.0002
l2_weight = 0.0003
baseLearningRateValue = 0.5
maxOutK = 10

# Dropout Annealing should make at least an array of size 10 so that each epoch can have its own droupout probability
# It should start with a base rate and gradually move to 1

train_dp = MNISTDataProvider(dset='train', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)
valid_dp = MNISTDataProvider(dset='valid', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)
test_dp = MNISTDataProvider(dset='eval', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)

list = []
# a = 1.00000001
# for i in range(2, numberOfEpochs + 2):
#     term = 1 / float((numpy.power(a, i) - 1) / (a - 1))
#     list.append([1 - term, 1 - term])


list =[]
list.append([dropoutBaseRate,dropoutBaseRate])
for i in xrange(1, numberOfEpochs-1):
    term = dropoutBaseRate * (1 + float(i) / (numberOfEpochs-1))
    list.append([term,term])
list.append([1,1])


learningRate = LearningRateFixed(learning_rate=learningRateValue,max_epochs=numberOfEpochs)
dropoutScheduler = DropoutAnnealing(list)
optimiser = SGDOptimiser(lr_scheduler=learningRate, dp_scheduler=dropoutScheduler)
cost = CECost()
model = MLP(cost=cost)
tanh1 = Tanh(idim=784, odim=500, rng=rng)
# relu = Relu(idim=500, odim=1000, rng=rng)
# tanh2 = Sigmoid(idim=1000, odim=500, rng=rng)
# maxOut = Maxout(idim=500, odim=100, k=maxOutK, rng=rng)
softmaxLayer = Softmax(idim=500, odim=10, rng=rng)
model.add_layer(tanh1)
# model.add_layer(relu)
# model.add_layer(tanh2)
# model.add_layer(maxOut)
model.add_layer(softmaxLayer)

logger.info('Training started...')
annealingTrainingStats, annealingValidationStats = optimiser.train(model, train_dp, valid_dp)

logger.info('Testing the model on test set:')
annealingTestingCost, annealingTestingAccuracy = optimiser.validate(model, test_dp)
logger.info('MNIST test set accuracy is %.2f %%, cost (%s) is %.3f' % (
    annealingTestingAccuracy * 100., cost.get_name(), annealingTestingCost))





dumpFileNameModel = '../ModelDumps/Part1Task2_Annealing.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part1Task2_Annealing.pkl'
DumpData(dumpFileNameModel, model)
DumpData(dumpFileNameOptimiser, optimiser)


