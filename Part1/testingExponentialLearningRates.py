import logging
import numpy

logging.basicConfig()
logger = logging.getLogger('My Logger')
logger.setLevel(logging.INFO)

logger.info("write something here")

from mlp.layers import MLP, Linear, Sigmoid, Softmax, Relu, Tanh, Maxout# import required layer types
from mlp.optimisers import SGDOptimiser  # import the optimiser
from mlp.dataset import MNISTDataProvider  # import data provider
from mlp.costs import MSECost, CECost  # import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed, DropoutAnnealing,DropoutFixed,LearningRateExponential,LearningRateNewBob,LearningRateReciprocal
import matplotlib.pyplot as plt

rng = numpy.random.RandomState([2015, 10, 10])


maxBatches = -10
batchSize = 10
numberOfEpochs = 100
learningRate = 0.5
baseLearningRate = 10

cost = CECost()
model = MLP(cost=cost)
tanh1 = Tanh(idim=784, odim=500, rng=rng)
tanh2 = Tanh(idim=500, odim=100, rng=rng)
softmaxLayer = Softmax(idim=100, odim=10, rng=rng)
model.add_layer(tanh1)
model.add_layer(tanh2)
model.add_layer(softmaxLayer)

lrExponential = LearningRateExponential(learning_rate=learningRate, max_epochs=numberOfEpochs,batchSize=batchSize, baseLearningRate=baseLearningRate)

lrReciprocal = LearningRateReciprocal(learning_rate=learningRate, max_epochs=numberOfEpochs,batchSize=batchSize, baseLearningRate=baseLearningRate)

optimiser = SGDOptimiser(lr_scheduler=lrExponential)
list = []
for i in range(0,numberOfEpochs):
    list.append(lrExponential.get_next_rate())

plt.plot(list)
plt.show()

list = []
for i in range(0,numberOfEpochs):
    list.append(lrReciprocal.get_next_rate())

plt.plot(list)
plt.show()
