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
from mlp.schedulers import LearningRateFixed, LearningRateExponential
import matplotlib.pyplot as plt

rng = numpy.random.RandomState([2015, 10, 10])


maxBatches = -10
batchSize = 10
numberOfEpochs = 20
learningRate = 0.5
baseLearningRate = 0.5

cost = CECost()
# model = MLP(cost=cost)
# sigmoidLayer1 = Sigmoid(idim=784, odim=1000, rng=rng)
# sigmoidLayer2 = Sigmoid(idim=1000, odim=800, rng=rng)
# sigmoidLayer3 = Sigmoid(idim=800, odim=500, rng=rng)
# sigmoidLayer4 = Sigmoid(idim=500, odim=100, rng=rng)
# softmaxLayer = Softmax(idim=100, odim=10, rng=rng)
# model.add_layer(sigmoidLayer1)
# model.add_layer(sigmoidLayer2)
# model.add_layer(sigmoidLayer3)
# model.add_layer(sigmoidLayer4)
# model.add_layer(softmaxLayer)

model = MLP(cost=cost)
sigmoidLayer1 = Sigmoid(idim=784, odim=1000, rng=rng)
sigmoidLayer2 = Sigmoid(idim=1000, odim=800, rng=rng)
softmaxLayer = Softmax(idim=800, odim=10, rng=rng)
model.add_layer(sigmoidLayer1)
model.add_layer(sigmoidLayer2)
model.add_layer(softmaxLayer)

lr_scheduler = LearningRateFixed(learning_rate=learningRate, max_epochs=numberOfEpochs)
lrExponential = LearningRateExponential(learning_rate=learningRate, max_epochs=numberOfEpochs,batchSize=batchSize, baseLearningRate=baseLearningRate)
optimiser = SGDOptimiser(lr_scheduler=lrExponential)

logger.info('Initialising data providers...')
train_dp = MNISTDataProvider(dset='train_expanded', batch_size=batchSize, max_num_batches=maxBatches, randomize=True)
valid_dp = MNISTDataProvider(dset='valid', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)

logger.info('Training started...')
trainingStats, validStats = optimiser.train(model, train_dp, valid_dp)

logger.info('Testing the model on test set:')
test_dp = MNISTDataProvider(dset='eval', batch_size=100, max_num_batches=-10, randomize=False)
cost, accuracy = optimiser.validate(model, test_dp)
logger.info('MNIST test set accuracy is %.2f %% (cost is %.3f)' % (accuracy * 100., cost))


numpyArrayTraining = numpy.array(trainingStats)
errorStats = numpyArrayTraining[:, 0]
finalError = errorStats[len(errorStats) - 1]
print('Final Error: ', finalError)
accuracyStats = numpyArrayTraining[:, 1]
finalAccuracy = accuracyStats[len(accuracyStats) - 1] * 100
print('Final Accuracy: ', finalAccuracy)

numpyArrayTraining = numpy.array(trainingStats)
numpyArrayValidation = numpy.array(validStats)
intermediateArrayTraining = 1. - numpyArrayTraining[:, 1]
intermediateArrayValidation = 1.0 - numpyArrayValidation[:,1]

plt.plot(intermediateArrayTraining,  label='LR = {0} \n Hidden Units = 800')
plt.show()
plt.plot(intermediateArrayValidation,  label='LR = {0} \n Hidden Units = 800')
plt.show()


