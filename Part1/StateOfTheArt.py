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
numberOfEpochs = 15
learningRate = 0.5
baseLearningRate = 0.1

cost = CECost()
model = MLP(cost=cost)
tanh1 = Tanh(idim=784, odim=2500, rng=rng)
tanh2 = Tanh(idim=2500, odim=2000, rng=rng)
tanh3 = Tanh(idim=2000, odim=1500, rng=rng)
tanh4 = Tanh(idim=1500, odim=1000, rng=rng)
tanh5 = Tanh(idim=1000, odim=500, rng=rng)
softmaxLayer = Softmax(idim=500, odim=10, rng=rng)
model.add_layer(tanh1)
model.add_layer(tanh2)
model.add_layer(tanh3)
model.add_layer(tanh4)
model.add_layer(tanh5)
model.add_layer(softmaxLayer)

list = []
a = 1.00000001
for i in range(2, numberOfEpochs+2):
    term = 1 / float((numpy.power(a,i) - 1) / (a - 1))
    list.append([1 - term, 1-term])


lr_scheduler = LearningRateExponential(learning_rate=learningRate, max_epochs=numberOfEpochs,batchSize=batchSize, baseLearningRate=baseLearningRate)
dropoutScheduler = DropoutAnnealing(list)
optimiser = SGDOptimiser(lr_scheduler=lr_scheduler,dp_scheduler=dropoutScheduler)

logger.info('Initialising data providers...')
train_dp = MNISTDataProvider(dset='train_expanded', batch_size=batchSize, max_num_batches=maxBatches, randomize=True)
valid_dp = MNISTDataProvider(dset='valid', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)

logger.info('Training started...')
trainingStats, validStats = optimiser.train(model, train_dp, valid_dp)

logger.info('Testing the model on test set:')
test_dp = MNISTDataProvider(dset='eval', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)
cost, accuracy = optimiser.validate(model, test_dp)
logger.info('MNIST test set accuracy is %.2f %% (cost is %.3f)' % (accuracy * 100., cost))


naTrainingStats = numpy.array(trainingStats)
naValidationStats = numpy.array(validStats)

costStatsTraining = naTrainingStats[:, 0]
accuracyStatsTraining = naTrainingStats[:, 1]
finalCostTraining = costStatsTraining[len(costStatsTraining) - 1]
finalAccuracy = accuracyStatsTraining[len(accuracyStatsTraining) - 1] * 100
print('Final Cost (Training) ', finalCostTraining)
print('Final Accuracy (Training) ', finalAccuracy)

costStatsValidation = naValidationStats[:, 0]
accuracyStatsValidation = naValidationStats[:, 1]
finalCostValidation = costStatsValidation[len(costStatsValidation) - 1]
finalAccuracyValidation = accuracyStatsValidation[len(accuracyStatsValidation) - 1] * 100
print('Final Cost (Validation) ', finalCostValidation)
print('Final Accuracy (Validation) ', finalAccuracyValidation)

errorTrainingSOTA1 = 1.0 - accuracyStatsTraining
errorValidationSOTA1 = 1.0 - accuracyStatsValidation

plt.plot(errorTrainingSOTA1, label='Training')
plt.plot(errorValidationSOTA1, label="Validation")
plt.title("Error of training and validation")
plt.show()

plt.plot(costStatsTraining, label='Training')
plt.plot(costStatsValidation, label="Validation")
plt.title("Cost of training and validation")
plt.legend()
plt.show()
