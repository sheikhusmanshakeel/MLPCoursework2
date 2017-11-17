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
numberOfEpochs = 20
learningRate = 0.5
baseLearningRate = 0.3

cost = CECost()
model = MLP(cost=cost)
tanh1 = Tanh(idim=784, odim=500, rng=rng)
tanh2 = Tanh(idim=500, odim=100, rng=rng)
softmaxLayer = Softmax(idim=100, odim=10, rng=rng)
model.add_layer(tanh1)
model.add_layer(tanh2)
model.add_layer(softmaxLayer)

lrExponential = LearningRateExponential(learning_rate=learningRate, max_epochs=numberOfEpochs,batchSize=batchSize, baseLearningRate=baseLearningRate)

optimiser = SGDOptimiser(lr_scheduler=lrExponential)

logger.info('Initialising data providers...')
train_dp = MNISTDataProvider(dset='train_expanded', batch_size=batchSize, max_num_batches=maxBatches, randomize=True)
valid_dp = MNISTDataProvider(dset='valid', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)
logger.info('Training started...')
trainingStatsLRE, validStatsLRE = optimiser.train(model, train_dp, valid_dp)
logger.info('Testing the model on test set:')
test_dp = MNISTDataProvider(dset='eval', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)
costLRE, accuracy = optimiser.validate(model, test_dp)
logger.info('MNIST test set accuracy is %.2f %% (cost is %.3f)' % (accuracy * 100., costLRE))


model = MLP(cost=cost)
tanh1 = Tanh(idim=784, odim=500, rng=rng)
tanh2 = Tanh(idim=500, odim=100, rng=rng)
softmaxLayer = Softmax(idim=100, odim=10, rng=rng)
model.add_layer(tanh1)
model.add_layer(tanh2)
model.add_layer(softmaxLayer)

lrReciprocal = LearningRateReciprocal(learning_rate=learningRate, max_epochs=numberOfEpochs,batchSize=batchSize, baseLearningRate=baseLearningRate)

optimiser = SGDOptimiser(lr_scheduler=lrReciprocal)

logger.info('Initialising data providers...')
train_dp.reset()
valid_dp.reset()
logger.info('Training started...')
trainingStatsLRR, validStatsLRR = optimiser.train(model, train_dp, valid_dp)
logger.info('Testing the model on test set:')
test_dp.reset()
costLRR, accuracy = optimiser.validate(model, test_dp)
logger.info('MNIST test set accuracy is %.2f %% (cost is %.3f)' % (accuracy * 100., costLRR))


model = MLP(cost=cost)
tanh1 = Tanh(idim=784, odim=500, rng=rng)
tanh2 = Tanh(idim=500, odim=100, rng=rng)
softmaxLayer = Softmax(idim=100, odim=10, rng=rng)
model.add_layer(tanh1)
model.add_layer(tanh2)
model.add_layer(softmaxLayer)

lrFixed = nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn

optimiser = SGDOptimiser(lr_scheduler=lrFixed)

logger.info('Initialising data providers...')
train_dp.reset()
valid_dp.reset()
logger.info('Training started...')
trainingStatsLRF, validStatsLRF = optimiser.train(model, train_dp, valid_dp)
logger.info('Testing the model on test set:')
test_dp.reset()
costLRF, accuracy = optimiser.validate(model, test_dp)
logger.info('MNIST test set accuracy is %.2f %% (cost is %.3f)' % (accuracy * 100., costLRF))



naTrainingStatsLRE = numpy.array(trainingStatsLRE)
naValidationStatsLRE = numpy.array(validStatsLRE)

errorStatsTrainingLRE = naTrainingStatsLRE[:, 0]
accuracyStatsTrainingLRE = naTrainingStatsLRE[:, 1]
finalErrorTrainingLRE = errorStatsTrainingLRE[len(errorStatsTrainingLRE) - 1]
finalAccuracyLRE = accuracyStatsTrainingLRE[len(accuracyStatsTrainingLRE) - 1] * 100
print('Final Error (Training) - Learning Rate Exponential ', finalErrorTrainingLRE)
print('Final Accuracy (Training) - Learning Rate Exponential ', finalAccuracyLRE)

errorStatsValidationLRE = naValidationStatsLRE[:, 0]
accuracyStatsValidationLRE = naValidationStatsLRE[:, 1]
finalErrorValidationLRE = errorStatsValidationLRE[len(errorStatsValidationLRE) - 1]
finalAccuracyValidation = accuracyStatsValidationLRE[len(accuracyStatsValidationLRE) - 1] * 100
print('Final Error (Validation) - Learning Rate Exponential', finalErrorValidationLRE)
print('Final Accuracy (Validation) - Learning Rate Exponential', finalAccuracyValidation)


naTrainingStatsLRR = numpy.array(trainingStatsLRR)
naValidationStatsLRR = numpy.array(validStatsLRR)

errorStatsTrainingLRR = naTrainingStatsLRR[:, 0]
accuracyStatsTrainingLRR = naTrainingStatsLRR[:, 1]
finalErrorTrainingLRR = errorStatsTrainingLRR[len(errorStatsTrainingLRR) - 1]
finalAccuracyLRR = accuracyStatsTrainingLRR[len(accuracyStatsTrainingLRR) - 1] * 100
print('Final Error (Training) - Learning Rate Reciprocal ', finalErrorTrainingLRR)
print('Final Accuracy (Training) - Learning Rate Reciprocal ', finalAccuracyLRR)

errorStatsValidationLRR = naValidationStatsLRR[:, 0]
accuracyStatsValidationLRR = naValidationStatsLRR[:, 1]
finalErrorValidationLRR = errorStatsValidationLRR[len(errorStatsValidationLRR) - 1]
finalAccuracyValidation = accuracyStatsValidationLRR[len(accuracyStatsValidationLRR) - 1] * 100
print('Final Error (Validation) - Learning Rate Exponential', finalErrorValidationLRR)
print('Final Accuracy (Validation) - Learning Rate Exponential', finalAccuracyValidation)

naTrainingStatsLRF = numpy.array(trainingStatsLRF)
naValidationStatsLRF = numpy.array(validStatsLRF)

errorStatsTrainingLRF = naTrainingStatsLRF[:, 0]
accuracyStatsTrainingLRF = naTrainingStatsLRF[:, 1]
finalErrorTrainingLRF = errorStatsTrainingLRF[len(errorStatsTrainingLRF) - 1]
finalAccuracyLRF = accuracyStatsTrainingLRF[len(accuracyStatsTrainingLRF) - 1] * 100
print('Final Error (Training) - Learning Rate Fixed ', finalErrorTrainingLRF)
print('Final Accuracy (Training) - Learning Rate Fixed ', finalAccuracyLRF)

errorStatsValidationLRF = naValidationStatsLRF[:, 0]
accuracyStatsValidationLRF = naValidationStatsLRF[:, 1]
finalErrorValidationLRF = errorStatsValidationLRF[len(errorStatsValidationLRF) - 1]
finalAccuracyValidation = accuracyStatsValidationLRF[len(accuracyStatsValidationLRF) - 1] * 100
print('Final Error (Validation) - Learning Rate Fixed', finalErrorValidationLRF)
print('Final Accuracy (Validation) - Learning Rate Fixed', finalAccuracyValidation)



costLRRTraining = 1.0 - accuracyStatsTrainingLRR
costLRETraining = 1.0 - accuracyStatsTrainingLRE
costLRFTraining = 1.0 - accuracyStatsTrainingLRF

plt.plot(costLRRTraining, label='Reciprocal')
plt.plot(costLRETraining, label="Exponential")
plt.plot(costLRFTraining, label="Fixed")
plt.title("Cost of Exponential, Reciprocal and Fixed Learning Rates - Training")
plt.legend()
plt.show()

costLRRValidation = 1.0 - accuracyStatsValidationLRR
costLREValidation = 1.0 - accuracyStatsValidationLRE
costLRFValidation = 1.0 - accuracyStatsValidationLRF

plt.plot(costLRRValidation, label='Reciprocal')
plt.plot(costLREValidation, label="Exponential")
plt.plot(costLRFValidation, label="Fixed")
plt.legend()
plt.title("Cost of Exponential, Reciprocal and Fixed Learning Rates - Validation")
plt.show()


plt.plot(costLRRValidation, label='Reciprocal - Validation')
plt.plot(costLREValidation, label="Exponential - Validation")
plt.plot(costLRFValidation, label="Fixed - Validation")
plt.plot(costLRRTraining, label='Reciprocal - Training')
plt.plot(costLRETraining, label="Exponential - Training")
plt.plot(costLRFTraining, label="Fixed - Training")
plt.title("Cost of Exponential, Reciprocal and Fixed Learning Rates")
plt.legend()
plt.show()


plt.plot(errorStatsTrainingLRR, label='LRR')
plt.plot(errorStatsTrainingLRE, label="LRE")
plt.plot(errorStatsTrainingLRF, label="LRF")
plt.title("Error of Exponential, Reciprocal and Fixed Learning Rates - Training")
plt.legend()
plt.show()

plt.plot(errorStatsValidationLRR, label='LRR')
plt.plot(errorStatsValidationLRE, label="LRE")
plt.plot(errorStatsValidationLRF, label="LRF")
plt.title("Error of Exponential, Reciprocal and Fixed Learning Rates - Validation")
plt.legend()
plt.show()




