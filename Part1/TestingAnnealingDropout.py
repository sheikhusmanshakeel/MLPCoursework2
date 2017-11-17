import logging
import matplotlib
import numpy
import scipy.ndimage as scipyImage
from matplotlib import pyplot
from mlp.utils import  DumpData

logging.basicConfig()
logger = logging.getLogger('My Logger')
logger.setLevel(logging.INFO)
#
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

logger.info("write something here")

from mlp.layers import MLP, Linear, Sigmoid, Softmax, Relu, Tanh, Maxout  # import required layer types
from mlp.optimisers import SGDOptimiser  # import the optimiser
from mlp.dataset import MNISTDataProvider  # import data provider
from mlp.costs import MSECost, CECost  # import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed, DropoutAnnealing, LearningRateExponential, LearningRateReciprocal, \
    LearningRateNewBob, DropoutFixed
import matplotlib.pyplot as plt

rng = numpy.random.RandomState([2015, 10, 10])

maxBatches = -10
batchSize = 100
numberOfEpochs = 5
learningRateValue = 0.5
dropoutBaseRate = 0.5
l1_weight = 0.2
l2_weight = 0.3
baseLearningRateValue = 0.3

# Dropout Annealing should make at least an array of size 10 so that each epoch can have its own droupout probability
# It should start with a base rate and gradually move to 1

train_dp = MNISTDataProvider(dset='train', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)
valid_dp = MNISTDataProvider(dset='valid', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)
test_dp = MNISTDataProvider(dset='eval', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)

list =[]
list.append([dropoutBaseRate,dropoutBaseRate])
for i in xrange(1, numberOfEpochs-1):
    term = dropoutBaseRate * (1 + float(i) / (numberOfEpochs-1))
    list.append([term,term])
list.append([1,1])


plt.plot(list)
plt.title("Dropout Annealing Values")
plt.show()

learningRate = LearningRateFixed(learning_rate=learningRateValue,max_epochs=numberOfEpochs)
dropoutScheduler = DropoutAnnealing(list)
optimiserAnnealing = SGDOptimiser(lr_scheduler=learningRate, dp_scheduler=dropoutScheduler)
cost = CECost()
model = MLP(cost=cost)
sigmoid1 = Sigmoid(idim=784, odim=300, rng=rng)
sigmoid2 = Sigmoid(idim=300, odim=100, rng=rng)
softmaxLayer = Softmax(idim=100, odim=10, rng=rng)
model.add_layer(sigmoid1)
model.add_layer(sigmoid2)
model.add_layer(softmaxLayer)

logger.info('Training started...')
annealingTrainingStats, annealingValidationStats = optimiserAnnealing.train(model, train_dp, valid_dp)

logger.info('Testing the model on test set:')
annealingTestingCost, annealingTestingAccuracy = optimiserAnnealing.validate(model, test_dp)
logger.info('MNIST test set accuracy is %.2f %%, cost (%s) is %.3f' % (
    annealingTestingAccuracy * 100., cost.get_name(), annealingTestingCost))


dumpFileNameModel = '../ModelDumps/Part1Task2_Annealing.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part1Task2_Annealing.pkl'
DumpData(dumpFileNameModel, model)
DumpData(dumpFileNameOptimiser, optimiserAnnealing)






train_dp.reset()
test_dp.reset()
valid_dp.reset()
learningRate.reset()

dropoutScheduler = DropoutFixed(dropoutBaseRate, dropoutBaseRate)
optimiserFixed = SGDOptimiser(lr_scheduler=learningRate, dp_scheduler=dropoutScheduler)
cost = CECost()
model = MLP(cost=cost)
sigmoid1 = Sigmoid(idim=784, odim=300, rng=rng)
sigmoid2 = Sigmoid(idim=300, odim=100, rng=rng)
softmaxLayer = Softmax(idim=100, odim=10, rng=rng)
model.add_layer(sigmoid1)
model.add_layer(sigmoid2)
model.add_layer(softmaxLayer)

logger.info('Training started...')
fixedDropOutTrainingStats, fixedDropOutValidationStats = optimiserFixed.train(model, train_dp, valid_dp)

logger.info('Testing the model on test set:')
fixedDropOutTestingCost, fixedDropOutTestingAccuracy = optimiserFixed.validate(model, test_dp)
logger.info('MNIST test set accuracy is %.2f %%, cost (%s) is %.3f' % (
fixedDropOutTestingAccuracy * 100., cost.get_name(), fixedDropOutTestingCost))

dumpFileNameModel = '../ModelDumps/Part1Task2_Fixed.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part1Task2_Fixed.pkl'
DumpData(dumpFileNameModel, model)
DumpData(dumpFileNameOptimiser, optimiserFixed)







train_dp.reset()
test_dp.reset()
valid_dp.reset()
learningRate.reset()


optimiserFixed = SGDOptimiser(lr_scheduler=learningRate,dp_scheduler=None, l1_weight=l1_weight,l2_weight=l2_weight)
cost = CECost()
model = MLP(cost=cost)
sigmoid1 = Sigmoid(idim=784, odim=300, rng=rng)
sigmoid2 = Sigmoid(idim=300, odim=100, rng=rng)
softmaxLayer = Softmax(idim=100, odim=10, rng=rng)
model.add_layer(sigmoid1)
model.add_layer(sigmoid2)
model.add_layer(softmaxLayer)

logger.info('Training started...')
lRegTrainingStats, lRegValidationStats = optimiserFixed.train(model, train_dp, valid_dp)

logger.info('Testing the model on test set:')
lRegTestingCost, lRegTestingAccuracy = optimiserFixed.validate(model, test_dp)
logger.info('MNIST test set accuracy is %.2f %%, cost (%s) is %.3f' % (
lRegTestingAccuracy * 100., cost.get_name(), lRegTestingCost))


dumpFileNameModel = '../ModelDumps/Part1Task2_L1L2.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part1Task2_L1L2.pkl'
DumpData(dumpFileNameModel, model)
DumpData(dumpFileNameOptimiser, optimiserFixed)




import cPickle

model = None
optimiser = None
dumpFileNameModel = '../ModelDumps/Part1Task2_Annealing.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part1Task2_Annealing.pkl'

with open(dumpFileNameModel, 'r') as f:
    model = cPickle.load(f)

with open(dumpFileNameOptimiser, 'r') as f:
    optimiser = cPickle.load(f)



naTrainingStatsAnnealing = numpy.array(optimiser.GetTrainingStats())
naValidationStatsAnnealing = numpy.array(optimiser.GetValidationStats())

costStatsTrainingAnnealing = naTrainingStatsAnnealing[:, 0]
accuracyStatsTrainingAnnealing = naTrainingStatsAnnealing[:, 1]
finalCostTrainingAnnealing = costStatsTrainingAnnealing[len(costStatsTrainingAnnealing) - 1]
finalAccuracyTrainingAnnealing = accuracyStatsTrainingAnnealing[len(accuracyStatsTrainingAnnealing) - 1] * 100
print('Final Cost (Training) Annealed Dropout ', finalCostTrainingAnnealing)
print('Final Accuracy (Training) Annealed Dropout ', finalAccuracyTrainingAnnealing)

costStatsValidationAnnealing = naValidationStatsAnnealing[:, 0]
accuracyStatsValidationAnnealing = naValidationStatsAnnealing[:, 1]
finalCostValidationAnnealing = costStatsValidationAnnealing[len(costStatsValidationAnnealing) - 1]
finalAccuracyValidationAnnealing = accuracyStatsValidationAnnealing[
                                             len(accuracyStatsValidationAnnealing) - 1] * 100
print('Final Cost (Validation) Annealed Dropout ', finalCostValidationAnnealing)
print('Final Accuracy (Validation) Annealed Dropout ', finalAccuracyValidationAnnealing)



model = None
optimiser = None
dumpFileNameModel = '../ModelDumps/Part1Task2_Fixed.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part1Task2_Fixed.pkl'

with open(dumpFileNameModel, 'r') as f:
    model = cPickle.load(f)

with open(dumpFileNameOptimiser, 'r') as f:
    optimiser = cPickle.load(f)

naTrainingStatsFixedDropOut = numpy.array(optimiser.GetTrainingStats())
naValidationStatsFixedDropOut = numpy.array(optimiser.GetValidationStats())

costStatsTrainingFixedDropOut = naTrainingStatsFixedDropOut[:, 0]
accuracyStatsTrainingFixedDropOut = naTrainingStatsFixedDropOut[:, 1]
finalCostTrainingFixedDropOut = costStatsTrainingFixedDropOut[len(costStatsTrainingFixedDropOut) - 1]
finalAccuracyTrainingFixedDropOut = accuracyStatsTrainingFixedDropOut[len(accuracyStatsTrainingFixedDropOut) - 1] * 100
print('Final Cost (Training) Fixed Dropout ', finalCostTrainingFixedDropOut)
print('Final Accuracy (Training) Fixed Dropout ', finalAccuracyTrainingFixedDropOut)

costStatsValidationFixedDropOut = naValidationStatsFixedDropOut[:, 0]
accuracyStatsValidationFixedDropOut = naValidationStatsFixedDropOut[:, 1]
finalCostValidationFixedDropOut = costStatsValidationFixedDropOut[len(costStatsValidationFixedDropOut) - 1]
finalAccuracyValidationFixedDropOut = accuracyStatsValidationFixedDropOut[
                                             len(accuracyStatsValidationFixedDropOut) - 1] * 100
print('Final Cost (Validation) Fixed Dropout ', finalCostValidationFixedDropOut)
print('Final Accuracy (Validation) Fixed Dropout ', finalAccuracyValidationFixedDropOut)



model = None
optimiser = None
dumpFileNameModel = '../ModelDumps/Part1Task2_L1L2.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part1Task2_L1L2.pkl'

with open(dumpFileNameModel, 'r') as f:
    model = cPickle.load(f)

with open(dumpFileNameOptimiser, 'r') as f:
    optimiser = cPickle.load(f)

naTrainingStatsRegularized = numpy.array(optimiser.GetTrainingStats())
naValidationStatsRegularized = numpy.array(optimiser.GetValidationStats())

costStatsTrainingRegularized = naTrainingStatsRegularized[:, 0]
accuracyStatsTrainingRegularized = naTrainingStatsRegularized[:, 1]
finalCostTrainingRegularized = costStatsTrainingRegularized[len(costStatsTrainingRegularized) - 1]
finalAccuracyTrainingRegularized = accuracyStatsTrainingRegularized[len(accuracyStatsTrainingRegularized) - 1] * 100
print('Final Cost (Training) L1 & L2 Regularization ', finalCostTrainingRegularized)
print('Final Accuracy (Training) L1 & L2 Regularization ', finalAccuracyTrainingRegularized)

costStatsValidationRegularized = naValidationStatsRegularized[:, 0]
accuracyStatsValidationRegularized = naValidationStatsRegularized[:, 1]
finalCostValidationRegularized = costStatsValidationRegularized[len(costStatsValidationRegularized) - 1]
finalAccuracyValidationRegularized = accuracyStatsValidationRegularized[
                                             len(accuracyStatsValidationRegularized) - 1] * 100
print('Final Cost (Validation) L1 & L2 Regularization ', finalCostValidationRegularized)
print('Final Accuracy (Validation) L1 & L2 Regularization ', finalAccuracyValidationRegularized)



intermediateArrayTraining1 = 1.0 - accuracyStatsTrainingRegularized
intermediateArrayTraining2 = 1.0 - accuracyStatsTrainingAnnealing
intermediateArrayTraining3 = 1.0 - accuracyStatsTrainingFixedDropOut

plt.plot(intermediateArrayTraining1, label='L1 & L2 Regularization')
plt.plot(intermediateArrayTraining2, label="Annealed Dropout")
plt.plot(intermediateArrayTraining3, label="Fixed Dropout")
plt.title("Error of training")
plt.legend()
plt.show()

plt.plot(costStatsTrainingRegularized, label='L1 & L2 Regularization')
plt.plot(costStatsTrainingAnnealing, label="Annealed Dropout")
plt.plot(costStatsTrainingFixedDropOut, label="Fixed Dropout")
plt.title("Cost of training ")
plt.legend()
plt.show()

intermediateArrayTraining1 = 1.0 - accuracyStatsValidationRegularized
intermediateArrayTraining2 = 1.0 - accuracyStatsValidationAnnealing
intermediateArrayTraining3 = 1.0 - accuracyStatsValidationFixedDropOut

plt.plot(intermediateArrayTraining1, label='L1 & L2 Regularization')
plt.plot(intermediateArrayTraining2, label="Annealed Dropout")
plt.plot(intermediateArrayTraining3, label="Fixed Dropout")
plt.title("Error of validation")
plt.legend()
plt.show()

plt.plot(costStatsValidationRegularized, label='L1 & L2 Regularization')
plt.plot(costStatsValidationAnnealing, label="Annealed Dropout")
plt.plot(costStatsValidationFixedDropOut, label="Fixed Dropout")
plt.title("Cost of validation ")
plt.legend()
plt.show()




