import logging
import numpy

logging.basicConfig()
logger = logging.getLogger('My Logger')
logger.setLevel(logging.INFO)

logger.info("write something here")

from mlp.layers import MLP, Linear, Sigmoid, Softmax, Relu, Tanh, Maxout  # import required layer types
from mlp.optimisers import SGDOptimiser  # import the optimiser
from mlp.dataset import MNISTDataProvider  # import data provider
from mlp.costs import MSECost, CECost  # import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed, DropoutAnnealing, DropoutFixed, LearningRateExponential, LearningRateNewBob, \
    LearningRateReciprocal
import matplotlib.pyplot as plt
from mlp.utils import DumpData

rng = numpy.random.RandomState([2015, 10, 10])

maxBatches = -10
batchSize = 10
numberOfEpochs = 30
learningRate = 0.5
baseLearningRate = 0.1
l1Weights = 0.0001
l2Weights = 0.0001
dropoutBaseRate=0.4

logger.info('Initialising data providers...')
train_dp = MNISTDataProvider(dset='train_expanded', batch_size=batchSize, max_num_batches=maxBatches, randomize=True)
valid_dp = MNISTDataProvider(dset='valid', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)
test_dp = MNISTDataProvider(dset='eval', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)

cost = CECost()
model = MLP(cost=cost)
tanh1 = Sigmoid(idim=784, odim=500, rng=rng)
tanh2 = Sigmoid(idim=500, odim=1000, rng=rng)
tanh3 = Sigmoid(idim=1000, odim=300, rng=rng)
tanh4 = Sigmoid(idim=300, odim=100, rng=rng)
softmaxLayer = Softmax(idim=100, odim=10, rng=rng)
model.add_layer(tanh1)
model.add_layer(tanh2)
model.add_layer(tanh3)
model.add_layer(tanh4)
model.add_layer(softmaxLayer)

listAnnealing =[]
listAnnealing.append([dropoutBaseRate,dropoutBaseRate])
for i in range(1, numberOfEpochs-1):
    term = dropoutBaseRate * (1 + float(i) / (numberOfEpochs-1))
    listAnnealing.append([term,term])
listAnnealing.append([1,1])

learningRateExponential = LearningRateExponential(learning_rate=learningRate, max_epochs=numberOfEpochs,
                                                  batchSize=batchSize,
                                                  baseLearningRate=baseLearningRate)
dropoutScheduler = DropoutAnnealing(listAnnealing)
optimiser = SGDOptimiser(lr_scheduler=learningRateExponential, dp_scheduler=dropoutScheduler, l1_weight=l1Weights,
                         l2_weight=l2Weights)

depthOfModel = len(model.layers)

dummyModelCost = CECost()
newModel = MLP(cost=dummyModelCost)
parametersArray = []
activationsPresent = False
activation = None
isNoise = False

inputs = []

for x, t in train_dp:
    dummyList = []
    dummyList.append(x)
    dummyList.append(t)
    inputs.append(dummyList)

for i in range(0, depthOfModel):  # since we don't need softmax as our layer
    logger.info('Pre-train called for layer {0}'.format(i+1))
    if i != 0:
        activationsPresent = True

    learningRateExponential.reset()
    inputDimensionForDummyLayers = model.layers[i].idim
    outputDimensionsForDummyLayers = model.layers[i].odim
    nameOfLayer = model.layers[i].get_name()
    if nameOfLayer == 'sigmoid':
        dummyLayer1 = Sigmoid(idim=inputDimensionForDummyLayers, odim=outputDimensionsForDummyLayers, rng=rng)
    else:
        dummyLayer1 = Linear(idim=inputDimensionForDummyLayers, odim=outputDimensionsForDummyLayers, rng=rng)

    dummyLayer2 = Softmax(idim=outputDimensionsForDummyLayers, odim=10, rng=rng)
    newModel.add_layer(dummyLayer1)
    newModel.add_layer(dummyLayer2)
    optimiser.pretrain_descriminative(newModel, train_dp, activationsPresent, inputs=inputs, isNoisy=isNoise,
                                      activations=activation)
    activation = newModel.activations[i + 1]
    # get rid of the last activation coz that was the dummy activation
    activations = newModel.activations[:-1]
    newModel.activations = activations
    trainedParameters = newModel.layers[i].get_params()
    model.layers[i].set_params(trainedParameters)

    parametersArray.append(trainedParameters)
    newModel.set_layers(newModel.layers[:-1])


train_dp.reset()
valid_dp.reset()
learningRateExponential.reset()
optimiser = SGDOptimiser(lr_scheduler=learningRateExponential, dp_scheduler=None, l1_weight=l1Weights,
                         l2_weight=l2Weights)
logger.info('Training started...')
trainingStats, validStats = optimiser.train(model, train_dp, valid_dp)

logger.info('Testing the model on test set:')

cost, accuracy = optimiser.validate(model, test_dp)
logger.info('MNIST test set accuracy is %.2f %% (cost is %.3f)' % (accuracy * 100., cost))

dumpFileNameModel = './ModelDumps/Part1Task5_SOTA2.pkl'
dumpFileNameOptimiser = './OptimiserDumps/Part1Task5_SOTA2.pkl'
DumpData(dumpFileNameModel, model)
DumpData(dumpFileNameOptimiser, optimiser)

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

intermediateArrayTraining = 1.0 - accuracyStatsTraining
intermediateArrayValidation = 1.0 - accuracyStatsValidation

plt.plot(intermediateArrayTraining, label='Training')
plt.plot(intermediateArrayValidation, label="Validation")
plt.title("Cost of training and validation")
plt.show()

plt.plot(costStatsTraining, label='Training')
plt.plot(costStatsValidation, label="Validation")
plt.title("Cost of training and validation")
plt.show()
