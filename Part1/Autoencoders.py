import logging
import matplotlib
import numpy
import scipy.ndimage as scipyImage
from matplotlib import pyplot

logging.basicConfig()
logger = logging.getLogger('My Logger')
logger.setLevel(logging.INFO)

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

logger.info("write something here")

from mlp.layers import MLP, Linear, Sigmoid, Softmax, Relu, Tanh, Maxout  # import required layer types
from mlp.optimisers import SGDOptimiser  # import the optimiser
from mlp.dataset import MNISTDataProvider  # import data provider
from mlp.costs import MSECost, CECost  # import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed, LearningRateExponential, LearningRateNewBob, LearningRateReciprocal
import matplotlib.pyplot as plt
from mlp.utils import DumpData

rng = numpy.random.RandomState([2015, 10, 10])

maxBatches = 10
batchSize = 100
numberOfEpochs = 2
learningRateValue = 0.1


cost = CECost()
model = MLP(cost=cost)
sigmoidLayer1 = Sigmoid(idim=784, odim=300, rng=rng)
sigmoidLayer2 = Sigmoid(idim=300, odim=100, rng=rng)


model.add_layer(sigmoidLayer1)
model.add_layer(sigmoidLayer2)

learningRate = LearningRateFixed(learning_rate=learningRateValue,max_epochs=numberOfEpochs)
optimiser = SGDOptimiser(lr_scheduler=learningRate)

train_dp = MNISTDataProvider(dset='train', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)
valid_dp = MNISTDataProvider(dset='valid', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)

depthOfModel = len(model.layers)
dummyModelCost = MSECost()
newModel = MLP(cost=dummyModelCost)
parametersArray = []
activationsPresent = False
activation = None
isNoisy = False

inputs = []
for x, t in train_dp:
    inputs.append(x)

for i in range(0, depthOfModel):  # since we don't need softmax as our layer
    if i != 0:
        activationsPresent = True
    learningRate.reset()
    inputDimensionForDummyLayers = model.layers[i].idim
    outputDimensionsForDummyLayers = model.layers[i].odim
    dummyLayer1 = Sigmoid(idim=inputDimensionForDummyLayers, odim=outputDimensionsForDummyLayers, rng=rng)
    dummyLayer2 = Sigmoid(idim=outputDimensionsForDummyLayers, odim=inputDimensionForDummyLayers, rng=rng)
    newModel.add_layer(dummyLayer1)
    newModel.add_layer(dummyLayer2)
    optimiser.pretrain(newModel, train_dp, activationsPresent,  inputs = inputs, isNoisy=isNoisy, activations=activation)
    activation = newModel.activations[i+1]
    # get rid of the last activation coz that was the dummy activation
    activations = newModel.activations[:-1]
    newModel.activations = activations
    trainedParameters = newModel.layers[i].get_params()
    model.layers[i].set_params(trainedParameters)

    parametersArray.append(trainedParameters)
    newModel.set_layers(newModel.layers[:-1])

logger.info('Training being called')

learningRate.reset()
softmaxLayer = Softmax(idim=100, odim=10, rng=rng)
model.add_layer(softmaxLayer)

trainingStats, validStats = optimiser.train(model, train_dp, valid_dp)

logger.info('Testing the model on test set trained.')
test_dp = MNISTDataProvider(dset='eval', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)
costWithPretraining, accuracyWithPretraining = optimiser.validate(model, test_dp)
logger.info('MNIST test set accuracy is %.2f %% (cost is %.3f)' % (accuracyWithPretraining * 100., costWithPretraining))



dumpFileNameModel = '../ModelDumps/Part1Task3_AutoencodersWPT.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part1Task3_AutoencodersWPT.pkl'
DumpData(dumpFileNameModel, model)
DumpData(dumpFileNameOptimiser, optimiser)




logger.info('========================================================================================')
logger.info('========================================================================================')
logger.info('Model Without Pre-training')
train_dp.reset()
valid_dp.reset()
learningRate.reset()
modelWithoutPreTraining = MLP(cost=cost)
sigmoidLayer1a = Sigmoid(idim=784, odim=300, rng=rng)
sigmoidLayer2a = Sigmoid(idim=300, odim=100, rng=rng)
softmaxLayera = Softmax(idim=100, odim=10, rng=rng)

modelWithoutPreTraining.add_layer(sigmoidLayer1a)
modelWithoutPreTraining.add_layer(sigmoidLayer2a)
modelWithoutPreTraining.add_layer(softmaxLayera)

optimiserWithoutPretraining = SGDOptimiser(lr_scheduler=learningRate)
trainingStatsWithoutPreTraining, validStatsWithoutPreTraining = optimiserWithoutPretraining.train(modelWithoutPreTraining,
                                                                                                  train_dp,valid_dp)
logger.info('Testing the model on test set trained using no autoencoders')
test_dp.reset()
costWithoutPreTraining, accuracyWithoutPreTraining = optimiserWithoutPretraining.validate(modelWithoutPreTraining, test_dp)
logger.info('MNIST test set accuracy is %.2f %% (cost is %.3f)' % (accuracyWithoutPreTraining * 100., costWithoutPreTraining))


dumpFileNameModel = '../ModelDumps/Part1Task3_AutoencodersWOPT.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part1Task3_AutoencodersWOPT.pkl'
DumpData(dumpFileNameModel, modelWithoutPreTraining)
DumpData(dumpFileNameOptimiser, optimiserWithoutPretraining)





import cPickle

model = None
optimiser = None
dumpFileNameModel = '../ModelDumps/Part1Task3_AutoencodersWPT.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part1Task3_AutoencodersWPT.pkl'

with open(dumpFileNameModel, 'r') as f:
    model = cPickle.load(f)

with open(dumpFileNameOptimiser, 'r') as f:
    optimiser = cPickle.load(f)

naTrainingStatsWithPretraining = numpy.array(optimiser.GetTrainingStats())
naValidationStatsWithPretraining = numpy.array(optimiser.GetValidationStats())

costStatsTrainingWithPretraining = naTrainingStatsWithPretraining[:, 0]
accuracyStatsTrainingWithPretraining = naTrainingStatsWithPretraining[:, 1]
finalCostTrainingWithPretraining = costStatsTrainingWithPretraining[len(costStatsTrainingWithPretraining) - 1]
finalAccuracyWithPretraining = accuracyStatsTrainingWithPretraining[len(accuracyStatsTrainingWithPretraining) - 1] * 100
print('Final Cost (Training) with Pretraining: ', finalCostTrainingWithPretraining)
print('Final Accuracy (Training) with Pretraining: ', finalAccuracyWithPretraining)

costStatsValidationWithPretraining = naValidationStatsWithPretraining[:, 0]
accuracyStatsValidationWithPretraining = naValidationStatsWithPretraining[:, 1]
finalCostValidationWithPretraining = costStatsValidationWithPretraining[len(costStatsValidationWithPretraining) - 1]
finalAccuracyValidationWithPretraining = accuracyStatsValidationWithPretraining[
                                             len(accuracyStatsValidationWithPretraining) - 1] * 100
print('Final Cost (Validation) with Pretraining: ', finalCostValidationWithPretraining)
print('Final Accuracy (Validation) with Pretraining: ', finalAccuracyValidationWithPretraining)


model = None
optimiser = None
dumpFileNameModel = '../ModelDumps/Part1Task3_AutoencodersWOPT.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part1Task3_AutoencodersWOPT.pkl'

with open(dumpFileNameModel, 'r') as f:
    model = cPickle.load(f)

with open(dumpFileNameOptimiser, 'r') as f:
    optimiser = cPickle.load(f)

naTrainingStatsWithoutPretraining = numpy.array(optimiser.GetTrainingStats())
naValidationStatsWithoutPretraining = numpy.array(optimiser.GetValidationStats())

costStatsTrainingWithoutPretraining = naTrainingStatsWithoutPretraining[:, 0]
accuracyStatsTrainingWithoutPretraining = naTrainingStatsWithoutPretraining[:, 1]
finalCostTrainingWithoutPretraining = costStatsTrainingWithoutPretraining[
    len(costStatsTrainingWithoutPretraining) - 1]
finalAccuracyWithoutPretraining = accuracyStatsTrainingWithoutPretraining[
                                      len(accuracyStatsTrainingWithoutPretraining) - 1] * 100
print('Final Cost (Training) without Pretraining: ', finalCostTrainingWithoutPretraining)
print('Final Accuracy (Training) without Pretraining: ', finalAccuracyWithoutPretraining)

costStatsValidationWithoutPretraining = naValidationStatsWithoutPretraining[:, 0]
accuracyStatsValidationWithoutPretraining = naValidationStatsWithoutPretraining[:, 1]
finalCostValidationWithoutPretraining = costStatsValidationWithoutPretraining[
    len(costStatsValidationWithoutPretraining) - 1]
finalAccuracyValidationWithoutPretraining = accuracyStatsValidationWithoutPretraining[
                                                len(accuracyStatsValidationWithoutPretraining) - 1] * 100
print('Final Cost (Validation) without Pretraining: ', finalCostValidationWithoutPretraining)
print('Final Accuracy (Validation) without Pretraining: ', finalAccuracyValidationWithoutPretraining)



intermediateArrayTraining = 1.0 - accuracyStatsTrainingWithPretraining
intermediateArrayValidation = 1.0 - accuracyStatsTrainingWithoutPretraining

plt.plot(intermediateArrayTraining, label='Pre-training')
plt.plot(intermediateArrayValidation, label="Without Pre-training")
plt.title("Error of training")
plt.legend()
plt.show()

plt.plot(costStatsTrainingWithPretraining, label='Pre-training')
plt.plot(costStatsTrainingWithoutPretraining, label="Without Pre-training")
plt.title("Cost of training ")
plt.legend()
plt.show()

intermediateArrayTraining = 1.0 - accuracyStatsValidationWithPretraining
intermediateArrayValidation = 1.0 - accuracyStatsValidationWithoutPretraining
plt.plot(intermediateArrayTraining, label='Pre-training')
plt.plot(intermediateArrayValidation, label="Without Pre-training")
plt.title("Error of validation")
plt.legend()
plt.show()

plt.plot(costStatsValidationWithPretraining, label='Pre-training')
plt.plot(costStatsValidationWithoutPretraining, label="Without Pre-training")
plt.title("Cost of validation")
plt.legend()
plt.show()




