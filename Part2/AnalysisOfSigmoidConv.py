import numpy
import scipy.ndimage as scipyImage
import logging
from matplotlib import pyplot
import matplotlib
import cPickle
import matplotlib.pyplot as plt
from mlp.optimisers import SGDOptimiser
model = None
optimiser = None
dumpFileNameModel = '../ModelDumps/Part2Task6_ConvSigmoid.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part2Task6_ConvSigmoid.pkl'

with open(dumpFileNameModel, 'r') as f:
    model = cPickle.load(f)

with open(dumpFileNameOptimiser, 'r') as f:
    optimiser = cPickle.load(f)

tsConvSigmoid1FM = optimiser.GetTrainingStats()
vsConvSigmoid1FM = optimiser.GetValidationStats()

natsConvSigmoid = numpy.array(tsConvSigmoid1FM)
navsConvSigmoid = numpy.array(vsConvSigmoid1FM)

costStatsTrainingConvSigmoid = natsConvSigmoid[:, 0]
accuracyStatsTrainingConvSigmoid = natsConvSigmoid[:, 1]
finalCostTrainingAnnealing = costStatsTrainingConvSigmoid[len(costStatsTrainingConvSigmoid) - 1]
finalAccuracyTrainingAnnealing = accuracyStatsTrainingConvSigmoid[len(accuracyStatsTrainingConvSigmoid) - 1] * 100
print('Final Cost (Training) - Convolution Sigmoid  ', finalCostTrainingAnnealing)
print('Final Accuracy (Training) - Convolution Sigmoid ', finalAccuracyTrainingAnnealing)

costStatsValidationConvSigmoid = navsConvSigmoid[:, 0]
accuracyStatsValidationConvSigmoid = navsConvSigmoid[:, 1]
finalCostValidationAnnealing = costStatsValidationConvSigmoid[len(costStatsValidationConvSigmoid) - 1]
finalAccuracyValidationAnnealing = accuracyStatsValidationConvSigmoid[
                                       len(accuracyStatsValidationConvSigmoid) - 1] * 100
print('Final Cost (Validation) - Convolution Sigmoid ', finalCostValidationAnnealing)
print('Final Accuracy (Validation) - Convolution Sigmoid ', finalAccuracyValidationAnnealing)

errorTrainingConvSigmoid = 1.0 - accuracyStatsTrainingConvSigmoid
errorValidationConvSigmoid = 1.0 - accuracyStatsValidationConvSigmoid

plt.plot(errorTrainingConvSigmoid, label="Training")
plt.plot(errorValidationConvSigmoid, label="Validation")
plt.legend()
plt.title("Error of training and validation - Convolution Sigmoid ")
plt.show()

plt.plot(accuracyStatsTrainingConvSigmoid*100, label="Training")
plt.plot(accuracyStatsValidationConvSigmoid*100, label="Validation")
plt.legend()
plt.title("Accuracy of training and validation - Convolution Sigmoid ")
plt.show()

plt.plot(costStatsTrainingConvSigmoid*100, label="Training")
plt.plot(costStatsValidationConvSigmoid*100, label="Validation")
plt.legend()
plt.title("Cost of training and validation - Convolution Sigmoid ")
plt.show()

