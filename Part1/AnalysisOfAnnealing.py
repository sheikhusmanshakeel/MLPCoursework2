import numpy
import scipy.ndimage as scipyImage
import logging
from matplotlib import pyplot
import matplotlib
import cPickle
import matplotlib.pyplot as plt

model = None
optimiser = None
dumpFileNameModel = '../ModelDumps/Part1Task2_Annealing.pkl'
dumpFileNameOptimiser = '../OptimiserDumps/Part1Task2_Annealing.pkl'

with open(dumpFileNameModel, 'r') as f:
    model = cPickle.load(f)

with open(dumpFileNameOptimiser, 'r') as f:
    optimiser = cPickle.load(f)

annealingTrainingStats = optimiser.GetTrainingStats()
annealingValidationStats = optimiser.GetValidationStats()

naTrainingStatsAnnealing = numpy.array(annealingTrainingStats)
naValidationStatsAnnealing = numpy.array(annealingValidationStats)

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

errorTraining = 1.0 - accuracyStatsTrainingAnnealing
errorValidation = 1.0 - accuracyStatsValidationAnnealing

plt.plot(errorTraining, label="Training")
plt.plot(errorValidation, label="Validation")
plt.legend()
plt.title("Error of training and validation - Annealing")
plt.show()

plt.plot(accuracyStatsTrainingAnnealing, label="Training")
plt.plot(accuracyStatsValidationAnnealing, label="Validation")
plt.legend()
plt.title("Accuracy of training and validation - Annealing")
plt.show()

for i in xrange(len(model.layers)):
    weightsForLayer = model.layers[i].get_params()[0]
    hist = numpy.histogram(weightsForLayer,)
    plt.hist(weightsForLayer, label="Histogram of weights for layer {0}".format(i + 1))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
