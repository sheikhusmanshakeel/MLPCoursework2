import cPickle
import gzip
import os.path
import random
# Third-party libraries
import numpy as np
import scipy
from scipy import ndimage
from scipy import misc
import numpy
import math
from numpy.random import random_integers
from scipy.signal import convolve2d
from matplotlib import pyplot
import matplotlib
import logging


logging.basicConfig()
logger = logging.getLogger('My Logger')
logger.setLevel(logging.INFO)

logger.info("write something here")


mnistTrainExpanded = "../data/mnist_train_expanded.pkl.gz"
mnistTrain = "../data/mnist_train.pkl.gz"

mnistValidExpanded = "../data/mnist_valid_expanded.pkl.gz"
mnistValid = "../data/mnist_valid.pkl.gz"

mnistEvalExpanded = "../data/mnist_eval_expanded.pkl.gz"
mnistEval = "../data/mnist_eval.pkl.gz"


def CreateGaussianKernel(dimension, sigma):
    kernel = numpy.zeros((dimension, dimension))
    center = dimension / 2
    variance = sigma * sigma
    coefficient = 1. / (2 * variance)

    # create the kernel
    for x in range(0, dimension):
        for y in range(0, dimension):
            x_val = abs(x - center)
            y_val = abs(y - center)
            numerator = x_val ** 2 + y_val ** 2
            denominator = 2 * variance
            kernel[x, y] = coefficient * numpy.exp(-1. * numerator / denominator)

    return kernel / sum(sum(kernel))


def ElasticDeformation(image, kernelDimensions=27, sigma=5, alpha=36):
    image = image.reshape(28, 28)

    result = numpy.zeros(image.shape)
    displacementX = numpy.zeros(image.shape)
    displacementY = numpy.zeros(image.shape)
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            randomInteger = numpy.random.mtrand.random_integers(-1, 1)
            displacementX[x, y] = randomInteger * alpha
            displacementY[x, y] = randomInteger * alpha

    kernel = CreateGaussianKernel(kernelDimensions, sigma)

    displacementX = convolve2d(displacementX, kernel)
    displacementY = convolve2d(displacementY, kernel)
    for i in range(0, image.shape[1]):
        for j in range(0, image.shape[0]):
            lowIndexForDisplacementX = i + int(math.floor(displacementX[i, j]))
            highIndexForDisplacementX = i + int(math.ceil(displacementX[i, j]))
            lowIndexForDisplacementY = j + int(math.floor(displacementY[i, j]))
            highIndexForDisplacementY = j + int(math.ceil(displacementY[i, j]))

            if lowIndexForDisplacementX < 0 or lowIndexForDisplacementY < 0 or highIndexForDisplacementX >= image.shape[
                1] - 1 or highIndexForDisplacementY >= image.shape[0] - 1:
                continue
            # takes the average of the 4 neigbours of the pixel
            res = image[lowIndexForDisplacementX, lowIndexForDisplacementY] + image[
                lowIndexForDisplacementX, highIndexForDisplacementY] + image[
                      highIndexForDisplacementX, lowIndexForDisplacementY] + image[
                      highIndexForDisplacementX, highIndexForDisplacementY]

            result[i, j] = float(res) / 4
    return result.flatten()

def AddGaussianNoiseToImage(image):
    noise = np.random.uniform()
    image = image.reshape(28, 28)
    noisyImage = scipy.ndimage.gaussian_filter(image, sigma=noise)
    noisyImage = noisyImage.flatten()
    return noisyImage


def RotateImage(image):
    image = image.reshape(28, 28)
    rotatedImage = scipy.misc.imrotate(image, 20.0)
    rotatedImage = rotatedImage.flatten()
    return rotatedImage

def ShowImages(normalImage, gaussianImage, rotatedImage):
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 3, 1)
    imgplot = ax.imshow(normalImage.reshape(28, 28), cmap=matplotlib.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    ax = fig.add_subplot(1, 3, 2)
    imgplot = ax.imshow(gaussianImage.reshape(28, 28), cmap=matplotlib.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('right')
    ax = fig.add_subplot(1, 3, 3)
    imgplot = ax.imshow(rotatedImage.reshape(28, 28), cmap=matplotlib.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


def ExpandMnist(fileToExpand, expandedFileName, noiseType):
    if os.path.exists(expandedFileName):
        os.remove(expandedFileName)
        logger.info("The expanded training set already exists so removed it.")

    f = gzip.open(fileToExpand, 'rb')
    training_data = cPickle.load(f)
    f.close()
    expanded_training_pairs = []
    j = 0  # counter
    stopCounter = 0
    newImages = 0
    for x, y in zip(training_data[0], training_data[1]):

        expanded_training_pairs.append((x, y))
        newImageRotated = RotateImage(x)
        newImageGaussian = AddGaussianNoiseToImage(x)
        newImageDeformed = ElasticDeformation(x)
        # if stopCounter < 10:
        #     ShowImages(x, newImageGaussian, newImageRotated)

        randomNumber = numpy.random.uniform(0, 1)
        if randomNumber <= 0.15:
            newImages += 1
            expanded_training_pairs.append((newImageRotated.flatten(), y))
        elif 0.15 < randomNumber <= 0.33:
            newImages += 1
            expanded_training_pairs.append((newImageGaussian.flatten(), y))
        elif 0.33 < randomNumber <= 0.5:
            newImages += 1
            expanded_training_pairs.append((newImageDeformed.flatten(), y))


        stopCounter += 1
        j += 1
        if j % 1000 == 0:
            logger.info("Expanding image number {0}".format(j))
            # break

    random.shuffle(expanded_training_pairs)
    expanded_training_data = [np.array(d) for d in zip(*expanded_training_pairs)]
    logger.info("Saving the expanded MNIST data set.")
    f = gzip.open(expandedFileName, "w")
    cPickle.dump((expanded_training_data), f)
    f.close()
    logger.info('Total number of images  in training set {0}'.format(j))
    logger.info('Total number of new images added in training set {0}'.format(newImages))
    logger.info('Total number of images added in the expanded training set {0}'.format(j + newImages))


ExpandMnist(mnistTrain, mnistTrainExpanded, 1)
# ExpandMnist(mnistEval, mnistEvalExpanded, 1)
# ExpandMnist(mnistValid, mnistValidExpanded,1)

