# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh


import numpy
import logging
from mlp.layers import Layer
import cython




logging.basicConfig()
logger = logging.getLogger('My Logger')
logger.setLevel(logging.INFO)

"""
You have been given some very initial skeleton below. Feel free to build on top of it and/or
modify it according to your needs. Just notice, you can factor out the convolution code out of
the layer code, and just pass (possibly) different conv implementations for each of the stages
in the model where you are expected to apply the convolutional operator. This will allow you to
keep the layer implementation independent of conv operator implementation, and you can easily
swap it layer, for example, for more efficient implementation if you came up with one, etc.
"""


def my1_conv2d(image, kernels, strides=(1, 1)):
    """
    Implements a 2d valid convolution of kernels with the inputs
    Note: filer means the same as kernel and convolution (correlation) of those with the inputs space
    produces feature maps (sometimes refereed to also as receptive fields). Also note, that
    feature maps are synonyms here to channels, and as such num_inp_channels == num_inp_feat_maps
    :param image: 4D tensor of sizes (batch_size, num_input_channels, img_shape_x, img_shape_y)
    :param filters: 4D tensor of filters of size (num_inp_feat_maps, num_out_feat_maps, kernel_shape_x, kernel_shape_y)
    :param strides: a tuple (stride_x, stride_y), specifying the shift of the kernels in x and y dimensions
    :return: 4D tensor of size (batch_size, num_out_feature_maps, feature_map_shape_x, feature_map_shape_y)
    """
    imageBatchSize = image.shape[0]
    numberOfInputChannels = image.shape[1]
    numberOfOutputFeatureMaps = kernels.shape[1]
    kernelSizeX = kernels.shape[2]
    kernelSizeY = kernels.shape[3]
    strideX = strides[0]
    strideY = strides[1]
    imageSizeX = image.shape[2]
    imageSizeY = image.shape[3]
    outputarray = numpy.zeros(
        shape=(imageBatchSize, numberOfOutputFeatureMaps, imageSizeX - kernelSizeX + 1, imageSizeY - kernelSizeY + 1))
    for b in range(0, imageBatchSize):
        for oFeatureMap in range(0, numberOfOutputFeatureMaps):
            for xloop in range(0, imageSizeX - kernelSizeX + 1, strideX):
                for yloop in range(0, imageSizeY - kernelSizeY + 1, strideY):
                    filt = kernels[:, oFeatureMap, :, :]
                    imageSlice = image[b, :, xloop:xloop + kernelSizeX, yloop:yloop + kernelSizeY]
                    fImageSlice = imageSlice.flatten()
                    fFilter = filt.flatten()
                    val = numpy.dot(fImageSlice, fFilter)
                    outputarray[b, oFeatureMap, xloop, yloop] = val
    return outputarray


class ConvLinear(Layer):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my1_conv2d,
                 conv_bck=my1_conv2d,
                 conv_grad=my1_conv2d):
        """

        :param num_inp_feat_maps: int, a number of inputs feature maps (channels)
        :param num_out_feat_maps: int, a number of output feature maps (channels)
        :param image_shape: tuple, a shape of the inputs
        :param kernel_shape: tuple, a shape of the kernel
        :param stride: tuple, shift of kernels in both dimensions
        :param irange: float, initial range of the parameters
        :param rng: RandomState object, random number generator
        :param conv_fwd: handle to a convolution function used in fwd-prop
        :param conv_bck: handle to a convolution function used in backward-prop
        :param conv_grad: handle to a convolution function used in pgrads
        :return:
        """

        super(ConvLinear, self).__init__(rng=rng)

        self.numberOfInputFeatureMaps = num_inp_feat_maps
        self.numberOfOutputFeatureMaps = num_out_feat_maps
        self.kernels = self.rng.uniform(-irange, irange, (
            self.numberOfInputFeatureMaps, self.numberOfOutputFeatureMaps, kernel_shape[0], kernel_shape[1]))
        self.stride = stride
        # self.Kernels = numpy.random.random(shape = (self.numberOfInputFeatureMaps, self.numberOfOutputFeatureMaps, kernel_shape[0],kernel_shape[1]))
        self.b = numpy.zeros(self.numberOfOutputFeatureMaps)

    def fprop(self, inputs):
        fpropConv = my1_conv2d(inputs, self.kernels, self.stride)
        for i in range(0, self.numberOfOutputFeatureMaps):
            fpropConv[:, i, :, :] += self.b[i]

        # logger.info('Fprop Conv Layer finished.')

        return fpropConv
        # imageBatchSize = inputs.shape[0]
        # numberOfInputChannels = self.numberOfInputFeatureMaps
        # numberOfOutputFeatureMaps = self.numberOfOutputFeatureMaps
        # kernelSizeX = self.kernels.shape[2]
        # kernelSizeY = self.kernels.shape[3]
        # strideX = self.stride[0]
        # strideY = self.stride[1]
        # imageSizeX = inputs.shape[2]
        # imageSizeY = inputs.shape[3]
        # outputarray = numpy.zeros(shape=(imageBatchSize, numberOfOutputFeatureMaps, imageSizeX - kernelSizeX + 1, imageSizeY - kernelSizeY + 1))
        # for b in range(0, imageBatchSize):
        #     for oFeatureMap in range(0, numberOfOutputFeatureMaps):
        #         for xloop in range(0, imageSizeX - kernelSizeX + 1, strideX):
        #             for yloop in range(0, imageSizeY - kernelSizeY + 1, strideY):
        #                 filt = self.kernels[:, oFeatureMap, :, :]
        #                 imageSlice = inputs[b, :, xloop:xloop + kernelSizeX, yloop:yloop + kernelSizeY]
        #                 fImageSlice = imageSlice.flatten()
        #                 fFilter = filt.flatten()
        #                 val = numpy.dot(fImageSlice, fFilter)
        #                 outputarray[b, oFeatureMap, xloop, yloop] = val + self.b[oFeatureMap]
        # return outputarray

    def bprop(self, h, igrads):


        numberOfKernels = self.kernels.shape[0]  # which is equal to input feature maps
        channels = self.kernels.shape[1]
        kernelSizeX = self.kernels.shape[2]
        kernelSizeY = self.kernels.shape[3]
        imageBatchSize = igrads.shape[0]

        paddedValue = 2 * (self.kernels.shape[2] - 1)

        paddedAarray = numpy.zeros(
            shape=(imageBatchSize, channels, igrads.shape[2] + paddedValue, igrads.shape[3] + paddedValue))
        paddedAarray[:, :, kernelSizeX - 1:igrads.shape[2] + kernelSizeX -1, kernelSizeY - 1:igrads.shape[3] + kernelSizeY-1] = igrads
        # print(paddedAarray.shape)
        temporaryKernels = numpy.zeros(shape=(channels, numberOfKernels, kernelSizeX, kernelSizeY))

        for m in range(0, numberOfKernels):
            for n in range(0, channels):
                temporaryKernels[n, m, :, :] = self.kernels[m, n, :, :]

        for i in range(0, channels):
            for j in range(0, numberOfKernels):
                temporaryKernels[i, j, :, :] = numpy.rot90(temporaryKernels[i, j, :, :], 2)

        # print(temporaryKernels.shape)

        convoulutionOgrad = my1_conv2d(paddedAarray, temporaryKernels, self.stride)

        # for b in range(0, imageBatchSize):
        #     for igradsLoop in range(0, igradsDepth):
        #         for xloop in range(0, igradsX):
        #             for yloop in range(0, igradsY):
        #                 kernel = self.kernels[:, igradsLoop, :, :]
        #                 igradForXAndY = igrads[igradsLoop, xloop, yloop]
        #                 multipliedKernel = kernel * igradForXAndY
        #                 convoulutionOgrad[b, xloop:xloop + strideX + 1, yloop:yloop + strideY + 1] += multipliedKernel

        # logger.info('Bprop Conv Layer finished.')
        return igrads, convoulutionOgrad

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('ConvLinear.bprop_cost method not implemented')

    def pgrads(self, inputs, deltas, l1_weight=0, l2_weight=0):

        imageBatchSize = inputs.shape[0]
        numberOfInputChannels = inputs.shape[1]
        numberOfOutputFeatureMaps = self.numberOfOutputFeatureMaps
        imageSizeX = inputs.shape[2]
        imageSizeY = inputs.shape[3]
        deltasSizeX = deltas.shape[2]
        deltasSizeY = deltas.shape[3]
        kernalSizeX = self.kernels.shape[2]
        kernalSizeY = self.kernels.shape[3]
        strideX = self.stride[0]
        strideY = self.stride[1]

        temporaryImage = numpy.zeros(shape=(numberOfInputChannels, imageBatchSize, imageSizeX, imageSizeY))
        # print(temporaryImage.shape)

        for m in range(0, imageBatchSize):
            for n in range(0, numberOfInputChannels):
                temporaryImage[n, m, :, :] = inputs[m, n, :, :]

        # pgrads = numpy.zeros(shape=(numberOfInputChannels, numberOfOutputFeatureMaps, kernalSizeX, kernalSizeY))
        pgrads = my1_conv2d(temporaryImage, deltas, self.stride)

        # for b in range(0, imageBatchSize):
        #     for oFeatureMap in range(0, numberOfOutputFeatureMaps):
        #         for xloop in range(0, imageSizeX - kernalSizeX + 1, strideX):
        #             for yloop in range(0, imageSizeY - kernalSizeY + 1, strideY):
        #                 singleDelta = deltas[b, oFeatureMap, xloop, yloop]
        #                 imageSlice = temporaryImage[:, b, xloop:xloop + kernalSizeX, yloop:yloop + kernalSizeY]
        #                 pgrads[:, oFeatureMap, :, :] += singleDelta * imageSlice

        numberOfDeltas = deltas.shape[1]
        biasGradient = numpy.zeros(shape=numberOfDeltas)

        for b in range(0, numberOfDeltas):
            biasGradient[b] = sum(deltas[:, b, :, :].flatten())
        # logger.info('Pgrads Conv Layer finished.')
        return pgrads, biasGradient

    def get_params(self):
        return self.kernels, self.b

    def set_params(self, params):
        self.kernels = params[0]
        self.b = params[1]

    def get_name(self):
        return 'convlinear'


# you can derive here particular non-linear implementations:
# class ConvSigmoid(ConvLinear):
# ...

class ConvSigmoid(ConvLinear):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my1_conv2d,
                 conv_bck=my1_conv2d,
                 conv_grad=my1_conv2d):
        super(ConvSigmoid, self).__init__(num_inp_feat_maps=num_inp_feat_maps,num_out_feat_maps=num_out_feat_maps)

    def fprop(self, inputs):
        a = super(ConvSigmoid, self).fprop(inputs)
        numpy.clip(a, -30.0, 30.0, out=a)
        h = 1.0/(1 + numpy.exp(-a))
        return h

    def bprop(self, h, igrads):

        dsigm = h * (1.0 - h)
        deltas = igrads * dsigm
        ___, ograds = super(ConvSigmoid, self).bprop(h=None, igrads=deltas)
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('ConvLinear.bprop_cost method not implemented')

    def get_params(self):
        return self.kernels, self.b

    def set_params(self, params):
        self.kernels = params[0]
        self.b = params[1]

    def get_name(self):
        return 'convSigmoid'


class ConvRelu(ConvLinear):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my1_conv2d,
                 conv_bck=my1_conv2d,
                 conv_grad=my1_conv2d):
        super(ConvRelu, self).__init__(num_inp_feat_maps=num_inp_feat_maps,num_out_feat_maps=num_out_feat_maps)

    def fprop(self, inputs):
        a = super(ConvRelu, self).fprop(inputs)
        h = numpy.clip(a, 0, 20.0)
        return h

    def bprop(self, h, igrads):

        deltas = (h > 0)*igrads
        ___, ograds = super(ConvRelu, self).bprop(h=None, igrads=deltas)
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('ConvLinear.bprop_cost method not implemented')

    def get_params(self):
        return self.kernels, self.b

    def set_params(self, params):
        self.kernels = params[0]
        self.b = params[1]

    def get_name(self):
        return 'convRelu'

class ConvMaxPool2D(Layer):
    def __init__(self,
                 batch_size,
                 num_feat_maps,
                 conv_shape,
                 pool_shape=(2, 2),
                 pool_stride=(2, 2)):
        """

        :param conv_shape: tuple, a shape of the lower convolutional feature maps output
        :param pool_shape: tuple, a shape of pooling operator
        :param pool_stride: tuple, a strides for pooling operator
        :return:
        """

        super(ConvMaxPool2D, self).__init__(rng=None)
        self.poolShape = pool_shape
        self.poolStride = pool_stride
        self.szFMX = conv_shape[0]
        self.szFMY = conv_shape[1]
        self.num_feat_maps = num_feat_maps
        self.G = numpy.zeros((batch_size, num_feat_maps, self.szFMX, self.szFMY))


    def fpropMaxPoolFunction(self, inputs):

        imageBatchSize = inputs.shape[0]
        numberOfFeatureMaps = self.num_feat_maps
        sizeOfFeatureMapX = self.szFMX
        sizeOfFeatureMapY = self.szFMY
        poolStrideX = self.poolStride[0]
        poolStrideY = self.poolStride[1]
        poolShapeX = self.poolShape[0]
        poolShapeY = self.poolShape[1]
        max_li = []
        maxPoolSizeX = sizeOfFeatureMapX / poolShapeX
        maxPoolSizeY = sizeOfFeatureMapY / poolShapeY
        outputArray = numpy.zeros(shape=(imageBatchSize,numberOfFeatureMaps, maxPoolSizeX, maxPoolSizeY))
        # reinitialize G for every fprop, coz we wanna keep the max value for the current fprop
        self.G = numpy.zeros((imageBatchSize, numberOfFeatureMaps, sizeOfFeatureMapX, sizeOfFeatureMapY))
        for b in range(0, imageBatchSize):
            for i in range(0, numberOfFeatureMaps):
                for xloop in range(0, sizeOfFeatureMapX, poolStrideX):
                    for yloop in range(0, sizeOfFeatureMapY, poolStrideY):
                        sliceForMax = inputs[b, i, xloop:xloop + poolShapeX, yloop:yloop + poolShapeY]
                        xshapeOfSlice = sliceForMax.shape[0]
                        yshapeOfSlice = sliceForMax.shape[1]
                        maxSoFar = sliceForMax[0][0]
                        maxPosition = (0, 0)
                        isNewMaxSelected = False
                        for f in range(xshapeOfSlice):
                            for g in range(yshapeOfSlice):
                                currentVal = sliceForMax[f, g]
                                if currentVal > maxSoFar:
                                    maxSoFar = currentVal
                                    maxPosition = (f + xloop, g + yloop)
                                    isNewMaxSelected = True
                        max_li.append(maxSoFar)
                        if not isNewMaxSelected:
                            for f1 in range(xshapeOfSlice):
                                for g1 in range(yshapeOfSlice):
                                    self.G[b, i, f1 + xloop, g1 + yloop] = 1
                        else:
                            self.G[b, i, maxPosition[0], maxPosition[1]] = 1
                        outputArray[b,i,xloop/poolStrideX, yloop/poolStrideY] = maxSoFar

        # logger.info('Fprop max pool finished')
        return outputArray

    def fprop(self, inputs):
        outputArray = self.fpropMaxPoolFunction(inputs)
        return outputArray

    def bpropMaxPool(self,h, igrads):
        imageBatchSize = h.shape[0]
        numberOfFeatureMaps = h.shape[1]
        maxPoolSizeX = h.shape[2]
        maxPoolSizeY = h.shape[3]
        maxStrideX = self.poolStride[0]
        maxStrideY = self.poolStride[1]
        featureMapSizeX = self.szFMX
        featureMapSizeY = self.szFMY
        igradReshaped = numpy.array(igrads).reshape((imageBatchSize, numberOfFeatureMaps, maxPoolSizeX, maxPoolSizeY))
        for b in range(0, imageBatchSize):
            for o in range(0, numberOfFeatureMaps):
                 for xloop in range(0, featureMapSizeX,maxStrideX):
                    for yloop in range(0, featureMapSizeY, maxStrideY):
                        igradValue = igradReshaped[b, o, xloop/maxStrideX, yloop/maxStrideY]
                        # multiply it with the a slice of G
                        sumValue = numpy.sum(self.G[b, o, xloop:xloop + self.poolShape[0], yloop:yloop + self.poolShape[1]])
                        self.G[b, o, xloop:xloop + self.poolShape[0], yloop:yloop + self.poolShape[1]] *= \
                            (float(igradValue) / sumValue)
        # logger.log('Bprop Maxpool finished')
        return self.G

    def bprop(self, h, igrads):
        igradsReceived = igrads
        hReceived = h
        ograds = self.bpropMaxPool(hReceived, igrads=igradsReceived)
        return igradsReceived, ograds


    def get_params(self):
        return []

    def pgrads(self, inputs, deltas, **kwargs):
        return []

    def set_params(self, params):
        pass

    def get_name(self):
        return 'convmaxpool2d'
