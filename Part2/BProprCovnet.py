import numpy as np

image = np.ones(shape=(2, 2, 3, 3))
kernels = np.ones(shape=(2, 3, 2, 2))
strides = (1, 1)

imageBatchSize = image.shape[0]
numberOfInputChannels = image.shape[1]
imageSizeX = image.shape[2]
imageSizeY = image.shape[3]

numberOfInputFeatureMaps = kernels.shape[0]
numberOfOutputFeatureMaps = kernels.shape[1]
kernalSizeX = kernels.shape[2]
kernalSizeY = kernels.shape[3]

convoulutionOgrad = np.zeros(shape=(numberOfInputChannels, imageSizeX, imageSizeY))

for i in range(0, numberOfInputChannels):
    s = 1
    for j in range(0, imageSizeX):
        for k in range(0, imageSizeY):
            convoulutionOgrad[i, j, k] = s
            s += 1

# the dimensions of the igrads will be the dimensions of the feature map

igrads = np.ones(shape=(numberOfOutputFeatureMaps, imageSizeX - kernalSizeX + 1, imageSizeY - kernalSizeY + 1))
igradsDepth = igrads.shape[0]
igradsX = igrads.shape[1]
igradsY = igrads.shape[2]
strideX = strides[0]
strideY = strides[1]

for igradsLoop in range(0, igradsDepth):
    for xloop in range(0, igradsX):
        for yloop in range(0, igradsY):
            kernel = kernels[:, igradsLoop, :, :]
            igradForXAndY = igrads[igradsLoop, xloop, yloop]
            multipliedKernel = kernel * igradForXAndY
            # test = convoulutionOgrad[:, 0:2, 0:2]
            # ogradSlice = convoulutionOgrad[:, xloop:xloop + strideX+1, yloop:yloop + strideY+1]
            convoulutionOgrad[:, xloop:xloop + strideX + 1, yloop:yloop + strideY + 1] += multipliedKernel
            # print(ogradSlice)

print(convoulutionOgrad)



