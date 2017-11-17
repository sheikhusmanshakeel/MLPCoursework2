import numpy as np

inputs = np.ones(shape=(2, 2, 3, 3))
kernels = np.ones(shape=(2, 3, 2, 2))
strides = (1, 1)

imageBatchSize = inputs.shape[0]
numberOfInputChannels = inputs.shape[1]
imageSizeX = inputs.shape[2]
imageSizeY = inputs.shape[3]

numberOfInputFeatureMaps = kernels.shape[0]
numberOfOutputFeatureMaps = kernels.shape[1]
kernalSizeX = kernels.shape[2]
kernalSizeY = kernels.shape[3]

deltas = np.random.random(
    size=(imageBatchSize, numberOfOutputFeatureMaps, imageSizeX - kernalSizeX + 1, imageSizeY - kernalSizeY + 1))
deltasSizeX = deltas.shape[2]
deltasSizeY = deltas.shape[3]

strideX = strides[0]
strideY = strides[1]

s = 1
for b in range(0, imageBatchSize):
    s = 1
    for o in range(0, numberOfInputFeatureMaps):
        for x in range(0, imageSizeX):
            for y in range(0, imageSizeY):
                inputs[b, o, x, y] = inputs[b, o, x, y] * s
                s += 1
s = 0
for b in range(0, imageBatchSize):
    s = b * 2
    for o in range(0, numberOfOutputFeatureMaps):
        for x in range(0, deltasSizeX):
            for y in range(0, deltasSizeY):
                deltas[b, o, x, y] = s
                s += 1

# pgrads return something as kernels so same shape as kernels
# calculate pgrads which is the same size as kernels
pgrads = np.zeros(shape=(numberOfInputFeatureMaps, numberOfOutputFeatureMaps, kernalSizeX, kernalSizeY))

for b in range(0, imageBatchSize):
    for oFeatureMap in range(0, numberOfOutputFeatureMaps):
        for xloop in range(0, imageSizeX - kernalSizeX + 1, strideX):
            for yloop in range(0, imageSizeY - kernalSizeY + 1, strideY):
                singleDelta = deltas[b, oFeatureMap, xloop, yloop]
                imageSlice = inputs[b, :, xloop:xloop + kernalSizeX, yloop:yloop + kernalSizeY]
                pgrads[:, oFeatureMap, :, :] += singleDelta  * imageSlice


print(pgrads)