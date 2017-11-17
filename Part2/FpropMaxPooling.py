# So max pooling will take each feature map and convert it into a linear output
# it works by taking 2x2 inputs space from feature maps and takes the largest value
import numpy as np

numberOfFeatureMaps = 2

kernalSizeX = 2
kernalSizeY = 2
imageSizeX = 24
imageSizeY = 24 # this should be the size of the feature map, not size of the image.
imageBatchSize = 2
sizeOfFeatureMapX = imageSizeX
sizeOfFeatureMapY = imageSizeY

# featureMaps = np.random.random(size=(imageBatchSize, numberOfFeatureMaps, sizeOfFeatureMapX, sizeOfFeatureMapY))
featureMaps = np.ones(shape=(imageBatchSize, numberOfFeatureMaps, sizeOfFeatureMapX, sizeOfFeatureMapY))
featureMapSizeX = featureMaps.shape[2]
featureMapSizeY = featureMaps.shape[3]

# s = 1
# for i in range(0, numberOfFeatureMaps):
#     for j in range(0, featureMapSizeX):
#         for k in range(0, featureMapSizeY):
#             featureMaps[i, j, k] = s
#             s += 1

poolStride = (2, 2)
poolShape = (2, 2)
poolStrideX = poolStride[0]
poolStrideY = poolStride[1]
poolShapeX = poolShape[0]
poolShapeY = poolShape[1]

maxPoolSizeX = featureMapSizeX / poolShapeX
maxPoolSizeY = featureMapSizeY / poolShapeY

outputArray = np.zeros(shape=(numberOfFeatureMaps, maxPoolSizeX, maxPoolSizeY))
# indexOfOutputArray = np.zeros(shape=(numberOfFeatureMaps, maxPoolSizeX, maxPoolSizeY))
indexOfOutputArray = [[[None] * maxPoolSizeX] * maxPoolSizeY] * numberOfFeatureMaps
G = np.zeros((imageBatchSize, numberOfFeatureMaps, sizeOfFeatureMapX, sizeOfFeatureMapX))

max_li = []
# print featureMaps.shape
# exit()

for b in range(0, imageBatchSize):
    for i in range(0, numberOfFeatureMaps):
        for xloop in range(0, sizeOfFeatureMapX, poolStrideX):
            for yloop in range(0, sizeOfFeatureMapY, poolStrideY):
                sliceForMax = featureMaps[b, i, xloop:xloop + poolShapeX, yloop:yloop + poolShapeY]
                # print 'debig', sliceForMax.shape, i, xloop, yloop
                xshapeOfSlice = sliceForMax.shape[0]
                yshapeOfSlice = sliceForMax.shape[1]
                maxSoFar = sliceForMax[0][0]
                maxPosition = (0, 0)
                isNewMaxSelected = False
                for f in range(xshapeOfSlice):
                    for g in range(xshapeOfSlice):
                        currentVal = sliceForMax[f, g]
                        if currentVal > maxSoFar:
                            maxSoFar = currentVal
                            maxPosition = (f + xloop, g + yloop)
                            isNewMaxSelected = True
                if not isNewMaxSelected:
                    maxPosition = (xloop,yloop)
                max_li.append(maxSoFar)
                G[b, i, maxPosition[0], maxPosition[1]] = 1





# print(max_li)
# print(len(max_li))

print(G)

print(G.shape)




igrads = np.ones(shape=(imageBatchSize*numberOfFeatureMaps*maxPoolSizeX*maxPoolSizeY))

igradReshaped = np.array(igrads).reshape((imageBatchSize, numberOfFeatureMaps,maxPoolSizeX,maxPoolSizeY))

print('IGRAD RESHAPED')

print(igradReshaped.shape)
print(G.shape)


# for b in range(0, imageBatchSize):
#     for o in range(0, numberOfFeatureMaps):



