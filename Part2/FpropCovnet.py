import numpy as np

# def my1_conv2d(inputs, kernels, strides=(1, 1)):
#     """
#     Implements a 2d valid convolution of kernels with the inputs
#     Note: filer means the same as kernel and convolution (correlation) of those with the inputs space
#     produces feature maps (sometimes refereed to also as receptive fields). Also note, that
#     feature maps are synonyms here to channels, and as such num_inp_channels == num_inp_feat_maps
#     :param inputs: 4D tensor of sizes (batch_size, num_input_channels, img_shape_x, img_shape_y)
#     :param filters: 4D tensor of filters of size (num_inp_feat_maps, num_out_feat_maps, kernel_shape_x, kernel_shape_y)
#     :param strides: a tuple (stride_x, stride_y), specifying the shift of the kernels in x and y dimensions
#     :return: 4D tensor of size (batch_size, num_out_feature_maps, feature_map_shape_x, feature_map_shape_y)
#     """
#     imageBatchSize = inputs.shape[0]
#     numberOfInputChannels = inputs.shape[1]
#     imageSizeX = inputs.shape[2]
#     imageSizeY = inputs.shape[3]
#
#     numberOfInputFeatureMaps = kernels.shape[0]
#     numberOfOutputFeatureMaps = kernels.shape[1]
#     kernalSizeX = kernels.shape[2]
#     kernalSizeY = kernels.shape[3]
#
#     strideX = strides.shape[0]
#     strideY = strides.shape[1]
#
#     for xloop in range(0, imageSizeX - kernalSizeX, strideX):
#         for yloop in range(0, imageSizeY - kernalSizeY, strideY):
#             print('something here')
#
#     raise NotImplementedError('Write me!')


batchSize = 10
numberOfInputChannels = 1
xShape = 3
yShape = 3

counter = 0

# inputs = np.random.random(size=(1, 2, 6, 6))
# kernels = np.random.random(size=(2, 2, 3, 3))

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

outputarray = np.random.random(
    size=(imageBatchSize, numberOfOutputFeatureMaps, imageSizeX - kernalSizeX + 1, imageSizeY - kernalSizeY + 1))

strideX = strides[0]
strideY = strides[1]

s = 1
for b in range(0, imageBatchSize):
    s = 1
    for o in range(0, numberOfInputFeatureMaps):
        for x in range(0, imageSizeX):
            for y in range(0, imageSizeY):
                image[b, o, x, y] = image[b, o, x, y] * s
                s += 1
# print(inputs)
for b in range(0, imageBatchSize):
    for oFeatureMap in range(0, numberOfOutputFeatureMaps):
        for xloop in range(0, imageSizeX - kernalSizeX + 1, strideX):
            for yloop in range(0, imageSizeY - kernalSizeY + 1, strideY):
                filt = kernels[:, oFeatureMap, :, :]
                imageSlice = image[b, :, xloop:xloop + kernalSizeX, yloop:yloop + kernalSizeY]
                fImageSlice = imageSlice.flatten()
                fFilter = filt.flatten()
                val = np.dot(fImageSlice, fFilter)
                outputarray[b, oFeatureMap, xloop, yloop] = val
                # print(val)

# print(outputarray)
print(outputarray.shape)






# inputImage = []
# iImage = np.array(inputImage)
# iImage.reshape(shape=(1, 1, 1, 1))
# iImage = np.ones((10, 1, 3, 3))
# iImage[0:5, 0:3, 1:2:2, ]
# for b in range(0, 10):
#     for c in range(0, 1):
#         counter = 1;
#         for x in range(0, 3):
#             for y in range(0, 3):
#                 inputImage[b][c][x][y] = counter;
#                 counter += 1
