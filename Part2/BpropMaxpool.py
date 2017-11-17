import numpy as np

bSize = 3
nFMaps = 2
featureMapSizeX = 4
featureMapSizeY = 4

G = np.zeros(shape=(bSize, nFMaps, featureMapSizeX, featureMapSizeY))
GDummy = np.array(
    [
        [
            [
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0]
            ],
            [
                 [1, 0, 0, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 0, 1, 0]
            ]
        ],
        [
             [
                 [1, 0, 1, 0],
                [0, 0, 0, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 0]
            ],
            [
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 0]
            ]
        ],
        [
             [
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 0]
            ],
            [
                 [1, 0, 1, 0],
                [0, 0, 0, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 0]
            ]
        ]
    ]
)

igradsDummy = np.array(
    [
        [
            [
                [1, 2],
                [3, 4]

            ],
            [
                [5, 6],
                [7, 8]
            ]
        ],
        [
             [
                [9, 10],
                [11, 12]
            ],
            [
                [13, 14],
                [15, 16]
            ]
        ],
        [
             [
                [17, 18],
                [19, 20]
            ],
            [
              [21, 22],
                [23, 24]
            ]
        ]
    ]
)
poolShape = (2, 2)
h = np.zeros(shape=(bSize, nFMaps, featureMapSizeX, featureMapSizeY))

imageBatchSize = h.shape[0]
numberOfFeatureMaps = h.shape[1]
maxPoolSizeX = h.shape[2] / poolShape[0]
maxPoolSizeY = h.shape[3] / poolShape[1]
maxStrideX = 2
maxStrideY = 2
igrads = np.ones(shape=(bSize * numberOfFeatureMaps * maxPoolSizeX * maxPoolSizeY))

igradReshaped = np.array(igrads).reshape((imageBatchSize, numberOfFeatureMaps, maxPoolSizeX, maxPoolSizeY))

G = GDummy
igradReshaped = igradsDummy

# print('IGRAD RESHAPED')
#
# print(igradReshaped.shape)
# print(G.shape)

for b in range(0, imageBatchSize):
    for o in range(0, numberOfFeatureMaps):
        for xloop in range(0, featureMapSizeX,maxStrideX):
            for yloop in range(0, featureMapSizeY, maxStrideY):
                igradValue = igradReshaped[b, o, xloop/maxStrideX, yloop/maxStrideY]
                # multiply it with the a slice of G
                G[b, o, xloop:xloop + poolShape[0], yloop:yloop + poolShape[1]] *= igradValue


print(G)
