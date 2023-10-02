from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from keras import layers

from math import floor, ceil

@dataclass
class NetParams:
    baseWidth: int = 1
    baseHeight: int = 1
    dropoutPeriod: float = 0
    dropoutValue: float = 0.2
    normPeriod: float = 0

    def __post_init__(self):
        self.baseWidth = int(self.baseWidth)
        self.baseHeight = int(self.baseHeight)

        if self.dropoutPeriod == 0:
            self.dropoutValue = 0
        if self.dropoutPeriod > 1:
            self.dropoutPeriod = 1

        if self.normPeriod > 1:
            self.normPeriod = 1
        if floor(self.normPeriod * self.baseHeight) <= 0:
            self.normPeriod = 0

    def genNetwork(self, dataset):
        (baseWidth, baseHeight, dropoutPeriod, normPeriod) = \
            (self.baseWidth, self.baseHeight,
             self.dropoutPeriod, self.normPeriod)

        dropoutEnabled = 0 if dropoutPeriod == 0 else 1
        dropoutHeight = floor(dropoutPeriod * baseHeight) + dropoutEnabled
        dropoutCounter = dropoutEnabled # Start at one

        normHeight = floor(normPeriod * baseHeight)
        normCounter = 0

        fullHeight = baseHeight + dropoutHeight + normHeight

        model = keras.Sequential()
        model.add(keras.Input(shape=(dataset.featureCount,)))

        for i in range(fullHeight):
            if dropoutCounter >= 1:
                dropoutCounter -= 1
                model.add(layers.Dropout(self.dropoutValue))
            elif normCounter >= 1:
                normCounter -= 1
                model.add(layers.LayerNormalization(axis=1))
            else:
                dropoutCounter += dropoutPeriod
                normCounter += normPeriod
                model.add(layers.Dense(baseWidth, activation='relu'))

        if dataset.problemType == 'regression':
            model.add(layers.Dense(dataset.y_size))

            model.compile(
                optimizer='adam',
                loss='mae',
            )

        elif dataset.problemType == 'classification':
            model.add(layers.Dense(dataset.y_size, activation='softmax'))

            model.compile(
                optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=[keras.metrics.SparseCategoricalAccuracy()]
            )

        return model
