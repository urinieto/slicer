import tensorflow as tf

from tensorflow.keras import layers, models, activations

from .constants import N_MELS
from .constants import PATCH_LEN
from .constants import N_CONV_LAYERS
from .constants import N_DENSE_LAYERS


class CNN(object):
    def __init__(self):
        self.model = None
        self.build()

    def add_conv_layer(self):
        self.model.add(layers.Conv2D(32, (3, 3),
                                     input_shape=(N_MELS, PATCH_LEN, 1)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation(activations.relu))
        self.model.add(layers.MaxPooling2D((1, 2)))

    def add_dense_layer(self):
        self.model.add(layers.Dense(256))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation(activations.relu))
        self.model.add(layers.Dropout(0.25))

    def build(self):
        self.model = models.Sequential()

        # Convolutional layers
        for _ in range(N_CONV_LAYERS):
            self.add_conv_layer()

        # Flatten
        self.model.add(layers.Flatten())

        # Dense layers
        for _ in range(N_DENSE_LAYERS):
            self.add_dense_layer()

        # Output
        self.model.add(layers.Dense(1))
        self.model.add(layers.Activation(activations.sigmoid))
