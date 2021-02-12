import numpy as np
import pandas as pd
import tensorflow as tf


class InceptionModule(tf.keras.layers.Layer):

    def __init__(self, num_filters = 32, activation = 'relu', **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.activation = tf.keras.activations.get(activation)
    
    def _default_Conv1D(self, filters, kernel_size):
        return tf.keras.layers.Conv1D(filters = filters, 
                                      kernel_size = kernel_size, 
                                      strides = 1, 
                                      activation = self.activation,
                                      data_format='channels_first')
    def call(self, inputs):
        # Step 1:
        z_bottleneck = self._default_Conv1D(filters = self.num_filters, kernel_size = 1)(inputs)
        z_maxpool = tf.keras.layers.MaxPool1D(pool_size = 3, strides = 1, padding = 'same')(inputs)

        # Step 2:
        z1 = self._default_Conv1D(filters=self.num_filters, kernel_size = 10)(z_bottleneck)
        z2 = self._default_Conv1D(filters=self.num_filters, kernel_size = 20)(z_bottleneck)
        z3 = self._default_Conv1D(filters=self.num_filters, kernel_size = 40)(z_bottleneck)
        z4 = self._default_Conv1D(filters=self.num_filters, kernel_size = 1)(z_maxpool)

        # Step 3:
        z = tf.keras.layers.Concatenate(axis = 2)([z1,z2,z3,z4])
        z = tf.keras.layers.BatchNormalization()(z)

        return self.activation(z)





