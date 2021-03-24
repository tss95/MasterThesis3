import numpy as np
import pandas as pd

import tensorflow as tf

from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.Modeling.InceptionModule import InceptionModule

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


class InceptionTimeModel:

    def __init__(self, input_shape, nr_classes, optimizer, use_residuals = True,
                 use_bottleneck = True, nr_modules = 6, kernel_size = 40, num_filters = 32, bottleneck_size = 32, 
                 shortcut_activation = "relu", module_activation = "linear", module_output_activation = "relu", 
                 output_activation = "sigmoid", l1_r = 0.01, l2_r = 0.01, reg_shortcut = True, reg_module = True):
        self.input_shape = input_shape
        self.nr_classes = nr_classes
        self.optimizer = optimizer

        self.use_residuals = use_residuals
        self.use_bottleneck = use_bottleneck
        self.nr_modules = nr_modules
        self.kernel_size = int(kernel_size)
        self.bottleneck_size = bottleneck_size
        self.num_filters = num_filters
        self.shortcut_activation = shortcut_activation
        self.module_activation = module_activation
        self.module_output_activation = module_output_activation
        self.output_activation = output_activation
        self.l1_r = l1_r
        self.l2_r = l2_r
        self.reg_shortcut = reg_shortcut
        self.reg_module = reg_module

        tf.config.optimizer.set_jit(True)
        mixed_precision.set_global_policy('mixed_float16')

        self.helper = HelperFunctions()
        

    def _shortcut_layer(self, input_tensor, out_tensor, reg_shortcut):
        if not reg_shortcut:
            shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                                padding='same', use_bias=False)(input_tensor)
        else: 
            shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                                padding='same', use_bias=False,
                                                kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.l1_r, 
                                                                                        l2=self.l2_r))(input_tensor)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

        x = tf.keras.layers.Add()([shortcut_y, out_tensor])
        x = tf.keras.layers.Activation(self.shortcut_activation)(x)
        return x

    def build_model(self, input_shape, num_classes, num_modules = 6):
        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.nr_modules):

            x = self._inception_module(x, self.reg_module, activation = self.module_activation)

            if self.use_residuals and d % 3 == 2:
                x = self._shortcut_layer(input_res, x, self.reg_shortcut)
                input_res = x

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        output_layer = tf.keras.layers.Dense(self.output_nr_nodes(num_classes), activation=self.output_activation, dtype = 'float32')(gap_layer)
        
        model = tf.keras.models.Model(inputs=input_layer, outputs = output_layer)
        compile_args = self.helper.generate_model_compile_args(self.optimizer, num_classes)
        model.compile(**compile_args)
        return model
    
    def _inception_module(self, input_tensor, reg_module, stride = 1, activation = 'linear'):
        
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = tf.keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        conv_list = []

        for i in range(len(kernel_size_s)):
            if not reg_module:
                conv_list.append(tf.keras.layers.Conv1D(filters=self.num_filters, kernel_size=kernel_size_s[i],
                                                    strides=stride, padding='same', activation=activation, use_bias=False)(
                    input_inception))
            else: 
                conv_list.append(tf.keras.layers.Conv1D(filters=self.num_filters, 
                                                        kernel_size=kernel_size_s[i],
                                                        strides=stride, 
                                                        padding='same', 
                                                        activation=activation, 
                                                        use_bias=False,
                                                        kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.l1_r, 
                                                                                                l2=self.l2_r))(input_inception))
        max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)
        if not reg_module:
            conv_6 = tf.keras.layers.Conv1D(filters=self.num_filters, 
                                            kernel_size=1,
                                            padding='same', 
                                            activation=activation, 
                                            use_bias=False)(max_pool_1)
        else:
            conv_6 = tf.keras.layers.Conv1D(filters=self.num_filters, 
                                            kernel_size=1,
                                            padding='same', 
                                            activation=activation, 
                                            use_bias=False,
                                            kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.l1_r, 
                                                                                    l2=self.l2_r))(max_pool_1)

        conv_list.append(conv_6)
        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=self.module_output_activation)(x)
        return x
    
    def output_nr_nodes(self, num_classes):
        if num_classes > 2:
            return num_classes
        else:
            return 1
                                                              