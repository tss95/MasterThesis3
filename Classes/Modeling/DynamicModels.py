import numpy as np
import pandas as pd
import h5py
import sklearn as sk
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split


from tensorflow.keras.layers import Activation, Conv1D, Dense, Dropout, Flatten, MaxPool1D, BatchNormalization, InputLayer, LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import utils
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from livelossplot import PlotLossesKeras

import tensorflow as tf
from tensorflow.keras import mixed_precision

class DynamicModels():
    
    """
    Class that allows the user to load a predefined model this class. Models are numbered between 1 and 8.
    
    PARAMETERS
    ---------
    model_nr: (int) which model to load
    input_shape : (int, int, int) shape of the input (batch, number of channels, channel lenght)
    num_classes: (int) number of classes
    dropout_rate: (float)
    activation: (string) activation in the layers
    l2_r: (float) l2 rate
    start_neruons: (int) how many neurons the first layer has. The remaining layers scale off this number
    filters: (int) number of filters of the first layer of the Conv layer
    
    """
    
    def __init__(self, model_type, num_layers,  input_shape, num_classes = 3, dropout_rate = 0.25, 
                 activation = 'relu', output_layer_activation = "softmax", 
                 l2_r = 0.001, l1_r = 0.0001, start_neurons = 512, decay_sequence = [1],
                 full_regularizer = False, filters = 16, kernel_size = 10, padding = 'valid',
                 use_layerwise_dropout_batchnorm = True):
        self.model_type = model_type
        self.num_layers = num_layers
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.output_layer_activation = output_layer_activation
        self.l2_r = l2_r
        self.l1_r = l1_r
        self.start_neurons = start_neurons
        self.decay_sequence = decay_sequence
        self.full_regularizer = full_regularizer
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_layerwise_dropout_batchnorm = use_layerwise_dropout_batchnorm
        tf.config.optimizer.set_jit(True)
        mixed_precision.set_global_policy('mixed_float16')
        self.load_model()
        
    def load_model(self):
        self.model = None
        if self.model_type == "LSTM":
            self.model = self.create_LSTM_model()
        if self.model_type == "CNN":
            self.model = self.create_CNN_model()
        if self.model_type == "DENSE":
            self.model = self.create_DENSE_model()
        self.model.summary()
        
    def weight_variable(self, shape, w=0.1):
        initial = tf.truncated_normal(shape, stddev=w)
        return tf.Variable(initial)
    
    def bias_variable(self, shape, w=0.1):
        initial = tf.constant(w, shape=shape)
        return tf.Variable(initial)
    
    def output_nr_nodes(self, num_classes):
        if num_classes > 2:
            return num_classes
        else:
            return 1
      
                                  
    def create_LSTM_model(self):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape = self.input_shape))
        for i in range(self.num_layers):
            return_sequences = False
            if i != self.num_layers:
                return_sequences = True
            self.model.add(LSTM(int(self.start_neurons//self.decay_sequence[i]), 
                               activation = self.activation,
                               recurrent_activation = "sigmoid",
                               use_bias = True,
                               recurrent_dropout = 0,
                               return_sequences = return_sequences,
                               kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r), 
                               bias_regularizer = regularizers.l2(self.l2_r),
                               activity_regularizer = regularizers.l2(self.l2_r*0.1)))
            if self.use_layerwise_dropout_batchnorm:
                self.model.add(Dropout(self.dropout_rate))
                self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(self.output_nr_nodes(self.num_classes), activation = self.output_layer_activation, dtype = 'float32'))
        
        return self.model
    
    
    """ 
    def create_CNN_model(self):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape = self.input_shape))
        for i in range(self.num_layers):
            self.model.add(Conv1D(int(self.filters//self.decay_sequence[i]), 
                                  kernel_size = [self.kernel_size], 
                                  padding = self.padding, 
                                  activation = None,
                                  kernel_regularizer = regularizers.l1_l2(l1 = self.l1_r, l2 = self.l2_r), 
                                  bias_regularizer = regularizers.l1_l2(l1 = self.l1_r, l2= self.l2_r)))
            self.model.add(Dropout(self.dropout_rate))
            self.model.add(BatchNormalization())
            self.model.add(MaxPool1D())
            self.model.add(tf.keras.layers.Activation(self.activation))(self.model)
            if self.use_layerwise_dropout_batchnorm:
                continue
                #self.model.add(Dropout(self.dropout_rate))
                #self.model.add(BatchNormalization())
        #self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(self.model.output_shape[1]), activation = self.activation)
        self.model.add(Dense(self.model.output_shape[1]//2, activation = self.activation))
        self.model.add(Dense(self.output_nr_nodes(self.num_classes), activation = self.output_layer_activation, dtype = 'float32'))
         
        return self.model

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

    """
    def create_CNN_model(self):
        input_layer = tf.keras.layers.Input(self.input_shape)
        x = input_layer

        for i in range(self.num_layers):
            x = Conv1D(int(self.filters//self.decay_sequence[i]), 
                                  kernel_size = [self.kernel_size], 
                                  padding = self.padding, 
                                  activation = None,
                                  kernel_regularizer = regularizers.l1_l2(l1 = self.l1_r, l2 = self.l2_r), 
                                  bias_regularizer = regularizers.l1_l2(l1 = self.l1_r, l2= self.l2_r))(x)
            x = Dropout(self.dropout_rate)(x)
            x = BatchNormalization()(x)
            x = MaxPool1D()(x)
            x = Activation(self.activation)(x)
        x = Flatten()(x)
        x = Dense(self.start_neurons, activation = self.activation)(x)
        x = Dense(self.start_neurons//2, activation = self.activation)(x)
        output_layer = Dense(self.output_nr_nodes(self.num_classes), activation = self.output_layer_activation, dtype = 'float32')(x)
        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
        return model

    
    
    
    def create_DENSE_model(self):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape = self.input_shape))
        for i in range(self.num_layers):
            self.model.add(Dense(int(self.start_neurons//self.decay_sequence[i]), activation = self.activation,
                                kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r), 
                                bias_regularizer = regularizers.l2(self.l2_r),
                                activity_regularizer = regularizers.l2(self.l2_r*0.1)))
            if self.use_layerwise_dropout_batchnorm:
                self.model.add(Dropout(self.dropout_rate))
                self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(self.output_nr_nodes(self.num_classes), activation = self.output_layer_activation, dtype = 'float32'))
        
        return self.model
            
    
    
    
    