import numpy as np
import pandas as pd
import h5py
import sklearn as sk
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

import keras

from keras.layers import Activation, Conv1D, Dense, Dropout, Flatten, MaxPooling3D, BatchNormalization, InputLayer, LSTM
from keras.layers import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.utils import Sequence
from keras.optimizers import Adam
from tensorflow.keras import regularizers
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from ..DataProcessing import DataGenerator
from livelossplot import PlotLossesKeras

import tensorflow as tf

class StaticModels():
    
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
    
    def __init__(self, model_nr, input_shape, num_classes = 3, dropout_rate = 0.25, 
                 activation = 'relu', output_layer_activation = "softmax", 
                 l2_r = 0.001, l1_r = 0.0001, start_neurons = 512,
                 full_regularizer = False, filters = 16, kernel_size = 10, padding = 'valid'):
        self.model_nr = model_nr
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.output_layer_activation = output_layer_activation
        self.l2_r = l2_r
        self.l1_r = l1_r
        self.start_neurons = start_neurons
        self.full_regularizer = full_regularizer
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.load_model()
        
    def load_model(self):
        self.model = None
        if self.model_nr == 1:
            self.create_model_1()
        if self.model_nr == 2:
            self.create_model_2()
        if self.model_nr == 3:
            self.create_model_3()
        if self.model_nr == 4:
            self.create_model_4()
        if self.model_nr == 5:
            self.create_model_5()
        if self.model_nr == 6:
            self.create_model_6()
        if self.model_nr == 7:
            self.create_model_7()
        if self.model_nr == 8:
            self.create_model_8()
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
    
    def create_model_1(self):
        self.model = Sequential()
        
        input_layer = InputLayer(batch_input_shape = self.input_shape)
        self.model.add(input_layer)
        if self.full_regularizer:
            #w_init = tf.random_normal_initializer()
            self.model.add(LSTM(self.start_neurons, activation = self.activation,
                           kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r), 
                           bias_regularizer = regularizers.l2(self.l2_r*0.1),
                           activity_regularizer = regularizers.l2(self.l2_r*0.1)))
        else:
            self.model.add(LSTM(self.start_neurons, activation=self.activation))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate))

        self.model.add(Dense(self.start_neurons//2,activation=self.activation, 
                             kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate))

        self.model.add(Dense(self.start_neurons//4,activation=self.activation, 
                             kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate))

        self.model.add(Dense(self.start_neurons//8,activation=self.activation, 
                             kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate/2))

        self.model.add(Dense(self.output_nr_nodes(self.num_classes), activation=self.output_layer_activation))

        return self.model
        
    def create_model_2(self):
        self.model = Sequential()
        self.model.add(InputLayer(batch_input_shape = self.input_shape))
        self.model.add(Dense(self.start_neruons, activation = self.activation))
        if self.full_regularizer:
            #initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.5, seed=None)
            self.model.add(Dense(self.start_neurons,
                                kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r), 
                                bias_regularizer = regularizers.l2(self.l2_r),
                                activity_regularizer = regularizers.l2(self.l2_r*0.1)))
        else:
            self.model.add(Dense(start_neurons, activation=self.activation))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate))

        self.model.add(Dense(self.start_neurons//2,activation=self.activation, 
                             kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate))

        self.model.add(Dense(self.start_neurons//4,activation=self.activation, 
                             kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate))

        self.model.add(Dense(self.start_neurons//8,activation=self.activation, 
                             kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate/2))

        self.model.add(Dense(self.start_neurons//16,activation=self.activation,
                             kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate/4))

        self.model.add(Dense(self.start_neurons//32,activation=self.activation,
                             kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate/4))
        self.model.add(Flatten())

        self.model.add(Dense(self.output_nr_nodes(self.num_classes), activation=self.output_layer_activation))
        return self.model
        
    def create_model_3(self):
        self.model = Sequential()
        self.model.add(InputLayer(batch_input_shape = self.input_shape))
        if self.full_regularizer:
            #initializer = tf.Variable(lambda : tf.truncated_normal([0, 0.5]))
            self.model.add(Dense(self.start_neurons, activation = self.activation,
                                kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r), 
                                bias_regularizer = regularizers.l2(self.l2_r),
                                activity_regularizer = regularizers.l2(self.l2_r*0.1)))
        else:
            self.model.add(Dense(self.start_neurons, activation = self.activation))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(self.output_nr_nodes(self.num_classes), activation=self.output_layer_activation))
        return self.model
    
    def create_model_4(self):
        self.model = Sequential()
        self.model.add(InputLayer(batch_input_shape = self.input_shape))
        if self.full_regularizer:
            #initializer = initializer = tf.Variable(lambda : tf.compat.v1.truncated_normal([0, 1]))
            self.model.add(LSTM(self.start_neurons, activation = self.activation,
                           kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r), 
                           bias_regularizer = regularizers.l2(self.l2_r),
                           activity_regularizer = regularizers.l2(self.l2_r*0.1)))
        else:
            self.model.add(LSTM(self.start_neurons, activation = self.activation))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(BatchNormalization())
        #model.add(Flatten())
        self.model.add(Dense(self.output_nr_nodes(self.num_classes), activation=self.output_layer_activation))
        return self.model
    
    
    def create_model_5(self):
        self.model = Sequential()
        self.model.add(InputLayer(batch_input_shape = self.input_shape))
        if self.full_regularizer:
            self.model.add(Conv1D(self.filters, kernel_size = self.kernel_size, padding = self.padding, activation = self.activation,
                                  kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r), 
                                  bias_regularizer = regularizers.l2(self.l2_r),
                                  activity_regularizer = regularizers.l2(self.l2_r*0.1)))
        else: 
            self.model.add(Conv1D(filters  = self.filters, activation = self.activation))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(self.output_nr_nodes(self.num_classes), activation = self.output_layer_activation))
        print(self.output_layer_activation)
         
        return self.model
    
    def create_model_6(self):
        self.model = Sequential()
        self.model.add(InputLayer(batch_input_shape = self.input_shape))
        
        self.model.add(Conv1D(self.filters, kernel_size = self.kernel_size, padding = self.padding, activation = self.activation,
                              kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r), 
                              bias_regularizer = regularizers.l2(self.l2_r),
                              activity_regularizer = regularizers.l2(self.l2_r*0.1)))
        
        
        self.model.add(Conv1D(self.filters//2, kernel_size = self.kernel_size//2, padding = self.padding, activation = self.activation,
                              kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r), 
                              bias_regularizer = regularizers.l2(self.l2_r),
                              activity_regularizer = regularizers.l2(self.l2_r*0.1)))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(self.output_nr_nodes(self.num_classes), activation = self.output_layer_activation))
        print(self.output_layer_activation)
         
        return self.model
    
    def create_model_7(self):
        self.model = Sequential()
        self.model.add(InputLayer(batch_input_shape = self.input_shape))
        self.model.add(LSTM(self.start_neurons, activation = self.activation,
                       return_sequences = True,
                       kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r), 
                       bias_regularizer = regularizers.l2(self.l2_r),
                       activity_regularizer = regularizers.l2(self.l2_r*0.1)))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(self.start_neurons//2, activation = self.activation,
                       kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r), 
                       bias_regularizer = regularizers.l2(self.l2_r),
                       activity_regularizer = regularizers.l2(self.l2_r*0.1)))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(self.output_nr_nodes(self.num_classes), activation=self.output_layer_activation))
        return self.model
    
    
    def create_model_8(self):
        self.model = Sequential()
        self.model.add(InputLayer(batch_input_shape = self.input_shape))
        self.model.add(LSTM(self.start_neurons, activation = self.activation,
                       return_sequences = True,
                       kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r), 
                       bias_regularizer = regularizers.l2(self.l2_r),
                       activity_regularizer = regularizers.l2(self.l2_r*0.1)))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(self.start_neurons//2, activation = self.activation,
                       return_sequences = True,
                       kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r), 
                       bias_regularizer = regularizers.l2(self.l2_r),
                       activity_regularizer = regularizers.l2(self.l2_r*0.1)))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(BatchNormalization())
        self.model.add(Dense(self.start_neurons//4, activation = self.activation,
                       kernel_regularizer = regularizers.l1_l2(l1=self.l1_r, l2=self.l2_r), 
                       bias_regularizer = regularizers.l2(self.l2_r),
                       activity_regularizer = regularizers.l2(self.l2_r*0.1)))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(self.output_nr_nodes(self.num_classes), activation=self.output_layer_activation))
        return self.model
      
                                  
    
    
    
    
    
    