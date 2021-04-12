import numpy as np
import pandas as pd
import h5py
import sklearn as sk
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split


from tensorflow.keras.layers import Activation, Conv1D, Dense, Dropout, Flatten, MaxPool1D, AveragePooling1D, BatchNormalization, InputLayer, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
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

from Classes.Modeling.InceptionTimeModel import InceptionTimeModel

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
    
    def __init__(self, model_type, num_classes, input_shape, **p):
        self.model_type = model_type
        self.num_classes  = num_classes
        self.input_shape = input_shape
        
        tf.config.optimizer.set_jit(True)
        mixed_precision.set_global_policy('mixed_float16')
        self.load_model(**p)
        
    def load_model(self, **p):
        self.model = None
        if self.model_type == "LSTM":
            self.model = self.create_LSTM_model(**p)
        if self.model_type == "CNN":
            self.model = self.create_CNN_model(**p)
        if self.model_type == "CNN_short":
            self.model = self.create_CNN_short_model(**p)
        if self.model_type == "DENSE":
            self.model = self.create_DENSE_model(**p)
        if self.model_type == "InceptionTime":
            self.model = self.create_InceptionTime_model(**p)
        if self.model_type == "Meier_CNN":
            self.model = self.create_Meier_CNN_model(**p)
        if self.model_type == "Modified_Meier_CNN":
            self.model = self.create_modified_Meier_CNN_model(**p) 
        if self.model_type == "CNN_grow":
            self.model = self.create_CNN_grow_model(**p)
        if self.model_type == "CNN_baseline":
            self.model = self.create_CNN_baseline_model(**p)
        self.model.summary()
        
    
    def output_nr_nodes(self, num_classes, two_output_units = False):
        if two_output_units:
            return 2
        if num_classes > 2:
            return num_classes
        else:
            return 1
      




    def create_LSTM_model(self, **p):
        num_layers = p['num_layers']
        decay_sequence = p['decay_sequence']
        units = p['units']
        l1_r = p['l1_r']
        l2_r = p['l2_r']
        dropout_rate = p['dropout_rate']
        use_layerwise_dropout_batchnorm = p['use_layerwise_dropout_batchnorm']
        dropout_T_bn_F = p['dropout_T_bn_F']
        first_dense_units = p['first_dense_units']
        second_dense_units = p['second_dense_units']
        dense_activation = p['dense_activation']
        output_layer_activation = p['output_layer_activation']


        input_layer = tf.keras.layers.Input(self.input_shape)
        x = input_layer

        for i in range(num_layers):
            return_sequences = False
            if i != num_layers - 1:
                return_sequences = True
            x = CuDNNLSTM(units//decay_sequence[i], 
                        return_sequences = return_sequences,
                        kernel_regularizer = regularizers.l1_l2(l1 = l1_r, l2 = l2_r),
                        bias_regularizer = regularizers.l1_l2(l1 = l1_r, l2 = l2_r))(x)
            
            if use_layerwise_dropout_batchnorm:
                if dropout_T_bn_F:
                    x = Dropout(dropout_rate)(x)
                else:
                    x = BatchNormalization()(x)
        x = Dense(first_dense_units, activation = dense_activation)(x)
        x = Dense(second_dense_units, activation = dense_activation)(x)
        output_layer = Dense(self.output_nr_nodes(self.num_classes), activation = output_layer_activation, dtype = 'float32')(x)
        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
        model.add_loss(1.00)
        return model



    def create_CNN_model(self, **p):

        num_layers = p['num_layers']
        decay_sequence = p['decay_sequence']
        num_filters = p['num_filters']
        filter_size = p['filter_size']
        cnn_activation = p['cnn_activation']
        dense_activation = p['dense_activation']
        padding = p['padding']
        l1_r = p['l1_r']
        l2_r = p['l2_r']
        dropout_rate = p['dropout_rate']
        use_layerwise_dropout_batchnorm = p['use_layerwise_dropout_batchnorm']
        dropout_T_bn_F = p['dropout_T_bn_F']
        first_dense_units = p['first_dense_units']
        second_dense_units = p['second_dense_units']
        output_layer_activation = p['output_layer_activation']


        input_layer = tf.keras.layers.Input(self.input_shape)
        x = input_layer

        for i in range(num_layers):
            x = Conv1D(num_filters//decay_sequence[i], 
                       kernel_size = [filter_size], 
                       padding = padding, 
                       activation = None,
                       kernel_regularizer = regularizers.l1_l2(l1 = l1_r, l2 = l2_r), 
                       bias_regularizer = regularizers.l1_l2(l1 = l1_r, l2 = l2_r))(x)
            if use_layerwise_dropout_batchnorm and not dropout_T_bn_F:
                x = BatchNormalization()(x)
            x = Activation(cnn_activation)(x)
            if use_layerwise_dropout_batchnorm and dropout_T_bn_F:
                x = Dropout(dropout_rate)(x)
            x = MaxPool1D()(x)
        x = Flatten()(x)
        x = Dense(first_dense_units, activation = dense_activation)(x)
        x = Dense(second_dense_units, activation = dense_activation)(x)
        output_layer = Dense(self.output_nr_nodes(self.num_classes), activation = output_layer_activation, dtype = 'float32')(x)
        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
        return model

    def create_CNN_short_model(self, **p):

        num_layers = p['num_layers']
        decay_sequence = p['decay_sequence']
        num_filters = p['num_filters']
        filter_size = p['filter_size']
        cnn_activation = p['cnn_activation']
        dense_activation = p['dense_activation']
        padding = p['padding']
        l1_r = p['l1_r']
        l2_r = p['l2_r']
        dropout_rate = p['dropout_rate']
        use_layerwise_dropout_batchnorm = p['use_layerwise_dropout_batchnorm']
        dropout_T_bn_F = p['dropout_T_bn_F']
        first_dense_units = p['first_dense_units']
        output_layer_activation = p['output_layer_activation']


        input_layer = tf.keras.layers.Input(self.input_shape)
        x = input_layer

        for i in range(num_layers):
            x = Conv1D(num_filters//decay_sequence[i], 
                       kernel_size = [filter_size], 
                       padding = padding, 
                       activation = None,
                       kernel_regularizer = regularizers.l1_l2(l1 = l1_r, l2 = l2_r), 
                       bias_regularizer = regularizers.l1_l2(l1 = l1_r, l2 = l2_r))(x)
            if use_layerwise_dropout_batchnorm and not dropout_T_bn_F:
                x = BatchNormalization()(x)
            x = Activation(cnn_activation)(x)
            if use_layerwise_dropout_batchnorm and dropout_T_bn_F:
                x = Dropout(dropout_rate)(x)
            x = MaxPool1D()(x)
        x = Flatten()(x)
        x = Dense(first_dense_units, activation = dense_activation)(x)
        output_layer = Dense(self.output_nr_nodes(self.num_classes), activation = output_layer_activation, dtype = 'float32')(x)
        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
        return model


    def create_CNN_grow_model(self, **p):

        num_layers = p['num_layers']
        growth_sequence = p['growth_sequence']
        num_filters = p['num_filters']
        filter_size = p['filter_size']
        cnn_activation = p['cnn_activation']
        dense_activation = p['dense_activation']
        padding = p['padding']
        l1_r = p['l1_r']
        l2_r = p['l2_r']
        dropout_rate = p['dropout_rate']
        use_layerwise_dropout_batchnorm = p['use_layerwise_dropout_batchnorm']
        dropout_T_bn_F = p['dropout_T_bn_F']
        first_dense_units = p['first_dense_units']
        output_layer_activation = p['output_layer_activation']


        input_layer = tf.keras.layers.Input(self.input_shape)
        x = input_layer

        for i in range(num_layers):
            x = Conv1D(int(num_filters*growth_sequence[i]), 
                       kernel_size = [filter_size], 
                       padding = padding, 
                       activation = None,
                       kernel_regularizer = regularizers.l1_l2(l1 = l1_r, l2 = l2_r), 
                       bias_regularizer = regularizers.l1_l2(l1 = l1_r, l2 = l2_r))(x)
            if use_layerwise_dropout_batchnorm and not dropout_T_bn_F:
                x = BatchNormalization()(x)
            x = Activation(cnn_activation)(x)
            if use_layerwise_dropout_batchnorm and dropout_T_bn_F:
                x = Dropout(dropout_rate)(x)
            x = MaxPool1D()(x)
        x = Flatten()(x)
        x = Dense(first_dense_units, activation = dense_activation)(x)
        output_layer = Dense(self.output_nr_nodes(self.num_classes), activation = output_layer_activation, dtype = 'float32')(x)
        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
        return model

    
    
    def create_DENSE_model(self, **p):

        num_layers = p['num_layers']
        decay_sequence = p['decay_sequence']
        units = p['units']
        activation = p['activation']
        l1_r = p['l1_r']
        l2_r = p['l2_r']
        dropout_rate = p['dropout_rate']
        use_layerwise_dropout_batchnorm = p['use_layerwise_dropout_batchnorm']
        dropout_T_bn_F = p['dropout_T_bn_F']
        output_layer_activation = p['output_layer_activation']

        input_layer = tf.keras.layers.Input(self.input_shape)
        x = input_layer
        x = Flatten()(x)
        for i in range(num_layers):
            x = Dense(units//decay_sequence[i], 
                      activation = activation,
                      kernel_regularizer = regularizers.l1_l2(l1 = l1_r, l2 = l2_r),
                      activity_regularizer = regularizers.l1_l2(l1 = l1_r, l2 = l2_r))(x)
            if use_layerwise_dropout_batchnorm:
                if dropout_T_bn_F:
                    x = Dropout(dropout_rate)(x)
                else: 
                    x = BatchNormalization()(x)
        output_layer = Dense(self.output_nr_nodes(self.num_classes), activation = output_layer_activation, dtype = 'float32')(x)
        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
        return model
    
    def create_InceptionTime_model(self, **p):
        return InceptionTimeModel(self.input_shape, self.num_classes, **p).model

    def create_Meier_CNN_model(self, **p):

        use_maxpool = p['use_maxpool']
        use_averagepool = p['use_averagepool']
        use_batchnorm = p['use_batchnorm']

        input_layer = tf.keras.layers.Input(self.input_shape)
        x = input_layer

        x = Conv1D(32, kernel_size = 16, padding = "same")(x)
        if use_maxpool:
            x = MaxPool1D()(x)
        if use_averagepool:
            x = AveragePooling1D()(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv1D(64, kernel_size = 16, padding = "same")(x)
        if use_maxpool:
            x = MaxPool1D()(x)
        if use_averagepool:
            x = AveragePooling1D()(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(128, kernel_size = 16, padding = "same")(x)
        if use_maxpool:
            x = MaxPool1D()(x)
        if use_averagepool:
            x = AveragePooling1D()(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Flatten()(x)

        x = Dense(80, activation = 'relu')(x)
        x = Dense(80, activation = 'relu')(x)

        output_layer = Dense(2, activation = 'softmax', dtype = 'float32')(x)

        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)

        loss = "categorical_crossentropy"
        acc = tf.metrics.CategoricalAccuracy(name="categorical_accuracy")

        compile_args = {
            "loss" : loss,
            "optimizer" : tf.keras.optimizers.Adam(learning_rate=0.001),
            "metrics" : [acc,
                        tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None),
                        tf.keras.metrics.Recall(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)]}
        model.compile(**compile_args)
        return model

    def create_modified_Meier_CNN_model(self, **p):
        output_layer_activation = p["output_layer_activation"]

        input_layer = tf.keras.layers.Input(self.input_shape)
        x = input_layer

        x = Conv1D(32, kernel_size = 16)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool1D()(x)
        x = Conv1D(64, kernel_size = 16)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool1D()(x)
        x = Conv1D(128, kernel_size = 16)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool1D()(x)

        x = Dense(80, activation = 'relu')(x)
        x = Dense(80, activation = 'relu')(x)

        output_layer = Dense(self.output_nr_nodes(self.num_classes), activation = output_layer_activation, dtype = 'float32')(x)

        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
        return model

def create_CNN_baseline_model(self, **p):
    cnn_activation = p['cnn_activation']
    dense_activation = p['dense_activation']
    output_layer_activation = p['output_layer_activation']
    input_layer = tf.keras.layers.Input(self.input_shape)
    x = input_layer
    x = Conv1D(32, kernel_size = 16)(x)
    x = Activation(cnn_activation)(x)
    x = Dense(100, activation = dense_activation)(x)
    output_layer = Dense(self.output_nr_nodes(self.num_classes), activation = output_layer_activation, dtype = 'float32')(x)
    model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
    return model
