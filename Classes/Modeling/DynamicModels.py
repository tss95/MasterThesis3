
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, MaxPool1D, AveragePooling1D, BatchNormalization, Permute, GlobalAveragePooling1D, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras import regularizers
from tensorflow.keras import mixed_precision

from Classes.Modeling.InceptionTimeModel import InceptionTimeModel
from Classes.DataProcessing.HelperFunctions import HelperFunctions

class DynamicModels():
    
    """
    Class that allows the user to load a predefined model this class. Models are numbered between 1 and 8.
    
    PARAMETERS
    -------------------------------------------------------------------------------------
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
        if self.model_type == "LSTM_baseline":
            self.model = self.create_LSTM_baseline_model(**p)
        if self.model_type == "LSTM_FCN":
            self.model = self.create_LSTM_FCN_model(**p)
        if self.model_type == "MLSTM_FCN":
            self.model = self.create_MLSTM_FCN_model(**p)
        if self.model_type == "MLSTM_FCN_edited":
            self.model = self.create_MLSTM_FCN_edited_model(**p)
        if self.model_type == "CNN":
            self.model = self.create_CNN_model(**p)
        if self.model_type == "CNN_short":
            self.model = self.create_CNN_short_model(**p)
        if self.model_type == "DENSE":
            self.model = self.create_DENSE_model(**p)
        if self.model_type == "DENSE_grow":
            self.model = self.create_DENSE_grow_model(**p)
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
        if self.model_type == "CNN_grow_double":
            self.model = self.create_CNN_grow_double_model(**p)
        if self.model_type == "DENSE_baseline":
            self.model = self.create_DENSE_baseline_model(**p)
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
        return model

    def create_LSTM_baseline_model(self, **p):
        units = p['units']
        output_layer_activation = p['output_layer_activation']
        input_layer = tf.keras.layers.Input(self.input_shape)
        x = input_layer
        x = CuDNNLSTM(units)(x)
        output_layer = Dense(self.output_nr_nodes(self.num_classes), activation = output_layer_activation, dtype = 'float32')(x)
        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
        return model

    def create_LSTM_FCN_model(self, **p):
        # Code borrowed from https://github.com/houshd/LSTM-FCN/blob/master/hyperparameter_search.py
        units = p['units']
        output_layer_activation = p['output_layer_activation']
        
        input_layer = tf.keras.layers.Input(self.input_shape)
        ip = input_layer

        x = CuDNNLSTM(units, name = "lstm_block_out")(ip)
        x = Dropout(0.8)(x)

        #y = Permute((2,1))(ip)
        # Filter_size = 8
        #y = Conv1D(128, 16, padding = "same", kernel_initializer ="he_uniform")(y)
        y = Conv1D(60, 80, padding = "same", kernel_initializer ="he_uniform")(ip)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        # Filter_size = 5
        y = Conv1D(120, 80, padding = "same", kernel_initializer ="he_uniform")(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        # Filter size = 3
        y = Conv1D(240, 80, padding = "same", kernel_initializer ="he_uniform")(y)
        y = BatchNormalization()(y)
        y = Activation('relu', name= "conv_block_activation")(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        output_layer = Dense(self.output_nr_nodes(self.num_classes), activation = output_layer_activation, dtype = 'float32')(x)
        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)

        return model
    
    def create_MLSTM_FCN_model(self, **p):
        # Code from: https://github.com/fazlekarim/MLSTM-FCN/
        units = p['units']
        output_layer_activation = p['output_layer_activation']
        
        input_layer = Input(self.input_shape)
        ip = input_layer

        # LSTM block
        # Permute layer is noramlly first in convolution block
        x = Permute((2,1))(ip)
        x = Masking()(x)
        x = LSTM(units, name = "lstm_block_out")(x)
        x = Dropout(0.8)(x)

        y = Conv1D(128, 8, padding = "same", kernel_initializer ="he_uniform")(ip)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(256, 5, padding = "same", kernel_initializer ="he_uniform")(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(128, 3, padding = "same", kernel_initializer ="he_uniform")(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])
        output_layer = Dense(self.output_nr_nodes(self.num_classes), activation = output_layer_activation, dtype = 'float32')(x)
        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)

        return model


    def create_MLSTM_FCN_edited_model(self, **p):
        units = p['units']
        output_layer_activation = p['output_layer_activation']
        
        input_layer = Input(self.input_shape)
        ip = input_layer

        # LSTM block
        # Permute layer is noramlly first in convolution block
        x = Permute((2,1))(ip)
        x = Masking()(x)
        x = LSTM(units, name = "lstm_block_out")(x)
        x = Dropout(0.8)(x)

        y = Conv1D(78, 72, padding = "same", kernel_initializer ="he_uniform")(ip)
        y = MaxPool1D()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(312, 72, padding = "same", kernel_initializer ="he_uniform")(y)
        y = MaxPool1D()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(624, 72, padding = "same", kernel_initializer ="he_uniform")(y)
        y = MaxPool1D()(y)
        y = Activation('relu')(y)
        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])
        output_layer = Dense(self.output_nr_nodes(self.num_classes), activation = output_layer_activation, dtype = 'float32')(x)
        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)

        return model

        
    def squeeze_excite_block(self, input):
        # Code from: https://github.com/fazlekarim/MLSTM-FCN/
        ''' Create a squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
            k: width factor
        Returns: a keras tensor
        '''
        filters = input.shape[-1] # channel_axis = -1 for TF

        se = GlobalAveragePooling1D()(input)
        se = Reshape((1, filters))(se)
        se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = multiply([input, se])
        return se


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

    def create_CNN_grow_double_model(self, **p):

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
        second_dense_units = p['second_dense_units']
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
        x = Dense(second_dense_units, activation = dense_activation)(x)
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


    def create_DENSE_grow_model(self, **p):
        num_layers = p['num_layers']
        growth_sequence = p['growth_sequence']
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
            x = Dense(units*growth_sequence[i], 
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

    def create_DENSE_baseline_model(self, **p):
        units = p['units']
        dense_activation = p['dense_activation']
        output_layer_activation = p['output_layer_activation']

        input_layer = tf.keras.layers.Input(self.input_shape)
        x = input_layer
        x = Flatten()(x)
        x = Dense(units, activation = dense_activation)(x)
        output_layer = Dense(self.output_nr_nodes(self.num_classes), activation = output_layer_activation, dtype = 'float32')(x)
        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
        return model
    
    def create_InceptionTime_model(self, **p):
        return InceptionTimeModel(self.input_shape, self.num_classes, **p).model

    def create_Meier_CNN_model(self, **p):
        helper = HelperFunctions()

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
                        helper.precision,
                        helper.recall]}
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
        x = Conv1D(16, kernel_size = 8, padding = "same")(x)
        x = Activation(cnn_activation)(x)
        x = GlobalAveragePooling1D()(x)
        output_layer = Dense(self.output_nr_nodes(self.num_classes), activation = output_layer_activation, dtype = 'float32')(x)
        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
        return model
