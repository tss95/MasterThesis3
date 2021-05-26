import tensorflow as tf

from Classes.DataProcessing.HelperFunctions import HelperFunctions
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


class InceptionTimeModel:

    """
    Code inspired by, and model gained from: https://github.com/hfawaz/InceptionTime

    Default values: 
    num_modules = 6
    filter_size = 40
    num_filters = 32
    bottleneck_size = 32
    residual_activation = relu
    module_activation = linear
    module_output_activation = relu
    output_activation = sigmoid

    """


    def __init__(self, input_shape, num_classes, **p):
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.use_residuals = p['use_residuals']
        self.use_bottleneck = p['use_bottleneck']
        self.num_modules = p['num_modules']
        self.filter_size = p['filter_size']
        self.bottleneck_size = p['bottleneck_size']
        self.num_filters = p['num_filters']
        self.residual_activation = p['residual_activation']
        self.module_activation = p['module_activation']
        self.module_output_activation = p['module_output_activation']
        self.output_layer_activation = p['output_layer_activation']
        self.l1_r = p['l1_r']
        self.l2_r = p['l2_r']
        self.reg_residual = p['reg_residual']
        self.reg_module = p['reg_module']

        tf.config.optimizer.set_jit(True)
        mixed_precision.set_global_policy('mixed_float16')

        self.helper = HelperFunctions()
        self.model = self.build_model(self.input_shape, self.num_classes, self.num_modules)
        

    def _residual_layer(self, input_tensor, out_tensor, reg_residual):
        if not reg_residual:
            residual_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                                padding='same', use_bias=False)(input_tensor)
        else: 
            residual_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                                padding='same', use_bias=False,
                                                kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.l1_r, 
                                                                                        l2=self.l2_r))(input_tensor)
        residual_y = tf.keras.layers.BatchNormalization()(residual_y)

        x = tf.keras.layers.Add()([residual_y, out_tensor])
        x = tf.keras.layers.Activation(self.residual_activation)(x)
        return x

    def build_model(self, input_shape, num_classes, num_modules):
        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(num_modules):

            x = self._inception_module(x, self.reg_module, activation = self.module_activation)

            if self.use_residuals and d % 3 == 2:
                x = self._residual_layer(input_res, x, self.reg_residual)
                input_res = x

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        output_layer = tf.keras.layers.Dense(self.output_nr_nodes(num_classes), activation=self.output_layer_activation, dtype = 'float32')(gap_layer)
        
        model = tf.keras.models.Model(inputs=input_layer, outputs = output_layer)
        return model
    
    def _inception_module(self, input_tensor, reg_module, stride = 1, activation = 'linear'):
        
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = tf.keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        filter_size_s = [self.filter_size // (2 ** i) for i in range(3)]
        conv_list = []

        for i in range(len(filter_size_s)):
            if not reg_module:
                conv_list.append(tf.keras.layers.Conv1D(filters=self.num_filters, kernel_size=[filter_size_s[i]],
                                                    strides=stride, padding='same', activation=activation, use_bias=False)(
                    input_inception))
            else: 
                conv_list.append(tf.keras.layers.Conv1D(filters=self.num_filters, 
                                                        kernel_size=[filter_size_s[i]],
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
                                                              