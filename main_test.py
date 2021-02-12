import numpy as np
import pandas as pd

import sklearn as sk
import matplotlib.pyplot as plt
#from obspy import Stream, Trace, UTCDateTime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import pylab as pl
import h5py

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
from sklearn.model_selection import ParameterGrid
import re
from sklearn.metrics import confusion_matrix

import os
import sys
classes_dir = 'D:\Thesis_ssd\MasterThesis3.0'
os.chdir(classes_dir)
from Classes.DataProcessing.LoadData import LoadData
from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.DataProcessing.DataHandler import DataHandler
from Classes.DataProcessing.RamGenerator import RamGenerator
from Classes.DataProcessing.NoiseAugmentor import NoiseAugmentor
from Classes.DataProcessing.TimeAugmentor import TimeAugmentor
from Classes.DataProcessing.RamLoader import RamLoader
from Classes.Modeling.DynamicModels import DynamicModels
from Classes.Modeling.CustomCallback import CustomCallback
from Classes.Scaling.ScalerFitter import ScalerFitter
from Classes.Scaling.MinMaxScalerFitter import MinMaxScalerFitter
from Classes.Scaling.StandardScalerFitter import StandardScalerFitter
#from Classes import Tf_shutup
#Tf_shutup.Tf_shutup()

from livelossplot import PlotLossesKeras

import tensorflow as tf

from matplotlib.colors import ListedColormap

plt.rcParams["figure.figsize"]= (15,15)
helper = HelperFunctions()

import sys
ISCOLAB = 'google.colab' in sys.modules

import random
import pprint

base_dir = 'D:\Thesis_ssd\MasterThesis3.0'
os.chdir(base_dir)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)



def main():
    print("Noise non noise model with upscaling")
    print("Subsample size has a significant impact on computational resources and training time")

    ##### Loading data
    #subsample_size = float(input("Enter subsample size (0 < subsample_size <= 1): "))
    subsample_size = 0.25
    loadData, load_args = load_data(subsample_size)
    full_ds, train_ds, val_ds, test_ds = loadData.get_datasets()
    noise_ds = loadData.noise_ds
    handler = DataHandler(loadData)

    timeAug = None
    if load_args['earth_explo_only']:
        full_and_noise_ds = np.concatenate((full_ds, noise_ds))
        timeAug = TimeAugmentor(handler, full_and_noise_ds, seed = load_args['seed'])
    else:
        timeAug = TimeAugmentor(handler, full_ds, seed = load_args['seed'])

    # Printing data stats:
    print(len(train_ds), len(val_ds), len(test_ds))
    print("All data:")
    classes, counts = handler.get_class_distribution_from_ds(full_ds)
    print("Train set:")
    classes, counts = handler.get_class_distribution_from_ds(train_ds)
    print("Validation set:")
    classes, counts = handler.get_class_distribution_from_ds(val_ds)
    print("Test set:")
    classes, counts = handler.get_class_distribution_from_ds(test_ds)
    print("Nr noise samples " + str(len(loadData.noise_ds)))

    # Model picker 
    model_nr_type = "LSTM"
    is_lstm = True
    num_layers = 1
    decay_sequence = [1]
    use_layerwise_dropout_batchnorm = True

    # Hyperparameters 
    batch_size = 512
    epochs = 30
    learning_rate = 0.05
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=1.0, clipvalue=0.5)
    #opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    activation = 'tanh'
    output_layer_activation = 'sigmoid'
    dropout_rate = 0.3
    filters = 17
    kernel_size = 5
    l1_r = 0.00001
    l2_r = 0.0001
    padding = 'same'
    start_neurons = 4
    num_channels = 3

    # Preprocessing parameters 
    use_noise_augmentor = True
    use_time_augmentor = True
    detrend = False
    use_scaler = True
    use_highpass = False
    highpass_freq = 0.2

    use_tensorboard = False
    use_livelossplot = False
    use_custom = False

    # Initializing preprocessing classes

    scaler = None
    noiseAug = None
    if use_time_augmentor:
        timeAug.fit()
    if use_scaler:
        scaler = StandardScalerFitter(train_ds, timeAug).fit_scaler(detrend = detrend)
    if use_noise_augmentor:
        noiseAug = NoiseAugmentor(train_ds, use_scaler, scaler, loadData, timeAug)

    # Preprocessing and loading all data to RAM:
    ramLoader = RamLoader(handler, timeAug, scaler, noiseAug)
    x_train, y_train = ramLoader.load_to_ram(train_ds, is_lstm)
    x_val, y_val = ramLoader.load_to_ram(val_ds, is_lstm)
    x_test, y_test = ramLoader.load_to_ram(test_ds, is_lstm)

    # Preparing callbacks:
    callbacks = []
    if use_tensorboard:
        import datetime
        clear_tensorboard_dir()
        log_dir = f"{base_dir}/tensorboard_dir/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_ballback)

    if use_custom:
        custom_callback = CustomCallback(data_gen)
        callbacks.append(custom_callback)
    elif use_livelossplot:
        callbacks.append(PlotLossesKeras())

    # Initializing model:
    num_ds, channels, timesteps = handler.get_trace_shape_no_cast(train_ds, use_time_augmentor)
    input_shape = (batch_size, channels, timesteps)

    build_model_args ={"model_type" : model_nr_type,
                        "num_layers": num_layers,
                        "input_shape" : (channels, timesteps),
                        "num_classes" : len(set(loadData.label_dict.values())),
                        "dropout_rate" : dropout_rate,
                        "activation" : activation,
                        "output_layer_activation" : output_layer_activation,
                        "l2_r" : l2_r,
                        "l1_r" : l1_r,
                        "full_regularizer" : True,
                        "start_neurons" : start_neurons,
                        "decay_sequence" : decay_sequence,
                        "filters" : filters,
                        "kernel_size" : kernel_size,
                        "padding" : padding,
                        "use_layerwise_dropout_batchnorm" : use_layerwise_dropout_batchnorm}
    model = DynamicModels(**build_model_args).model


    model_args = {'loss' : "binary_crossentropy",
                  'optimizer' : opt,
                  'metrics' : [tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None), 
                               tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
                               tf.keras.metrics.Recall(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)]}
                           

    model.compile(**model_args)

    # Initializing training:

    gen = RamGenerator(loadData)
    train_gen = gen.data_generator(x_train, y_train, batch_size)
    val_gen = gen.data_generator(x_val, y_val, batch_size)
    test_gen = gen.data_generator(x_test, y_test, batch_size)
    
    args = {'steps_per_epoch' : helper.get_steps_per_epoch(train_ds, batch_size),
            'epochs' : epochs,
            'validation_data' : val_gen,
            'validation_steps' : helper.get_steps_per_epoch(val_ds, batch_size),
            'verbose' : 1,
            'use_multiprocessing' : False, 
            'workers' : 1,
            'callbacks' : callbacks
    }

    model_fit = model.fit(train_gen, **args)

    # Evaluation:
    helper.evaluate_model(model, test_gen, test_ds, batch_size, handler.label_dict)

def load_data(subsample_size = 0.25):
    load_args = {
        'earth_explo_only' : False,
        'noise_earth_only' : False,
        'noise_not_noise' : True,
        'downsample' : True,
        'upsample' : True,
        'frac_diff' : 0.3,
        'seed' : 1,
        'subsample_size' : subsample_size,
        'balance_non_train_set' : True,
        'use_true_test_set' : False
    }
    return LoadData(**load_args), load_args

def clear_tensorboard_dir():
    import os
    import shutil
    path = f"{base_dir}/Tensorboard_dir/fit"
    files = os.listdir(path)
    print(files)
    for f in files:
        shutil.rmtree(os.path.join(path,f))

if __name__ == '__main__':
    main()

