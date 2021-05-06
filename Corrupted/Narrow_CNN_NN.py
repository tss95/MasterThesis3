import numpy as np
import os
import sys
import pandas as pd
import seaborn as sns

import pylab as pl
import h5py

import tensorflow as tf
from tensorflow.keras import mixed_precision

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES']="0" 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


from sklearn.metrics import confusion_matrix


base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3'
os.chdir(base_dir)
from Classes.DataProcessing.LoadData import LoadData
from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.DataProcessing.DataHandler import DataHandler
from Classes.Modeling.NarrowOpt import NarrowOpt
import json
#from Classes import Tf_shutup
#Tf_shutup.Tf_shutup()

helper = HelperFunctions()

import sys

import random
import gc
import pprint

mixed_precision.set_global_policy('mixed_float16')





load_args = {
    'earth_explo_only' : False,
    'noise_earth_only' : False,
    'noise_not_noise' : True,
    'downsample' : True,
    'upsample' : True,
    'frac_diff' : 1,
    'seed' : 1,
    'subsample_size' : 0.05,
    'balance_non_train_set' : False,
    'use_true_test_set' : False,
    'even_balance' : True
}
loadData = LoadData(**load_args)
train_ds, val_ds, test_ds = loadData.get_datasets()
noise_ds = loadData.noise_ds
handler = DataHandler(loadData)


static_grid = {     'batch_size': 256,
                    'cnn_activation': 'relu',
                    'dense_activation': 'relu',
                    'dropout_T_bn_F': False,
                    'dropout_rate': 0.01,
                    'epochs': 50,
                    'filter_size': 56,
                    'first_dense_units': 252,
                    'growth_sequence': [1, 2, 4, 8, 16],
                    'l1_r': 0.0001,
                    'l2_r': 0.01,
                    'learning_rate': 0.0001,
                    'num_filters': 72,
                    'num_layers': 3,
                    'optimizer': 'adam',
                    'output_layer_activation': 'sigmoid',
                    'padding': 'same',
                    'second_dense_units': 214,
                    'use_layerwise_dropout_batchnorm': False}

search_grid = {
                    "num_layers" : [2, 4, 5],
                    "learning_rate" : [0.001, 0.00001],
                    "batch_size" : [128,512],
                    "epochs" : [50],
                    "optimizer" : ["sgd", "adam"],
                    "num_filters" : np.arange(68, 78 , 2),
                    "filter_size" : np.arange(52, 60, 2),
                    "cnn_activation" : ["tanh", "relu"],
                    "dense_activation" : ["relu", "tanh"],
                    "padding" : ["same"],
                    "use_layerwise_dropout_batchnorm" : [True, False],
                    "dropout_T_bn_F" : [True, False],
                    "growth_sequence" : [[1,4,8,8,4,1],  [1,8,16,32,64,128]],
                    "dropout_rate" : [0],
                    "l2_r" : [0.01, 0.001, 0],
                    "l1_r" : [0.01, 0.001],
                    "first_dense_units" : np.arange(248,290, 2),
                    "second_dense_units" : np.arange(248, 254, 2),
                    "output_layer_activation" : ["sigmoid"]
}

model_type = "CNN_grow_double"
is_lstm = True
num_channels = 3

use_time_augmentor = True
scaler_name = "normalize"
use_noise_augmentor = True
filter_name = None
band_min = 2.0
band_max = 4.0
highpass_freq = 15

use_tensorboard = True
use_liveplots = False
use_custom_callback = True
use_early_stopping = True
start_from_scratch = False
use_reduced_lr = True
log_data = False
skip_to_index = 8

shutdown = False

def clear_tensorboard_dir():
        import os
        import shutil
        path = f"{base_dir}/Tensorboard_dir/fit"
        files = os.listdir(path)
        print(files)
        for f in files:
            shutil.rmtree(os.path.join(path,f))
if use_tensorboard:
    clear_tensorboard_dir()


narrowOpt = NarrowOpt(loadData, model_type, scaler_name, use_time_augmentor, use_noise_augmentor,
                      filter_name, static_grid, search_grid, 
                      use_tensorboard = use_tensorboard, 
                      use_liveplots = use_liveplots, 
                      use_custom_callback = use_custom_callback, 
                      use_early_stopping = use_early_stopping, 
                      band_min = band_min,
                      band_max = band_max, 
                      highpass_freq = highpass_freq, 
                      start_from_scratch = start_from_scratch, 
                      use_reduced_lr = use_reduced_lr, 
                      num_channels = num_channels,
                      log_data = log_data,
                      skip_to_index = skip_to_index)
narrowOpt.fit()
