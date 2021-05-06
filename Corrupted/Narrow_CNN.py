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
    'earth_explo_only' : True,
    'noise_earth_only' : False,
    'noise_not_noise' : False,
    'downsample' : True,
    'upsample' : True,
    'frac_diff' : 1,
    'seed' : 1,
    'subsample_size' : 0.25,
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
                    'dropout_T_bn_F': True,
                    'dropout_rate': 0.001,
                    'epochs': 50,
                    'filter_size': 74,
                    'first_dense_units': 254,
                    'growth_sequence': [1, 4, 8, 8, 8],
                    'l1_r': 0.0,
                    'l2_r': 0.001,
                    'learning_rate': 0.001,
                    'num_filters': 78,
                    'num_layers': 3,
                    'optimizer': 'adam',
                    'output_layer_activation': 'sigmoid',
                    'padding': 'same',
                    'second_dense_units': 234,
                    'use_layerwise_dropout_batchnorm': True}

search_grid = {
                    "growth_sequence" : [[1,2,4,4,2,1], [1,4,8,8,4,1], [1, 2, 4, 6, 8, 10], [1,8,8], [1,2,2], [1,4,4]],
                    "num_filters" : np.arange(74, 84 , 2),
                    "filter_size" : np.arange(68, 76, 2),
                    "num_layers" : [5,4],
                    "learning_rate" : [0.001, 0.00001],
                    "batch_size" : [256, 512],
                    "epochs" : [50],
                    "optimizer" : ["adam", "sgd"],
                    "dense_activation" : ["relu", "tanh"],
                    "padding" : ["same"],
                    "use_layerwise_dropout_batchnorm" : [True, False],
                    "dropout_T_bn_F" : [True, False],
                    
                    "dropout_rate" : [0.01,0.001],
                    "l2_r" : [0.01, 0.001, 0.0001],
                    "l1_r" : [0.01, 0.001,0],
                    "first_dense_units" : np.arange(250,258, 2),
                    "second_dense_units" : np.arange(230, 238, 2),
                    "output_layer_activation" : ["sigmoid"]
}

model_type = "CNN_grow_double"
is_lstm = True
num_channels = 3
beta = 3

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
start_from_scratch = True
use_reduced_lr = True
log_data = True
skip_to_index = 8
# Increase num filters. 82 is better performing than 78. But 80 is not better than 78
# 5 Layers perform better than less layers. Cost is of course training time is ridiculous.

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
                      skip_to_index = skip_to_index,
                      beta = beta)
narrowOpt.fit()
