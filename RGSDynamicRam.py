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
from Classes.Modeling.RandomGridSearchDynamic import RandomGridSearchDynamic
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
    'balance_non_train_set' : True,
    'use_true_test_set' : False,
    'even_balance' : True
}
loadData = LoadData(**load_args)
full_ds, train_ds, val_ds, test_ds = loadData.get_datasets()
noise_ds = loadData.noise_ds
handler = DataHandler(loadData)

# Printing data stats:
print(len(train_ds), len(val_ds), len(test_ds))
classes, counts = handler.get_class_distribution_from_ds(train_ds)
classes, counts = handler.get_class_distribution_from_ds(val_ds)
print("Nr noise samples " + str(len(loadData.noise_ds)))
print(f"Non noise prop: {len(full_ds[full_ds[:,1] != 'noise'])/len(full_ds)}")
print(f"Train non noise prop: {len(train_ds[train_ds[:,1] != 'noise'])/len(train_ds)}")
print(f"Val non noise prop: {len(val_ds[val_ds[:,1] != 'noise'])/len(val_ds)}")



#Grid used for LSTM models.

#- Consider editing the decay_sequences.

hyper_grid = {
        #num_layers" : [1, 2, 3, 4, 5, 6],
        "num_layers" : [1, 2, 3, 4, 5],
        "batch_size" : [64, 128, 256],
        "epochs" : [50, 50, 50, 50, 50, 50, 50, 50, 50],
        "learning_rate" : [0.1, 0.01, 0.01, 0.001, 0.001, 0.0001, 0.0001],
        "optimizer" : ["sgd", "sgd", "sgd", "sgd", "rmsprop", "adam", "rmsprop", "sgd"]
    }
model_grid = {
    #"start_neurons" : [20, 25, 30, 35],
    "start_neurons" : np.arange(100, 300, 10),
    "use_layerwise_dropout_batchnorm" : [False, True],
    "decay_sequence" : [[1,2,4,4,2,1], [1,4,8,8,4,1], [1,1,1,1,1,1], [1, 2, 4, 6, 8, 10]],
    "dropout_rate" : [0.3, 0.2, 0.1, 0.01, 0.001, 0],
    "filters" : np.arange(10, 80, 2),
    "kernel_size" : np.arange(3, 100, 3),
    "padding" : ["same"],
    "l2_r" : [0.3, 0.2, 0.1, 0.01, 0.001, 0.0001],
    "l1_r" : [0.3, 0.2, 0.1, 0.01, 0.001, 0.0001],
    "activation" : ["tanh", "relu"],
    #"activation" : ["tanh", "tanh", "relu", "relu", "relu", "sigmoid", "softmax"],
    "output_layer_activation" : ["sigmoid"]
}



model_type = "CNN"
is_lstm = True
num_channels = 2    

use_time_augmentor = True
scaler_name = "standard"
use_noise_augmentor = True
filter_name = None
band_min = 2.0
band_max = 4.0
highpass_freq = 15

n_picks = 100

use_tensorboard = True
use_liveplots = False
use_custom_callback = True
use_early_stopping = True
start_from_scratch = False
use_reduced_lr = True
log_data = True

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


try:
    randomGridSearch = RandomGridSearchDynamic(loadData, train_ds, val_ds, test_ds, model_type, scaler_name, use_time_augmentor, use_noise_augmentor,
                                                filter_name, n_picks, hyper_grid=hyper_grid, model_grid=model_grid, use_tensorboard = use_tensorboard, 
                                                use_liveplots = use_liveplots, use_custom_callback = use_custom_callback, use_early_stopping = use_early_stopping, 
                                                use_reduced_lr = use_reduced_lr, band_min = band_min, band_max = band_max, highpass_freq = highpass_freq, 
                                                start_from_scratch = start_from_scratch, is_lstm = is_lstm, log_data = log_data, num_channels = num_channels)
    results_df, min_loss, max_accuracy, max_precision, max_recall = randomGridSearch.fit()

except Exception as e:
    print(str(e))

finally:
    if shutdown:
        os.system("shutdown now -h")
    else:
        print("Process completed.")
            


