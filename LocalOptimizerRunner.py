import numpy as np
import os
import sys
import pandas as pd

import sklearn as sk

import pylab as pl
import h5py

import tensorflow as tf

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
from Classes.Modeling.GridSearchResultProcessor import GridSearchResultProcessor
from Classes.Modeling.LocalOptimizer import LocalOptimizer
from Classes.Modeling.LocalOptimizerIncepTime import LocalOptimizerIncepTime


helper = HelperFunctions()

import sys
ISCOLAB = 'google.colab' in sys.modules

import random
import pprint


load_args = {
    'earth_explo_only' : True,
    'noise_earth_only' : False,
    'noise_not_noise' : False,
    'downsample' : True,
    'upsample' : True,
    'frac_diff' : 1,
    'seed' : 1,
    'subsample_size' : 0.45,
    'balance_non_train_set' : True,
    'use_true_test_set' : False,
    'even_balance' : True
}
loadData = LoadData(**load_args)
full_ds, train_ds, val_ds, test_ds = loadData.get_datasets()
noise_ds = loadData.noise_ds
handler = DataHandler(loadData)


# PARAMS 
num_channels = 3

use_time_augmentor = True
scaler_name = "standard"
use_noise_augmentor = True
filter_name = None
band_min = 2
band_max = 4
highpass_freq = 0.1

use_tensorboard = True
use_liveplots = False
use_custom_callback = False
use_early_stopping = True
start_from_scratch = False
use_reduced_lr = True

result_file_name = 'results_InceptionTime_earthExplo_timeAug_sscale_noiseAug_earlyS.csv'
quick_mode = False
continue_from_result_file = True
start_grid = None


# The higher this value is, the more focused the best model will be on the initial metric
metric_gap = 0.4

# Number of models to consider:
nr_candidates = 20

# If cancelled prior to completion, write the number in which it was cancelled in order to pick up where you left off.
skip_to_index = 0

# Only False if testing
log_data = True

# Not used:
depth = 5



narrowOpt = LocalOptimizerIncepTime(loadData = loadData, 
                                    scaler_name = scaler_name, 
                                    use_time_augmentor = use_time_augmentor, 
                                    use_noise_augmentor = use_noise_augmentor, 
                                    filter_name = filter_name,
                                    use_tensorboard = use_tensorboard, 
                                    use_liveplots = use_liveplots, 
                                    use_custom_callback = use_custom_callback, 
                                    use_early_stopping = use_early_stopping, 
                                    band_min = band_min, 
                                    band_max = band_max, 
                                    highpass_freq = highpass_freq, 
                                    use_reduced_lr = use_reduced_lr, 
                                    num_channels = num_channels, 
                                    depth = depth, 
                                    quick_mode = quick_mode, 
                                    continue_from_result_file = continue_from_result_file, 
                                    result_file_name = result_file_name, 
                                    start_grid = start_grid)

#top_10 = narrowOpt.get_best_model(result_file_name, 2, optimize_metric = ['val_accuracy', 'val_f1'], nr_candidates = 10)
try:

  narrowOpt.run(['val_accuracy', 'val_f1'], 
                nr_candidates = nr_candidates, 
                metric_gap = metric_gap, 
                log_data = log_data, 
                skip_to_index = skip_to_index)
finally:
  #os.shutdown('shutdown -s')
  print("Im done bitch")