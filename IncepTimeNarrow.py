import numpy as np
import os
import sys
import pandas as pd

import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

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

base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3.0'
os.chdir(base_dir)

from Classes.DataProcessing.LoadData import LoadData
from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.DataProcessing.DataHandler import DataHandler
from Classes.DataProcessing.TimeAugmentor import TimeAugmentor
from Classes.DataProcessing.NoiseAugmentor import NoiseAugmentor
from Classes.DataProcessing.RamLoader import RamLoader
from Classes.DataProcessing.RamGenerator import RamGenerator
from Classes.Modeling.InceptionTimeModel import InceptionTimeModel
from Classes.Modeling.NarrowSearchIncepTime import NarrowSearchIncepTime
from Classes.Modeling.CustomCallback import CustomCallback
from Classes.Modeling.ResultFitter import ResultFitter
from Classes.Scaling.ScalerFitter import ScalerFitter
from Classes.Scaling.MinMaxScalerFitter import MinMaxScalerFitter
from Classes.Scaling.StandardScalerFitter import StandardScalerFitter
from Classes.Modeling.GridSearchResultProcessor import GridSearchResultProcessor
import json
#from Classes import Tf_shutup
#Tf_shutup.Tf_shutup()

from livelossplot import PlotLossesKeras



from matplotlib.colors import ListedColormap

plt.rcParams["figure.figsize"]= (15,15)
helper = HelperFunctions()

import sys
ISCOLAB = 'google.colab' in sys.modules

import random
import pprint

load_args = {
    'earth_explo_only' : False,
    'noise_earth_only' : False,
    'noise_not_noise' : True,
    'downsample' : True,
    'upsample' : True,
    'frac_diff' : 1,
    'seed' : 1,
    'subsample_size' : 0.4,
    'balance_non_train_set' : True,
    'use_true_test_set' : False,
    'even_balance' : True
}
loadData = LoadData(**load_args)
full_ds, train_ds, val_ds, test_ds = loadData.get_datasets()
noise_ds = loadData.noise_ds
handler = DataHandler(loadData)

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


main_grid = {
    "batch_size" : [512],
    "epochs" : [100],
    "learning_rate" : [0.01],
    "optimizer" : ["rmsprop"],
    "use_residuals" : [True],
    "use_bottleneck" : [False],
    "nr_modules" : [1],
    "kernel_size" : [40],
    "bottleneck_size" : [26],
    "num_filters" : [32],
    "shortcut_activation" : ["tanh"],
    "module_activation" : ["sigmoid"],
    "module_output_activation" : ["tanh"],
    "output_activation": ["sigmoid"],
    "reg_module" : [False],
    "reg_shortcut" : [True],
    "l1_r" : [0.01],
    "l2_r" : [0.01]
    }

hyper_grid = {
    "batch_size" : [64, 128, 512, 768, 1024],
    "epochs" : [100],
    "learning_rate" : [0.1,0.001, 0.0005],
    "optimizer" : ["rmsprop", "sgd"]
    }
model_grid = {
    "use_residuals" : [False, True],
    "use_bottleneck" : [True, False],
    "nr_modules" : [3, 6, 9, 14],
    "kernel_size" : [10, 30, 50],
    "bottleneck_size" : [22],
    "num_filters" : [28, 30, 34, 36, 40, 44, 48],
    "shortcut_activation" : ["relu", "sigmoid"],
    "module_activation" : ["linear", "relu", "softmax", "tanh"],
    "module_output_activation" : ["relu", "linear", "sigmoid", "softmax"],
    "output_activation": ["sigmoid"],
    "reg_module" : [True, True],
    "reg_shortcut" : [False, False],
    "l1_r" : [0.1, 0.001, 0.0001, 0.0],
    "l2_r" : [0.1, 0.001, 0.0001, 0.0]
}


num_channels = 3

use_time_augmentor = True
use_scaler = True
use_noise_augmentor = True
detrend = False
use_minmax = False
use_highpass = False
highpass_freq = 0.1

use_tensorboard = True
use_liveplots = False
use_custom_callback = False
use_early_stopping = True
start_from_scratch = False
use_reduced_lr = True

narrowSearch = NarrowSearchIncepTime(loadData, train_ds, val_ds, detrend, 
                               use_scaler,  use_time_augmentor, use_noise_augmentor, 
                               use_minmax,use_highpass, main_grid, hyper_grid, model_grid, 
                               use_tensorboard = use_tensorboard, use_liveplots = use_liveplots, 
                               use_custom_callback = use_custom_callback, 
                               use_early_stopping = use_early_stopping, highpass_freq = highpass_freq,
                               start_from_scratch = start_from_scratch, num_channels = num_channels)

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
    #%tensorboard --logdir tensorboard_dir/fit

results_df, min_loss, max_accuracy, max_precision, max_recall = narrowSearch.fit()