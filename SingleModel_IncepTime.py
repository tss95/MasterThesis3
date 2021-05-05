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
from Classes.Modeling.DynamicModels import DynamicModels
from Classes.Modeling.TrainSingleModelRam import TrainSingleModelRam
from Classes.DataProcessing.RamLoader import RamLoader
from Classes.DataProcessing.ts_RamGenerator import modified_data_generator
import json

import gc

import datetime
import re
from livelossplot import PlotLossesKeras
from GlobalUtils import GlobalUtils
from Classes.Modeling.CustomCallback import CustomCallback
from tensorflow.keras.callbacks import EarlyStopping
utils = GlobalUtils()

from tensorflow.keras.utils import GeneratorEnqueuer






import sys


helper = HelperFunctions()

tf.config.optimizer.set_jit(True)
mixed_precision.set_global_policy('mixed_float16')

load_args = {
    'earth_explo_only' : False,
    'noise_earth_only' : False,
    'noise_not_noise' : True,
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



model_type = "InceptionTime"
is_lstm = True
num_channels = 3    
beta = 1

use_time_augmentor = True
scaler_name = "standard"
use_noise_augmentor = False
filter_name = None
band_min = 2.0
band_max = 4.0
highpass_freq = 0.075


use_tensorboard = True
use_liveplots = False
use_custom_callback = True
use_early_stopping = True
start_from_scratch = False
use_reduced_lr = True
log_data = True

shutdown = False

num_classes = len(list(set(loadData.label_dict.values())))

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


ramLoader = RamLoader(loadData, 
                      handler, 
                      use_time_augmentor = use_time_augmentor, 
                      use_noise_augmentor = use_noise_augmentor, 
                      scaler_name = scaler_name,
                      filter_name = filter_name, 
                      band_min = band_min,
                      band_max = band_max,
                      highpass_freq = highpass_freq, 
                      load_test_set = True, 
                      meier_load = False)

x_train, y_train, x_val, y_val, x_test, y_test, noiseAug = ramLoader.load_to_ram()


singleModel= TrainSingleModelRam(noiseAug, helper, loadData,
                                 model_type, num_channels, use_tensorboard,
                                 use_liveplots, use_custom_callback, 
                                 use_early_stopping, use_reduced_lr, ramLoader,
                                 log_data = log_data,
                                 start_from_scratch = start_from_scratch, 
                                 beta = beta)
params = {    
    "batch_size" : 512,
    "epochs" : 50,
    "learning_rate" : 0.01,
    "optimizer" : "rmsprop",
    "use_residuals" : True,
    "use_bottleneck" : False,
    "num_modules" : 1,
    "filter_size" : 40,
    "bottleneck_size" : 26,
    "num_filters" : 32,
    "residual_activation" : "relu",
    "module_activation" : "sigmoid",
    "module_output_activation" : "sigmoid",
    "output_layer_activation": "sigmoid",
    "reg_residual": True,
    "reg_module" : False,
    "l1_r" : 0,
    "l2_r" : 0.1
}

model = singleModel.run(x_train, y_train, x_val, y_val, x_test, y_test, 16, 15, 
                            evaluate_train = False, 
                            evaluate_val = False, 
                            evaluate_test = False, 
                            meier_load = False, 
                            index = None,
                            **params)
save_dir = '/media/tord/T7/Thesis_ssd/SavedModels/InceptionTime'
model_name = 'trained_3n_inceptionTime'
model_path = f'{save_dir}/{model_name}'

model.save(model_path)
#model = tf.keras.models.load_model(model_path)

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
del model

