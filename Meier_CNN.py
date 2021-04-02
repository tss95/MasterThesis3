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
from Classes.DataProcessing.RamLoader import RamLoader
from Classes.DataProcessing.ts_RamGenerator import modified_data_generator
import json

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
    'subsample_size' : 0.2,
    'balance_non_train_set' : True,
    'use_true_test_set' : False,
    'even_balance' : True
}
loadData = LoadData(**load_args)
train_ds, val_ds, test_ds = loadData.get_datasets()
noise_ds = loadData.noise_ds
handler = DataHandler(loadData)



model_type = "Meier_CNN"
is_lstm = True
num_channels = 3    

use_time_augmentor = True
scaler_name = None
use_noise_augmentor = True
filter_name = "highpass"
band_min = 2.0
band_max = 4.0
highpass_freq = 0.075


use_tensorboard = True
use_liveplots = False
use_custom_callback = True
use_early_stopping = False
start_from_scratch = False
use_reduced_lr = False
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

def generate_meier_fit_args(train_ds, val_ds, helper, batch_size, epoch, val_gen, use_tensorboard, use_liveplots, use_custom_callback, use_early_stopping, use_reduced_lr = False):
    callbacks = []
    if use_liveplots:
        print("")
        #callbacks.append(PlotLossesKeras())
    if use_tensorboard:
        log_dir = f"{utils.base_dir}/Tensorboard_dir/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)
    if use_custom_callback:
        custom_callback = CustomCallback()
        callbacks.append(custom_callback)
    if use_early_stopping:
        earlystop = EarlyStopping(monitor = 'val_categorical_accuracy',
                    min_delta = 0,
                    patience = 5,
                    verbose = 1,
                    restore_best_weights = True)
        callbacks.append(earlystop)
    
    if use_reduced_lr:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', 
                                                            factor=0.5, patience=3,
                                                            min_lr=0.00005, 
                                                            verbose = 1))
    return {"steps_per_epoch" : helper.get_steps_per_epoch(train_ds, batch_size),
            "epochs" : epoch,
            "validation_data" : val_gen,
            "validation_steps" : helper.get_steps_per_epoch(val_ds, batch_size),
            "verbose" : 1,
            "max_queue_size" : 10,
            "use_multiprocessing" : False, 
            "workers" : 1,
            "callbacks" : callbacks
            }

epochs = 40
batch_size = 48


_,_ , timesteps = handler.get_trace_shape_no_cast(train_ds, use_time_augmentor)
input_shape = (timesteps, num_channels)




ramLoader = RamLoader(loadData, 
                      handler, 
                      use_time_augmentor = use_time_augmentor, 
                      use_noise_augmentor = use_noise_augmentor, 
                      scaler_name = scaler_name,
                      filter_name = filter_name, 
                      band_min = band_min,
                      band_max = band_max,
                      highpass_freq = highpass_freq, 
                      load_test_set = False, 
                      meier_load = True)

x_train, y_train, x_val, y_val, noiseAug = ramLoader.load_to_ram()

train_enq = GeneratorEnqueuer(modified_data_generator(x_train, y_train, batch_size, noiseAug, num_channels = num_channels, is_lstm  = is_lstm), use_multiprocessing = False)
val_enq = GeneratorEnqueuer(modified_data_generator(x_val, y_val,batch_size, noiseAug, num_channels = num_channels, is_lstm  = is_lstm), use_multiprocessing = False)
train_enq.start(workers = 16, max_queue_size = 15)
val_enq.start(workers = 16, max_queue_size = 15)
train_gen = train_enq.get()
val_gen = train_enq.get()

fit_args = generate_meier_fit_args(train_ds, val_ds, helper, batch_size, val_gen, use_tensorboard, use_liveplots, use_custom_callback, use_early_stopping, use_reduced_lr)

params = {
    "use_maxpool" : False,
    "use_avgpool" : False,
    "use_batchnorm" : False
}

model = DynamicModels(model_type, len(set(loadData.label_dict.values())), input_shape, params)


model.fit(train_gen, **fit_args)

conf, _ = helper.evaluate_model(model, x_val, y_val, loadData.label_dict, num_channels = num_channels, plot = True, run_evaluate = True)

