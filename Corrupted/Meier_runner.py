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
from Classes.Modeling.TrainSingleModel import TrainSingleModel
from Classes.DataProcessing.RamLoader import RamLoader
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


def run(scaler_name, earth_explo_only = False, noise_not_noise = False):
    load_args = {
        'earth_explo_only' : earth_explo_only,
        'noise_earth_only' : False,
        'noise_not_noise' : noise_not_noise,
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



    model_type = "Meier_CNN"
    is_lstm = True
    num_channels = 3
    beta = 2    

    use_time_augmentor = True
    scaler_name = scaler_name
    use_noise_augmentor = True
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



    epochs = 40
    batch_size = 48


    _,_ , timesteps = handler.get_trace_shape_no_cast(train_ds, use_time_augmentor)
    input_shape = (timesteps, num_channels)

    tf.config.optimizer.set_jit(True)
    mixed_precision.set_global_policy('mixed_float16')


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
                        meier_load = True)

    x_train, y_train, x_val, y_val, noiseAug = ramLoader.load_to_ram()






    modelTrain = TrainSingleModelRam(noiseAug, helper, loadData,
                                     model_type, num_channels, use_tensorboard,
                                     use_liveplots, use_custom_callback, 
                                     use_early_stopping, use_reduced_lr, ramLoader,
                                     log_data = log_data,
                                     start_from_scratch = False, 
                                     beta = beta)

    params = {
        "epochs" : epochs,
        "batch_size" : batch_size,
        "use_maxpool" : False,
        "use_averagepool" : False,
        "use_batchnorm" : False
    }

    model = modelTrain.run(x_train, y_train, x_val, y_val, None, None, 16, 15, meier_mode = True, **params)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    del model

    # =================================================================

    params = {
        "epochs" : epochs,
        "batch_size" : batch_size,
        "use_maxpool" : True,
        "use_averagepool" : False,
        "use_batchnorm" : False
    }

    model = modelTrain.run(x_train, y_train, x_val, y_val, None, None, 16, 15, meier_mode = True, **params)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    del model

    # =================================================================
    tf.config.optimizer.set_jit(True)
    mixed_precision.set_global_policy('mixed_float16')

    params = {
        "epochs" : epochs,
        "batch_size" : batch_size,
        "use_maxpool" : False,
        "use_averagepool" : True,
        "use_batchnorm" : False
    }

    model = modelTrain.run(x_train, y_train, x_val, y_val, None, None, 16, 15, meier_mode = True, **params)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    del model

    # =================================================================
    
    tf.config.optimizer.set_jit(True)
    mixed_precision.set_global_policy('mixed_float16')

    params = {
        "epochs" : epochs,
        "batch_size" : batch_size,
        "use_maxpool" : False,
        "use_averagepool" : False,
        "use_batchnorm" : True
    }

    model = modelTrain.run(x_train, y_train, x_val, y_val, None, None, 16, 15, meier_mode = True, **params)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    del model


    # =================================================================
    tf.config.optimizer.set_jit(True)
    mixed_precision.set_global_policy('mixed_float16')

    params = {
        "epochs" : epochs,
        "batch_size" : batch_size,
        "use_maxpool" : True,
        "use_averagepool" : False,
        "use_batchnorm" : True
    }

    model = modelTrain.run(x_train, y_train, x_val, y_val, None, None, 16, 15, meier_mode = True, **params)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    del model

    # =================================================================
    tf.config.optimizer.set_jit(True)
    mixed_precision.set_global_policy('mixed_float16')

    params = {
        "epochs" : epochs,
        "batch_size" : batch_size,
        "use_maxpool" : False,
        "use_averagepool" : True,
        "use_batchnorm" : True
    }

    model = modelTrain.run(x_train, y_train, x_val, y_val, None, None, 16, 15, meier_mode = True, **params)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    del model

if __name__ == "__main__":
    run("standard", noise_not_noise = True)
    gc.collect()
    run("robust", noise_not_noise = True)
    gc.collect()
    run("minmax", noise_not_noise = True)
    gc.collect()
    run("standard", earth_explo_only = True)
    gc.collect()
    run("robust", earth_explo_only = True)
    gc.collect()
    run("minmax", earth_explo_only = True)
    gc.collect()
    run("normalize", earth_explo_only = True)
    gc.collect()
