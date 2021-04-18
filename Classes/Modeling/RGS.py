import numpy as np
import pandas as pd
import h5py
import gc
import traceback

import sklearn as sk

from sklearn.model_selection import ParameterGrid




import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import GeneratorEnqueuer

import os
base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3'
os.chdir(base_dir)

from Classes.Modeling.DynamicModels import DynamicModels
from Classes.Modeling.StaticModels import StaticModels
from Classes.DataProcessing.LoadData import LoadData
from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.DataProcessing.DataHandler import DataHandler
from Classes.DataProcessing.RamLoader import RamLoader
from Classes.Modeling.GridSearchResultProcessor import GridSearchResultProcessor
from Classes.DataProcessing.ts_RamGenerator import data_generator
from Classes.Modeling.TrainSingleModel import TrainSingleModel


import sys


import random
import pprint
import re
import json

class RGS(GridSearchResultProcessor):
    
    def __init__(self, loadData, train_ds, val_ds, test_ds, model_type, scaler_name, use_time_augmentor, use_noise_augmentor,
                 filter_name, n_picks, hyper_grid, use_tensorboard = False, 
                 use_liveplots = True, use_custom_callback = False, use_early_stopping = False, use_reduced_lr = False,
                 band_min = 2.0, band_max = 4.0, highpass_freq = 0.1, start_from_scratch = True, is_lstm = False, 
                 log_data = True, num_channels = 3):
        
        self.loadData = loadData
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.model_type = model_type
        self.num_classes = len(set(self.loadData.label_dict.values()))

        self.scaler_name = scaler_name
        self.use_noise_augmentor = use_noise_augmentor
        self.use_time_augmentor = use_time_augmentor
        self.filter_name = filter_name
        self.n_picks = n_picks
        self.hyper_grid = hyper_grid
        self.use_tensorboard = use_tensorboard
        self.use_liveplots = use_liveplots
        self.use_custom_callback = use_custom_callback
        self.use_reduced_lr = use_reduced_lr
        self.use_early_stopping = use_early_stopping

        self.band_min = band_min
        self.band_max = band_max
        self.highpass_freq = highpass_freq
        self.start_from_scratch = start_from_scratch
        self.is_lstm = is_lstm
        self.log_data = log_data
        self.num_channels = num_channels
        
        self.helper = HelperFunctions()
        self.handler = DataHandler(self.loadData)
        self.is_dynamic = False
        if type(self.model_type) == str:
            self.is_dynamic = True


    def fit(self):
        pp = pprint.PrettyPrinter(indent=4)
        # Creating grid:
        self.p = ParameterGrid(self.hyper_grid)
        if len(self.p) < self.n_picks:
            self.n_picks = len(self.p)
            print(f"Picks higher than max. Reducing picks to {self.n_picks} picks")
        self.p = self.get_n_params_from_list(self.p, self.n_picks)
        
        if self.start_from_scratch:
            print("================================================================================================================================================")
            print("================================ YOU WILL BE PROMPTED AFTER DATA HAS BEEN LOADED TO CONFIRM CLEARING OF DATASET ================================")
            print("================================================================================================================================================")
        # Preprocessing and loading all data to RAM:
        self.ramLoader = RamLoader(self.loadData, 
                              self.handler, 
                              use_time_augmentor = self.use_time_augmentor, 
                              use_noise_augmentor = self.use_noise_augmentor, 
                              scaler_name = self.scaler_name,
                              filter_name = self.filter_name, 
                              band_min = self.band_min,
                              band_max = self.band_max,
                              highpass_freq = self.highpass_freq, 
                              load_test_set = False)
        self.x_train, self.y_train, self.x_val, self.y_val, self.noiseAug = self.ramLoader.load_to_ram()

        # Create name of results file, get initiated results df, either brand new or continue old.
        self.results_file_name = self.get_results_file_name()
        print(self.results_file_name)
        self.results_df = self.initiate_results_df_opti(self.results_file_name, self.num_classes, self.start_from_scratch, self.p[0])
        print(self.results_df)
        for i in range(len(self.p)):
            gc.collect()
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            tf.config.optimizer.set_jit(True)
            mixed_precision.set_global_policy('mixed_float16')
            try:
                trainSingleModel = TrainSingleModel(self.x_train, self.y_train, self.x_val, self.y_val,
                                                    None, None, self.noiseAug, self.helper, self.loadData,
                                                    self.model_type, self.num_channels, self.use_tensorboard,
                                                    self.use_liveplots, self.use_custom_callback, 
                                                    self.use_early_stopping, self.use_reduced_lr, self.ramLoader,
                                                    log_data = self.log_data, results_df = self.results_df,
                                                    results_file_name = self.results_file_name, index = i,
                                                    start_from_scratch = False)
                # Add try catch clauses here
                model, self.results_df = trainSingleModel.run(16, 15, evaluate_train = False, evaluate_val = False, evaluate_test = False, meier_load = False, **self.p[i])
                del model
            except Exception as e:
                traceback.print_exc()
                continue
            finally:
                gc.collect()
                    
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                continue

    
    def get_n_params_from_list(self, grid, n_picks):
        print(f"Length of grid: {len(grid)}")
        indexes = random.sample(range(0, len(grid)), n_picks)
        picks = [grid[idx] for idx in indexes]
        return picks