import numpy as np
import pandas as pd
import gc

import sklearn as sk
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
from tensorflow.keras import mixed_precision

import os
base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3'
os.chdir(base_dir)

from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.DataProcessing.DataHandler import DataHandler
from Classes.DataProcessing.RamLoader import RamLoader
from Classes.Modeling.GridSearchResultProcessor import GridSearchResultProcessor
from Classes.Modeling.TrainSingleModel import TrainSingleModel

import random
import pprint
from itertools import chain


class NarrowOpt(GridSearchResultProcessor):
    def __init__(self, loadData, model_type, scaler_name, use_time_augmentor, use_noise_augmentor,
                filter_name, static_grid, search_grid, use_tensorboard = False, 
                use_liveplots = False, use_custom_callback = False, use_early_stopping = False, band_min = 2.0,
                band_max = 4.0, highpass_freq = 1, start_from_scratch = True, use_reduced_lr = False, num_channels = 3):
        self.loadData = loadData
        self.model_type = model_type
        
        
        self.num_classes = len(set(self.loadData.label_dict.values()))
        self.scaler_name = scaler_name
        self.use_noise_augmentor = use_noise_augmentor
        self.use_time_augmentor = use_time_augmentor
        self.filter_name = filter_name

        self.static_grid = static_grid
        self.search_grid = search_grid
        
        self.use_tensorboard = use_tensorboard
        self.use_liveplots = use_liveplots
        self.use_custom_callback = use_custom_callback
        self.use_early_stopping = use_early_stopping
        self.use_reduced_lr = use_reduced_lr
        
        self.highpass_freq = highpass_freq
        self.band_min = band_min
        self.band_max = band_max
        self.start_from_scratch = start_from_scratch
        self.num_channels = num_channels

        
        self.helper = HelperFunctions()
        self.handler = DataHandler(self.loadData)

    def fit(self):
        pp = pprint.PrettyPrinter(indent=4)
        # Creating grid:
        self.p = self.create_search_space(self.static_grid, self.search_grid)
        pp.pprint(self.p)
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
                print(str(e))
                continue
            finally:
                gc.collect()
                    
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                continue
        
    def create_search_space(self, static_grid, search_grid):
        key_list = list(static_grid.keys())
        search_list = []
        for key in key_list:
            if len(search_grid[key]) > 1:
                one_model = static_grid.copy()
                one_model[key] = search_grid[key]
                key_grid = list(ParameterGrid(one_model))
                search_list.append(key_grid)
            else:
                continue
        search_list = list(chain.from_iterable(search_list))
        pprint.pprint(search_list)
        return search_list