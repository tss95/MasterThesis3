import numpy as np
import pandas as pd
import gc
import traceback

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
from Classes.DataProcessing.RamLessLoader import RamLessLoader
from Classes.Modeling.GridSearchResultProcessor import GridSearchResultProcessor
from Classes.Modeling.TrainSingleModelRam import TrainSingleModelRam
from Classes.Modeling.TrainSingleModelRamLess import TrainSingleModelRamLess

import random
import pprint
from itertools import chain


class NarrowOpt(GridSearchResultProcessor):
    def __init__(self, loadData, model_type, scaler_name, use_time_augmentor, use_noise_augmentor,
                filter_name, static_grid, search_grid, use_tensorboard = False, 
                use_liveplots = False, use_custom_callback = False, use_early_stopping = False, band_min = 2.0,
                band_max = 4.0, highpass_freq = 1, start_from_scratch = True, use_reduced_lr = False, num_channels = 3, 
                log_data = False, skip_to_index = 0, beta = 1, ramLess = False):
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
        self.log_data = log_data
        self.skip_to_index = skip_to_index
        self.beta = beta

        self.ramLess = ramLess

        
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

        if not self.ramLess:
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
            x_train, y_train, x_val, y_val, noiseAug = self.ramLoader.load_to_ram()


            singleModel = TrainSingleModelRam(noiseAug, self.helper, self.loadData,
                                            self.model_type, self.num_channels, self.use_tensorboard,
                                            self.use_liveplots, self.use_custom_callback, 
                                            self.use_early_stopping, self.use_reduced_lr, self.ramLoader,
                                            log_data = self.log_data,
                                            start_from_scratch = self.start_from_scratch, 
                                            beta = self.beta)

        else: 
            self.ramLessLoader = RamLessLoader(self.loadData, self.handler,
                                               use_time_augmentor = self.use_time_augmentor,
                                               use_noise_augmentor = self.use_time_augmentor,
                                               scaler_name = self.scaler_name,
                                               filter_name = self.filter_name,
                                               band_min = self.band_min,
                                               band_max = self.band_max,
                                               highpass_freq = self.highpass_freq,
                                               load_test_set = False,
                                               meier_load = False)
            self.ramLessLoader.fit()

            singleModel = TrainSingleModelRamLess(self.ramLessLoader, self.helper, self.loadData, self.model_type,
                                                  self.num_channels, self.use_tensorboard, self.use_liveplots,
                                                  self.use_custom_callback, self.use_early_stopping, self.use_reduced_lr,
                                                  log_data = self.log_data, start_from_scratch = self.start_from_scratch,
                                                  beta = self.beta)
        _, self.used_m, _ = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])

        print(f"Starting narrow optimizer. Will finish after training {len(self.p)} models.")
        for i in range(len(self.p)):
            if i < self.skip_to_index:
                continue
            gc.collect()
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            gc.collect()
            tf.config.optimizer.set_jit(True)
            mixed_precision.set_global_policy('mixed_float16')
            tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
            print(f"------------------RAM usage: {used_m}/{tot_m} (Free: {free_m})------------------")
            if not self.find_changed_key(self.static_grid, self.p[i]):
                continue
            if used_m > self.used_m:
                print("====================================================================================")
                print(f"POTENTIAL MEMORY LEAK ALERT!!! Previosuly max RAM usage was {self.used_m} now it is {used_m}. Leak size: {used_m - self.used_m}")
                print("====================================================================================")
                self.used_m = used_m
            try:
                # The hard defined variables in the run call refer to nr_workers and max_queue_size respectively.
                if not self.ramLess:
                    self.train_model(singleModel, x_train, y_train, x_val, y_val, i, **self.p[i])
                else:
                    self.train_model_ramless(singleModel, self.ramLessLoader, i, **self.p[i])
            except Exception:
                traceback.print_exc()
            finally:
                gc.collect()
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                gc.collect()
                print("After everything in the iteration:")
                tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
                print(f"------------------RAM usage: {used_m}/{tot_m} (Free: {free_m})------------------")
                continue

        
    def create_search_space(self, static_grid, search_grid):
        key_list = list(search_grid.keys())
        static_grid = self.bracket_dict(static_grid)
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
        search_list = self.remove_redundant_models(static_grid, search_list)
        return search_list

    def bracket_dict(self, d):
        for element in d:
            d[element] = [d[element]]
        return d

    def find_changed_key(self, static, current):
        key_list = list(static.keys())
        for key in key_list:
            if static[key][0] != current[key]:
                print("=================================================================")
                print(f"The current different parameter: {key}: {current[key]} ({static[key]})")
                print("=================================================================")
                return True
        else:
            print("=================================================================")
            print("Skipped as it is the same as the static model.")
            print("=================================================================")
            return False

    def remove_redundant_models(self, static, search):
        key_list = list(static.keys())
        search_mod = search
        for idx, candidate in enumerate(search):
            unique = False
            for key in key_list:
                if static[key][0] != candidate[key]:
                    unique = True
            if not unique:
                del search_mod[idx]
        return search_mod

    def train_model(self, singleModel, x_train, y_train, x_val, y_val, index, **p):
        _ = singleModel.run(x_train, y_train, x_val, y_val, None, None, 16, 15, 
                            evaluate_train = False, 
                            evaluate_val = False, 
                            evaluate_test = False, 
                            meier_load = False, 
                            index = index,
                            **p)

    def train_model_ramless(self, singleModel, ramLessLoader, index, **p):
        _ = singleModel.run(ramLessLoader, 16, 15, 
                            evaluate_train = False, 
                            evaluate_val = False, 
                            evaluate_test = False, 
                            meier_mode = False, 
                            index = index, 
                            **p)
                