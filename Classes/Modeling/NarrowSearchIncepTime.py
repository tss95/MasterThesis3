import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import mixed_precision

from sklearn.model_selection import ParameterGrid
from itertools import chain



from Classes.DataProcessing.LoadData import LoadData
from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.DataProcessing.DataHandler import DataHandler
from Classes.DataProcessing.RamGenerator import RamGenerator
from Classes.DataProcessing.RamLoader import RamLoader
from Classes.Modeling.InceptionTimeModel import InceptionTimeModel
from Classes.DataProcessing.NoiseAugmentor import NoiseAugmentor
from Classes.DataProcessing.TimeAugmentor import TimeAugmentor
from Classes.Scaling.MinMaxScalerFitter import MinMaxScalerFitter
from Classes.Scaling.StandardScalerFitter import StandardScalerFitter
from .GridSearchResultProcessor import GridSearchResultProcessor

import sys
import os


from livelossplot import PlotLossesKeras
import random
import pprint
import re
import json

base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3.0'

class NarrowSearchIncepTime(GridSearchResultProcessor):

    def __init__(self, loadData, train_ds, val_ds, detrend, use_scaler, use_time_augmentor, use_noise_augmentor,
                use_minmax, use_highpass, main_grid, hyper_grid, model_grid, use_tensorboard = False, 
                use_liveplots = False, use_custom_callback = False, use_early_stopping = False,
                highpass_freq = 0.1, start_from_scratch = True, use_reduced_lr = False, num_channels = 3):
    
        self.loadData = loadData
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.model_nr_type = "InceptionTime"
        
        
        self.num_classes = len(set(self.loadData.label_dict.values()))
        self.detrend = detrend
        self.use_scaler = use_scaler
        self.use_noise_augmentor = use_noise_augmentor
        self.use_time_augmentor = use_time_augmentor
        self.use_minmax = use_minmax
        self.use_highpass = use_highpass

        self.main_grid = main_grid
        self.hyper_grid = hyper_grid
        self.hyper_grid["num_channels"] = [num_channels]
        self.model_grid = model_grid
        
        self.use_tensorboard = use_tensorboard
        self.use_liveplots = use_liveplots
        self.use_custom_callback = use_custom_callback
        self.use_early_stopping = use_early_stopping
        self.use_reduced_lr = use_reduced_lr
        
        self.highpass_freq = highpass_freq
        self.start_from_scratch = start_from_scratch
        self.num_channels = num_channels

        print("Note that highpass and detrend have not been implemented yet in this class.")
        
        self.helper = HelperFunctions()
        self.handler = DataHandler(self.loadData)

        if self.loadData.earth_explo_only:
            self.full_ds = np.concatenate((self.loadData.noise_ds, self.loadData.full_ds))
        else:
            self.full_ds = self.loadData.full_ds



    def fit(self):
        self.timeAug, self.scaler, self.noiseAug = self.init_preprocessing(self.use_time_augmentor, 
                                                                           self.use_scaler, 
                                                                           self.use_noise_augmentor)
        self.results_file_name = self.get_results_file_name(narrow = True)
        self.hyper_picks, self.model_picks = self.create_search_space(self.main_grid, self.hyper_grid, self.model_grid)
        self.results_df = self.initiate_results_df(self.results_file_name, 
                                                    self.num_classes, 
                                                    self.start_from_scratch, 
                                                    self.hyper_picks[0], 
                                                    self.model_picks[0])
        pp = pprint.PrettyPrinter(indent=4)

        # Preprocessing and loading all data to RAM:
        ramLoader = RamLoader(self.handler, self.timeAug, self.scaler)
        self.x_train, self.y_train = ramLoader.load_to_ram(self.train_ds, False, self.num_channels)
        self.x_val, self.y_val = ramLoader.load_to_ram(self.val_ds, False, self.num_channels)  

        for i in range(len(self.hyper_picks)):
            tf.keras.backend.clear_session()
            mixed_precision.set_global_policy('mixed_float16')
            model_info = {"model_nr_type" : self.model_nr_type, "index" : i}
            print(f"Model nr {i + 1} of {len(self.hyper_picks)}")   
            # Translate picks to a more readable format:
            
            batch_size = self.hyper_picks[i]["batch_size"]
            epochs = self.hyper_picks[i]["epochs"]
            learning_rate = self.hyper_picks[i]["learning_rate"]
            opt = self.helper.get_optimizer(self.hyper_picks[i]["optimizer"], learning_rate)
            
            use_residuals = self.model_picks[i]["use_residuals"]
            use_bottleneck = self.model_picks[i]["use_bottleneck"]
            nr_modules = self.model_picks[i]["nr_modules"]
            kernel_size = self.model_picks[i]["kernel_size"]
            bottleneck_size = self.model_picks[i]["bottleneck_size"]
            num_filters =  self.model_picks[i]["num_filters"]
            shortcut_activation = self.model_picks[i]["shortcut_activation"]
            module_activation = self.model_picks[i]["module_activation"]
            module_output_activation = self.model_picks[i]["module_output_activation"]
            output_activation = self.model_picks[i]["output_activation"]

            reg_shortcut = self.model_picks[i]["reg_shortcut"]
            reg_module = self.model_picks[i]["reg_module"]
            l1_r = self.model_picks[i]["l1_r"]
            l2_r = self.model_picks[i]["l2_r"]
            
            current_picks = [model_info, self.hyper_picks[i], self.model_picks[i]]
            print(current_picks)
            # Store picked parameters:
            self.results_df = self.store_params_before_fit(current_picks, self.results_df, self.results_file_name)
            
            # Generate build model args using the picks from above.
            _, channels, timesteps = self.handler.get_trace_shape_no_cast(self.train_ds, self.use_time_augmentor)
            input_shape = (channels, timesteps)
            
            
            model_args = self.helper.generate_inceptionTime_build_args(input_shape, self.num_classes, opt,
                                                                      use_residuals, use_bottleneck, nr_modules,
                                                                      kernel_size, num_filters, bottleneck_size,
                                                                      shortcut_activation, module_activation,
                                                                      module_output_activation, output_activation,
                                                                      reg_shortcut, reg_module, l1_r, l2_r)
            # Build model using args generated above
            inceptionTime = InceptionTimeModel(**model_args)
            model = inceptionTime.build_model(input_shape, self.num_classes)

            # Initializing generators:
            gen = RamGenerator(self.loadData, self.handler, self.noiseAug)
            train_gen = gen.data_generator(self.x_train, self.y_train, batch_size)
            val_gen = gen.data_generator(self.x_val, self.y_val, batch_size)

            
            print("Starting: ")
            pp.pprint(self.hyper_picks[i])
            print("---------------------------------------------------------------------------------")
            pp.pprint(self.model_picks[i])

            # Generate fit args using picks.
            fit_args = self.helper.generate_fit_args(self.train_ds, self.val_ds, batch_size, 
                                                     epochs, val_gen, use_tensorboard = self.use_tensorboard, 
                                                     use_liveplots = self.use_liveplots, 
                                                     use_custom_callback = self.use_custom_callback,
                                                     use_early_stopping = self.use_early_stopping,
                                                     use_reduced_lr = self.use_reduced_lr)
            # Fit the model using the generated args
            try:
                model.fit(train_gen, **fit_args)
                
                # Evaluate the fitted model on the validation set
                loss, accuracy, precision, recall = model.evaluate_generator(generator=val_gen,
                                                                        steps=self.helper.get_steps_per_epoch(self.val_ds, batch_size))
                # Record metrics for train
                metrics = {}
                metrics['val'] = {  "val_loss" : loss,
                                    "val_accuracy" : accuracy,
                                    "val_precision": precision,
                                    "val_recall" : recall}
                current_picks.append(metrics['val'])
                
                # Evaluate the fitted model on the train set
                # Likely very redundant
                train_loss, train_accuracy, train_precision, train_recall = model.evaluate_generator(generator=train_gen,
                                                                                            steps=self.helper.get_steps_per_epoch(self.train_ds,
                                                                                                                                batch_size))
                metrics['train'] = { "train_loss" : train_loss,
                                    "train_accuracy" : train_accuracy,
                                    "train_precision": train_precision,
                                    "train_recall" : train_recall}
                current_picks.append(metrics['train'])
                self.results_df = self.store_metrics_after_fit(metrics, self.results_df, self.results_file_name)

            except Exception:
                print("Error (hopefully) occured during training.")
                continue
            
        min_loss, max_accuracy, max_precision, max_recall = self.find_best_performers(self.results_df)
        self.print_best_performers(min_loss, max_accuracy, max_precision, max_recall)
        return self.results_df, min_loss, max_accuracy, max_precision, max_recall
    



    def print_best_performers(self, min_loss, max_accuracy, max_precision, max_recall):
        print("----------------------------------------------------LOSS----------------------------------------------------------")
        print(f"Min val loss: {min_loss['val_loss']}, at index: {min_loss['val_index']}")
        print(f"Min training loss: {min_loss['train_loss']}, at index: {min_loss['train_index']}")
        print("----------------------------------------------------ACCURACY------------------------------------------------------")
        print(f"Highest val accuracy: {max_accuracy['val_accuracy']}, at index: {max_accuracy['val_index']}")
        print(f"Highest training accuracy: {max_accuracy['train_accuracy']}, at index: {max_accuracy['train_index']}")
        print("----------------------------------------------------PRECISION-----------------------------------------------------")
        print(f"Highest val precision: {max_precision['val_precision']}, at index: {max_precision['val_index']}")
        print(f"Highest training precision: {max_precision['train_precision']}, at index: {max_precision['train_index']}") 
        print("-----------------------------------------------------RECALL-------------------------------------------------------")
        print(f"Highest val recall: {max_recall['val_recall']}, at index: {max_recall['val_index']}")
        print(f"Highest training recall: {max_recall['train_recall']}, at index: {max_recall['train_index']}")
        print("------------------------------------------------------------------------------------------------------------------")
        
        
    def create_search_space(self, main_grid, hyper_grid, model_grid):
        hypermodel_grid = {**hyper_grid, **model_grid}

        key_list = list(main_grid.keys())
        np.random.shuffle(key_list)
        search_list = []
        for key in key_list:
            if len(hypermodel_grid[key]) > 1:
                one_model = main_grid.copy()
                one_model[key] = hypermodel_grid[key]
                key_grid = list(ParameterGrid(one_model))
                search_list.append(key_grid)
            else:
                continue
        search_list = list(chain.from_iterable(search_list))
        pprint.pprint(search_list)
        hyper_search, model_search = self.unmerge_search_space(search_list, hyper_grid, model_grid)
        return hyper_search, model_search
    
    def unmerge_search_space(self, search_space, hyper_grid, model_grid):
        hyper_keys = list(hyper_grid.keys())
        model_keys = list(model_grid.keys())
        hyper_search_grid = []
        model_search_grid = []
        for space in search_space:
            hyper_search_grid.append({key:value for (key,value) in space.items() if key in hyper_keys})
            model_search_grid.append({key:value for (key,value) in space.items() if key in model_keys})
        return hyper_search_grid, model_search_grid
    
    def init_preprocessing(self, use_time_augmentor, use_scaler, use_noise_augmentor):
        timeAug = None
        scaler = None
        noiseAug = None
        if use_time_augmentor:
            timeAug = TimeAugmentor(self.handler, self.full_ds, seed = self.loadData.seed)
            timeAug.fit()
        if use_scaler:
            if self.use_minmax:
                scaler = MinMaxScalerFitter(self.train_ds, timeAug).fit_scaler(detrend = self.detrend)
            else:
                scaler = StandardScalerFitter(self.train_ds, timeAug).fit_scaler(detrend = self.detrend)
        if use_noise_augmentor:
            noiseAug = NoiseAugmentor(self.loadData.noise_ds, use_scaler, scaler, self.loadData, timeAug)
        return timeAug, scaler, noiseAug
