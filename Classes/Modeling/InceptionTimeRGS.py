import numpy as np
import pandas as pd
import h5py

import sklearn as sk

from sklearn.model_selection import ParameterGrid




import tensorflow as tf
from tensorflow.keras import mixed_precision

import os
base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3'
os.chdir(base_dir)

from Classes.Modeling.InceptionTimeModel import InceptionTimeModel
from Classes.DataProcessing.LoadData import LoadData
from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.DataProcessing.DataHandler import DataHandler
from Classes.DataProcessing.RamGenerator import RamGenerator
from Classes.DataProcessing.RamLoader import RamLoader
from Classes.DataProcessing.NoiseAugmentor import NoiseAugmentor
from Classes.DataProcessing.TimeAugmentor import TimeAugmentor
from Classes.DataProcessing.DataGenerator import DataGenerator
from Classes.Scaling.MinMaxScalerFitter import MinMaxScalerFitter
from Classes.Scaling.StandardScalerFitter import StandardScalerFitter
from .GridSearchResultProcessor import GridSearchResultProcessor

import sys


import random
import pprint
import re
import json


class InceptionTimeRGS(GridSearchResultProcessor):
    
    hyper_grid = {    
        "batch_size" : [128, 256, 512, 1024],
        "epochs" : [100, 100, 100],
        "learning_rate" : [0.1, 0.01, 0.001, 0.0001, 0.00001],
        "optimizer" : ["adam", "rmsprop", "sgd"]
        }
    model_grid = {
        "use_residuals" : [True, False],
        "use_bottleneck" : [True, False],
        "nr_modules" : [1, 3, 6, 9, 12],
        "kernel_size" : [20, 30, 40, 50],
        "bottleneck_size" : [ 26, 28, 30, 32, 34, 36],
        "num_filters" : [24, 26, 28, 30, 32, 34, 36, 38, 40 ,42],
        "shortcut_activation" : ["relu", "sigmoid", "softmax", "tanh"],
        "module_activation" : ["linear", "relu", "sigmoid", "softmax", "tanh"],
        "module_output_activation" : ["relu", "linear", "sigmoid", "softmax", "tanh"],
        "output_activation": ["sigmoid"],
        "reg_shortcut": [True, False],
        "reg_module" : [True, False],
        "l1_r" : [0.1, 0.001, 0.0001, 0],
        "l2_r" : [0.1, 0.001, 0.0001, 0]
    }
    

    def __init__(self, loadData, train_ds, val_ds, scaler_name, use_time_augmentor, use_noise_augmentor,
                 filter_name, n_picks, hyper_grid=hyper_grid, model_grid=model_grid, use_tensorboard = False, 
                 use_liveplots = False, use_custom_callback = False, use_early_stopping = False, band_min = 2.0, band_max = 4.0,
                 highpass_freq = 0.1, start_from_scratch = True, use_reduced_lr = False, num_channels = 3, log_data = True):
        
        self.loadData = loadData
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.model_nr_type = "InceptionTime"
        
        
        self.num_classes = len(set(self.loadData.label_dict.values()))
        self.scaler_name = scaler_name
        self.use_noise_augmentor = use_noise_augmentor
        self.use_time_augmentor = use_time_augmentor
        self.filter_name = filter_name
        
        self.n_picks = n_picks
        self.hyper_grid = hyper_grid
        self.hyper_grid["num_channels"] = [num_channels]
        self.model_grid = model_grid
        
        self.use_tensorboard = use_tensorboard
        self.use_liveplots = use_liveplots
        self.use_custom_callback = use_custom_callback
        self.use_early_stopping = use_early_stopping
        self.use_reduced_lr = use_reduced_lr
        
        self.band_min = band_min
        self.band_max = band_max
        self.highpass_freq = highpass_freq
        self.start_from_scratch = start_from_scratch
        self.num_channels = num_channels
        
        self.helper = HelperFunctions()
        self.handler = DataHandler(self.loadData)
        self.log_data = log_data

        if self.loadData.earth_explo_only:
            self.full_ds = np.concatenate((self.loadData.noise_ds, self.loadData.full_ds))
        else:
            self.full_ds = self.loadData.full_ds

        print("Initialized num channels: " + str(self.num_channels))


    
    def fit(self):
        # Creating grid:
        self.model_params = ParameterGrid(self.model_grid)
        self.hyper_params = ParameterGrid(self.hyper_grid)
        if len(self.model_params) < self.n_picks or len(self.hyper_params) < self.n_picks:
            self.n_picks = min(len(self.model_params), len(self.hyper_params))
            print(f"Picks higher than max. Reducing picks to {self.n_picks} picks")

        self.hyper_picks = self.get_n_params_from_list(self.hyper_params, self.n_picks)
        self.model_picks = self.get_n_params_from_list(self.model_params, self.n_picks)
        
        
        
        # Create name of results file, get initiated results df, either brand new or continue old.
        self.results_file_name = self.get_results_file_name()
        
        self.results_df = self.initiate_results_df(self.results_file_name, self.num_classes, self.start_from_scratch, self.hyper_picks[0], self.model_picks[0])
        
        # Preprocessing and loading all data to RAM:
        ramLoader = RamLoader(self.loadData, 
                              self.handler, 
                              use_time_augmentor = self.use_time_augmentor, 
                              use_noise_augmentor = self.use_noise_augmentor, 
                              scaler_name = self.scaler_name,
                              filter_name = self.filter_name, 
                              band_min = self.band_min,
                              band_max = self.band_max,
                              highpass_freq = self.highpass_freq, 
                              load_test_set = False)
        self.x_train, self.y_train, self.x_val, self.y_val, self.timeAug, self.scaler, self.noiseAug = ramLoader.load_to_ram()

        pp = pprint.PrettyPrinter(indent=4)
        

        for i in range(len(self.hyper_picks)):
            tf.keras.backend.clear_session()
            tf.config.optimizer.set_jit(True)
            mixed_precision.set_global_policy('mixed_float16')
            
            model_info = {"model_nr_type" : self.model_nr_type, "index" : i}
            print(f"Model nr {i + 1} of {len(self.hyper_picks)}")   
            print(i+1)
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
            if self.log_data:
                # Store picked parameters:
                self.results_df = self.store_params_before_fit(current_picks, self.results_df, self.results_file_name)
            
            # Generate build model args using the picks from above.
            num_ds, channels, timesteps = self.handler.get_trace_shape_no_cast(self.train_ds, self.use_time_augmentor)
            input_shape = (timesteps, self.num_channels)

            
            compile_args = self.helper.generate_model_compile_args(opt, self.num_classes)
            
            model_args = self.helper.generate_inceptionTime_build_args(input_shape, self.num_classes, opt,
                                                                      use_residuals, use_bottleneck, nr_modules,
                                                                      kernel_size, num_filters, bottleneck_size,
                                                                      shortcut_activation, module_activation,
                                                                      module_output_activation, output_activation,
                                                                      reg_shortcut, reg_module, l1_r, l2_r)

            print("Build args input shape: " + str(model_args['input_shape']))
            # Build model using args generated above
            inceptionTime = InceptionTimeModel(**model_args)
            model = inceptionTime.build_model(input_shape, self.num_classes)

            print(model.summary())

            # Initializing generators:
            gen = RamGenerator(self.loadData, self.handler, self.noiseAug)
            train_gen = gen.data_generator(self.x_train, self.y_train, batch_size, num_channels = self.num_channels, is_lstm = True)
            val_gen = gen.data_generator(self.x_val, self.y_val, batch_size, num_channels = self.num_channels, is_lstm = True)

            
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
            try:
                # Fit the model using the generated args
                model_fit = model.fit(train_gen, **fit_args)
                
                # Evaluate the fitted model on the validation set
                loss, accuracy, precision, recall = model.evaluate(x=val_gen,
                                                                        steps=self.helper.get_steps_per_epoch(self.val_ds, batch_size))
                # Record metrics for train
                metrics = {}
                metrics['val'] = {  "val_loss" : loss,
                                    "val_accuracy" : accuracy,
                                    "val_precision": precision,
                                    "val_recall" : recall}
                current_picks.append(metrics['val'])
                
                # Evaluate the fitted model on the train set
                train_loss, train_accuracy, train_precision, train_recall = model.evaluate(x = train_gen,
                                                                                            steps=self.helper.get_steps_per_epoch(self.train_ds,
                                                                                                                                batch_size))
                metrics['train'] = { "train_loss" : train_loss,
                                    "train_accuracy" : train_accuracy,
                                    "train_precision": train_precision,
                                    "train_recall" : train_recall}
                current_picks.append(metrics['train'])
                if self.log_data:
                    self.results_df = self.store_metrics_after_fit(metrics, self.results_df, self.results_file_name)
            except Exception as e:
                print(e)
                print("Something went wrong while training the model (most likely)")
                continue
            
        min_loss, max_accuracy, max_precision, max_recall = self.find_best_performers(self.results_df)
        self.print_best_performers(min_loss, max_accuracy, max_precision, max_recall)
        return self.results_df, min_loss, max_accuracy, max_precision, max_recall
        

    
    def get_n_params_from_list(self, grid, n_picks):
        print(f"Length of grid: {len(grid)}")
        indexes = random.sample(range(0, len(grid)), n_picks)
        picks = [grid[idx] for idx in indexes]
        return picks      

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
       