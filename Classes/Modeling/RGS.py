import numpy as np
import pandas as pd
import h5py
import gc

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
        self.model_nr_type = model_type
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
        if type(self.model_nr_type) == str:
            self.is_dynamic = True

            

    def fit(self):
        pp = pprint.PrettyPrinter(indent=4)
        # Creating grid:
        self.p = ParameterGrid(self.hyper_grid)
        if len(self.p) < self.n_picks:
            self.n_picks = len(self.p)
            print(f"Picks higher than max. Reducing picks to {self.n_picks} picks")
        self.p = self.get_n_params_from_list(self.p, self.n_picks)
        
        # Create name of results file, get initiated results df, either brand new or continue old.
        self.results_file_name = self.get_results_file_name()
        print(self.results_file_name)
        if self.start_from_scratch:
            confirmation = input("Are you sure you want to erase the results file? Y/N \n").upper()
            if confirmation != "Y":
                print("Terminating process.")
                return
        self.results_df = self.initiate_results_df_opti(self.results_file_name, self.num_classes, self.start_from_scratch, self.p[0])
        print(self.results_df)
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
        self.x_train, self.y_train, self.x_val, self.y_val, self.noiseAug = ramLoader.load_to_ram()

        
        
        for i in range(len(self.p)):
            gc.collect()
            tf.keras.backend.clear_session()
            tf.config.optimizer.set_jit(True)
            mixed_precision.set_global_policy('mixed_float16')

            model_info = {"model_nr_type" : self.model_nr_type, "index" : i}
            print(f"Model nr {i + 1} of {len(self.p)}")           
            # Translate picks to a more readable format:
            num_channels = self.num_channels
            epoch = self.p[i]["epochs"]
            batch_size = self.p[i]["batch_size"]
            
            opt = self.helper.get_optimizer(self.p[i]["optimizer"], self.p[i]["learning_rate"])
    

            if "decay_sequence" in self.p[i]:
                if "num_filters" in self.p[i]:
                    units_or_num_filters = self.p[i]["num_filters"]
                else:
                    units_or_num_filters = self.p[i]["units"]
                num_layers = self.p[i]["num_layers"]
                self.p[i]["decay_sequence"] = self.helper.get_max_decay_sequence(num_layers, 
                                                                                units_or_num_filters, 
                                                                                self.p[i]["decay_sequence"], 
                                                                                self.num_classes)
   
            if "first_dense_units" in self.p[i] and "second_dense_units" in self.p[i]:
                if self.p[i]["first_dense_units"] < self.p[i]["second_dense_units"]:
                    self.p[i]["second_dense_units"] = self.p[i]["first_dense_units"]
            
            current_picks = [model_info, self.p[i]]
            pp.pprint(current_picks)
            # Store picked parameters:
            if self.log_data:
                self.results_df = self.store_params_before_fit_opti(self.p[i], self.results_df, self.results_file_name)

            _, _, timesteps = self.x_train.shape

            input_shape = (timesteps, self.num_channels)
            
            if self.is_dynamic:  
                model = DynamicModels(self.model_nr_type, self.num_classes, input_shape, **self.p[i]).model
            else:
                raise Exception("Static models are not handled by this class yet.")
            
            # Initializing generators:
            #gen = RamGenerator(self.loadData, self.handler, self.noiseAug)
            train_enq = GeneratorEnqueuer(data_generator(self.x_train, self.y_train, batch_size, self.noiseAug, num_channels = num_channels, is_lstm  = self.is_lstm), use_multiprocessing = False)
            val_enq = GeneratorEnqueuer(data_generator(self.x_val, self.y_val,batch_size, self.noiseAug, num_channels = num_channels, is_lstm  = self.is_lstm), use_multiprocessing = False)
            train_enq.start(workers = 16, max_queue_size = 15)
            val_enq.start(workers = 16, max_queue_size = 15)
            train_gen = train_enq.get()
            val_gen = train_enq.get()

            # Generate compiler args using picks
            model_compile_args = self.helper.generate_model_compile_args(opt, self.num_classes)
            # Compile model using generated args
            model.compile(**model_compile_args)
            
            print("Starting: ")
            pp.pprint(self.p[i])
            print("---------------------------------------------------------------------------------")

            
            # Generate fit args using picks.
            fit_args = self.helper.generate_fit_args(self.train_ds, self.val_ds, self.loadData, 
                                                     batch_size, epoch, val_gen, 
                                                     use_tensorboard = self.use_tensorboard, 
                                                     use_liveplots = self.use_liveplots, 
                                                     use_custom_callback = self.use_custom_callback,
                                                     use_early_stopping = self.use_early_stopping,
                                                     use_reduced_lr = self.use_reduced_lr)
            try:
                print(f"Utilizes {self.helper.get_steps_per_epoch(self.loadData.val, batch_size)*batch_size}/{len(self.loadData.val)} validation points")
                print(f"Utilizes {self.helper.get_steps_per_epoch(self.loadData.train, batch_size)*batch_size}/{len(self.loadData.train)} training points")
                print("---------------------------------------------------------------------------------")
                
                # Fit the model using the generated args
                model.fit(train_gen, **fit_args)
                
                # Evaluate the fitted model on the validation set
                val_eval = model.evaluate(x=val_gen,
                                          steps=self.helper.get_steps_per_epoch(self.loadData.val, batch_size),
                                          return_dict = True)
                pp.pprint(val_eval)
                
                metrics = {}
                metrics['val'] = {  "val_loss" : val_eval["loss"],
                                    "val_accuracy" : val_eval["binary_accuracy"],
                                    "val_precision": val_eval["precision"],
                                    "val_recall" : val_eval["recall"]}
                
                # Evaluate the fitted model on the train set
                
                train_eval = model.evaluate(x=train_gen,
                                            steps=self.helper.get_steps_per_epoch(self.loadData.train, batch_size),
                                            return_dict = True)
                train_enq.stop()
                val_enq.stop()
                
                metrics['train'] = { "train_loss" : train_eval["loss"],
                                    "train_accuracy" : train_eval["binary_accuracy"],
                                    "train_precision": train_eval["precision"],
                                    "train_recall" : train_eval["recall"]}
                
                
                conf, _ = self.helper.evaluate_model_generator(model, self.x_val, self.y_val, self.loadData.label_dict, num_channels = self.num_channels, plot = False, run_evaluate = False)
                train_enq.stop()
                val_enq.stop()
                gc.collect()
                
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                del model, train_gen, val_gen, train_enq, val_enq
                
                if self.log_data:
                    self.results_df = self.store_metrics_after_fit(metrics, conf, self.results_df, self.results_file_name)

            except Exception as e:
                print(str(e))
                print("Something went wrong while training the model (most likely)")

                continue


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
       

    
    def get_n_params_from_list(self, grid, n_picks):
        print(f"Length of grid: {len(grid)}")
        indexes = random.sample(range(0, len(grid)), n_picks)
        picks = [grid[idx] for idx in indexes]
        return picks