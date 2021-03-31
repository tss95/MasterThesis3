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

from Classes.Modeling.LocalOptimizer import LocalOptimizer
from Classes.Modeling.DynamicModels import DynamicModels
from Classes.DataProcessing.LoadData import LoadData
from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.DataProcessing.DataHandler import DataHandler
from Classes.DataProcessing.RamLoader import RamLoader
from .GridSearchResultProcessor import GridSearchResultProcessor
from Classes.DataProcessing.ts_RamGenerator import data_generator


import sys

import time
import datetime


import random
import pprint
import re
import json

class LocalOptimizerDynamic(LocalOptimizer):

    def __init__(self, loadData, scaler_name, use_time_augmentor, use_noise_augmentor, filter_name,
                 use_tensorboard, use_liveplots, use_custom_callback, use_early_stopping, band_min, band_max, 
                 highpass_freq, use_reduced_lr, num_channels, depth, model_nr_type, quick_mode = False, 
                 continue_from_result_file = False, result_file_name = "", start_grid = []):
        super().__init__(loadData, scaler_name, use_time_augmentor, use_noise_augmentor, filter_name,
                        use_tensorboard, use_liveplots, use_custom_callback, use_early_stopping, band_min, band_max, 
                        highpass_freq, use_reduced_lr, num_channels, depth, quick_mode, 
                        continue_from_result_file, result_file_name, start_grid)
        self.model_nr_type = model_nr_type
        self.is_dynamic = False
        if type(self.model_nr_type) == str:
            self.is_dynamic = True
        self.cnn_mode = False
        if self.model_nr_type == "CNN":
            self.cnn_mode = True

    
    def run_exhaustive_mode(self, optimize_metric, nr_candidates, metric_gap, log_data, skip_to_index = 0):
        """
        This function starts of by chosing the best model from the current results file, based on the user defined metrics.

        The model will then create a search space that is near what the current best model is.

        Then the model will train till completion on this search space. One of two things will happen:

        1. If none of the newly trained models are better than the current, then we can do one of three things:
            1.1: End training. Assume we have reached some kind of minima
            1.2 Select the second best model and train on this. This means that the heuristic will never naturally end.
        2. If any of the models are better than the current best model, start this process again.
        

        """
        # Due to rewrite of best_model, we need to store the best model in the last iteration
        previous_best_model = None
        previous_best_metrics = None
        if self.current_best_model is not None and self.current_best_metrics is not None:
            previous_best_model = self.current_best_model
            previous_best_metrics = self.current_best_metrics

        pp = pprint.PrettyPrinter(indent=4)
        # To start of I will only implement what to do when we are continuing off existing file.
        self.current_best_model = self.get_best_model(self.result_file_name, self.num_classes, optimize_metric, nr_candidates, metric_gap)
        self.current_best_metrics = self.get_metrics(self.current_best_model, optimize_metric)

        if previous_best_model is not None:
            # This is where we determine whether the new model is better than the previous model.
            if not self.is_new_model_better(previous_best_metrics, self.current_best_metrics):
                print("The new model is not better")
                return


        print(f"Current best metrics: {optimize_metric[0]} = {self.current_best_metrics[0]}, {optimize_metric[1]} = {self.current_best_metrics[1]}")
        best_model_dict = self.delete_metrics(self.current_best_model).iloc[0].to_dict()
        print("Gained with this model:")
        pp.pprint(best_model_dict)
        search_grid = self.create_search_grid(best_model_dict)
        print("Which will be explored with this search space:")
        pp.pprint(search_grid)
        search_space = self.create_search_space(self.adapt_best_model_dict(best_model_dict), search_grid)
        print("Current search space is of length: ", len(search_space))

        #assert self.get_results_file_name(narrow = True) == self.result_file_name, f"{self.get_results_file_name(narrow = True)} != {self.result_file_name}"
        
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

        self.results_file_name = self.get_results_file_name(narrow = True)
        start = time.time()
        self.x_train, self.y_train, self.x_val, self.y_val, self.timeAug, self.scaler, self.noiseAug = ramLoader.load_to_ram()
        end = time.time()
        print(f"Fitting augmentors and scaler as well as loading to ram completed after: {datetime.timedelta(seconds=(end-start))}")
        
        self.results_df = self.initiate_results_df_opti(self.result_file_name, 
                                                        self.num_classes, 
                                                        False, 
                                                        best_model_dict)

        num_models = len(search_space)



        # Everything prior to the for loop should be general enough to work for any model
        for i in range(num_models):
            if i < skip_to_index:
                continue
            # Housekeeping
            gc.collect()
            tf.keras.backend.clear_session()
            tf.config.optimizer.set_jit(True)
            mixed_precision.set_global_policy('mixed_float16')

            print(f"Model nr {i + 1} of {len(search_space)}")

            opt = self.helper.get_optimizer(search_space[i]['optimizer'], search_space[i]['learning_rate'])

            pp.pprint(search_space[i])
            if self.model_nr_type != "CNN":
                search_space[i]['decay_sequence']= self.helper.get_max_decay_sequence(search_space[i]['num_layers'],
                                                                                    search_space[i]['start_neurons'],
                                                                                    search_space[i]['decay_sequence'],
                                                                                    self.num_classes)

            if log_data:

                self.results_df = self.store_params_before_fit_opti(search_space[i], self.results_df, self.result_file_name)
            
            # Generate build model args using the picks from above.
            _, _, timesteps = self.handler.get_trace_shape_no_cast(self.loadData.train, self.use_time_augmentor)
            input_shape = (timesteps, self.num_channels)

            model_args = self.helper.generate_build_model_args(self.model_nr_type,
                                                               search_space[i]['batch_size'],
                                                               search_space[i]['dropout_rate'],
                                                               search_space[i]['activation'],
                                                               search_space[i]['output_layer_activation'],
                                                               search_space[i]['l2_r'],
                                                               search_space[i]['l1_r'],
                                                               search_space[i]['start_neurons'],
                                                               search_space[i]['filters'],
                                                               search_space[i]['kernel_size'],
                                                               search_space[i]['padding'],
                                                               search_space[i]['num_layers'],
                                                               search_space[i]['decay_sequence'],
                                                               search_space[i]['use_layerwise_dropout_batchnorm'],
                                                               is_lstm = True,
                                                               num_classes = self.num_classes,
                                                               channels = self.num_channels,
                                                               timesteps = timesteps)
            # Build model using args generated above
            if self.is_dynamic:  
                model = DynamicModels(**model_args).model
            else:
                model = StaticModels(**model_args).model

            # Initializing generators:
            train_enq = GeneratorEnqueuer(data_generator(self.x_train, self.y_train, search_space[i]['batch_size'], self.loadData, self.handler, self.noiseAug, num_channels = self.num_channels, is_lstm  = True), use_multiprocessing = False)
            val_enq = GeneratorEnqueuer(data_generator(self.x_val, self.y_val, search_space[i]['batch_size'], self.loadData, self.handler, self.noiseAug, num_channels = self.num_channels, is_lstm  = True), use_multiprocessing = False)
            train_enq.start(workers = 16, max_queue_size = 15)
            val_enq.start(workers = 16, max_queue_size = 15)
            train_gen = train_enq.get()
            val_gen = train_enq.get()


            # Generate compiler args using picks
            model_compile_args = self.helper.generate_model_compile_args(opt, self.num_classes)
            # Compile model using generated args
            model.compile(**model_compile_args)

            # Generate fit args using picks.
            fit_args = self.helper.generate_fit_args(self.loadData.train, self.loadData.val, search_space[i]['batch_size'], 
                                                     search_space[i]['epochs'], val_gen, use_tensorboard = self.use_tensorboard, 
                                                     use_liveplots = self.use_liveplots, 
                                                     use_custom_callback = self.use_custom_callback,
                                                     use_early_stopping = self.use_early_stopping,
                                                     use_reduced_lr = self.use_reduced_lr)

            # Fit the model using the generated args
            try:
                model.fit(train_gen, **fit_args)
                
                # Evaluate the fitted model on the validation set
                loss, accuracy, precision, recall = model.evaluate(x=np.reshape(self.x_val,(self.x_val.shape[0], self.x_val.shape[2], self.x_val.shape[1])), y= self.y_val)
                # Record metrics for train
                metrics = {}
                metrics['val'] = {  "val_loss" : loss,
                                    "val_accuracy" : accuracy,
                                    "val_precision": precision,
                                    "val_recall" : recall}
                
                # Evaluate the fitted model on the train set
                # Likely very redundant
                train_loss, train_accuracy, train_precision, train_recall = model.evaluate(x=np.reshape(self.x_train,(self.x_train.shape[0], self.x_train.shape[2], self.x_train.shape[1])), y = self.y_train)
                metrics['train'] = { "train_loss" : train_loss,
                                    "train_accuracy" : train_accuracy,
                                    "train_precision": train_precision,
                                    "train_recall" : train_recall}

                _ = self.helper.evaluate_model(model, self.x_val, self.y_val, self.loadData.label_dict, plot = False, run_evaluate = False)
                train_enq.stop()
                val_enq.stop()
                gc.collect()
                
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                del model, train_gen, val_gen, train_enq, val_enq
                if log_data:
                    self.results_df = self.store_metrics_after_fit(metrics, self.results_df, self.result_file_name)

            except Exception as e:
                print(e)
                print("Error (hopefully) occured during training.")
                continue
        self.run_exhaustive_mode(optimize_metric, nr_candidates, metric_gap, log_data)




    def create_search_grid(self, main_grid):
        if self.model_nr_type == "CNN":
            return self.create_cnn_search_grid(main_grid)
        if self.model_nr_type == "LSTM":
            raise Exception(f"Create search grid is not implemented for LSTM yet")
            return self.create_lstm_search_grid(main_grid)
        if self.model_nr_type == "DENSE":
            raise Exception(f"Create search grid is not implemented for Dense yet")
            return self.create_dense_search_grid(main_grid)
        raise Exception(f"Local optimizer for {self.model_nr_type} has not yet been implemented.")
    
        
    
    def create_cnn_search_grid(self, main_grid):
        return {'batch_size' : self.create_batch_params(main_grid['batch_size']),
                'epochs' : self.create_epochs_params(main_grid['epochs']),
                'learning_rate' : self.create_learning_rate_params(main_grid['learning_rate']),
                'optimizer' : self.create_optimizer_params(main_grid['optimizer']),
                'activation' : self.create_activation_params(main_grid['activation'], include_linear = False, cnn_mode = self.cnn_mode),
                'dropout_rate' : self. create_dropout_params(main_grid['dropout_rate']),
                'kernel_size' : self.create_filter_size_params(main_grid['kernel_size']),
                'l1_r' : self.create_reg_params(main_grid['l1_r']),
                'l2_r' : self.create_reg_params(main_grid['l2_r']),
                'start_neurons' : self.create_start_neurons_params(main_grid['start_neurons']),
                'filters' : self.create_num_filter_params(main_grid['filters']),
                'padding' : self.create_padding_params(main_grid['padding']),
                'decay_sequence' : np.array(self.create_decay_sequence_params(main_grid['decay_sequence'])),
                'num_layers' : self.create_num_layers_params(main_grid['num_layers']),
                'use_layerwise_dropout_batchnorm' : self.create_boolean_params(main_grid['use_layerwise_dropout_batchnorm']),
                'output_layer_activation' : self.create_output_activation(main_grid['output_layer_activation'])
                }

    def create_dropout_params(self, dropout_center):
        min_dropout = 0
        max_dropout = 0.5
        new_dropout = [dropout_center*10**x for x in range(-2, 2, 1)]
        new_dropout.append((dropout_center*10)/2)
        for idx, param in enumerate(new_dropout):
            new_dropout[idx] = max(min(max_dropout, param), min_dropout)
        return list(set(new_dropout))

    def create_start_neurons_params(self, start_neurons_center):
        max_start_neurons = 300
        min_start_neurons = 4
        new_start_neurons = np.arange(start_neurons_center - 15, start_neurons_center + 15, 5)
        for idx, param in enumerate(new_start_neurons):
            new_start_neurons[idx] = max(min(max_start_neurons, param), min_start_neurons)
        return list(set(new_start_neurons))

    
    
    def create_padding_params(self, current_padding):
        return ["valid", "same"]

    def create_decay_sequence_params(self, current_sequence):
        possibilities = np.array([[1,2,4,4,2,1], [1,4,8,8,4,1], [1,0.5,0.25,0.25,0.5,1], [1,1,1,1,1,1]])
        possibilities = possibilities[possibilities != current_sequence]
        return possibilities

    def create_num_layers_params(self, current_num_layers):
        possibilities = list(range(1, 7, 1))
        possibilities.remove(current_num_layers)
        return possibilities
    