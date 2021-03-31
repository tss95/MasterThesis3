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
from Classes.DataProcessing.NoiseAugmentor import NoiseAugmentor
from Classes.DataProcessing.TimeAugmentor import TimeAugmentor
from Classes.DataProcessing.DataGenerator import DataGenerator
from Classes.Scaling.MinMaxScalerFitter import MinMaxScalerFitter
from Classes.Scaling.StandardScalerFitter import StandardScalerFitter
from .GridSearchResultProcessor import GridSearchResultProcessor
from Classes.DataProcessing.ts_RamGenerator import data_generator


import sys


import random
import pprint
import re
import json

class RandomGridSearchDynamic(GridSearchResultProcessor):
    hyper_grid = {
            "num_layers" : [2,3,4,5,6], 
            # so"batch_size" : [8, 16, 32, 64, 128, 256],
            "batch_size" : [256, 512],
            "epochs" : [50, 65, 70, 75, 80],
            "learning_rate" : [0.1, 0.01, 0.001, 0.0001, 0.00001],
            "optimizer" : ["adam", "rmsprop", "sgd"]
        }
    model_grid = {
        "activation" : ["relu", "sigmoid", "softmax", "tanh"],
        "start_neurons" : [2,4,8,16, 32, 64, 128, 256, 512],
        "decay_sequence" : [[1,2,4,6,8,10]],
        "dropout_rate" : [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0],
        "filters" : [3, 9, 15, 21],
        "kernel_size" : [3],
        "padding" : ["same", "valid"],
        "l2_r" : [0.3, 0.2, 0.1, 0.01, 0.001, 0.0001],
        "l1_r" : [0.3, 0.2, 0.1, 0.01, 0.001, 0.0001],
        "output_layer_activation" : ["relu", "sigmoid", "softmax", "tanh"]
    }
    

    def __init__(self, loadData, train_ds, val_ds, test_ds, model_type, scaler_name, use_time_augmentor, use_noise_augmentor,
                 filter_name, n_picks, hyper_grid=hyper_grid, model_grid=model_grid, use_tensorboard = False, 
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
        self.model_grid = model_grid
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
        #if self.loadData.earth_explo_only:
        #    self.full_ds = np.concatenate((self.loadData.noise_ds, self.loadData.full_ds))
        #else:
        #    self.full_ds = self.loadData.full_ds
            

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
            gc.collect()
            tf.keras.backend.clear_session()
            tf.config.optimizer.set_jit(True)
            mixed_precision.set_global_policy('mixed_float16')

            model_info = {"model_nr_type" : self.model_nr_type, "index" : i}
            print(f"Model nr {i + 1} of {len(self.hyper_picks)}")           
            # Translate picks to a more readable format:
            num_channels = self.num_channels
            epoch = self.hyper_picks[i]["epochs"]
            batch_size = self.hyper_picks[i]["batch_size"]
            dropout_rate = self.model_picks[i]["dropout_rate"]
            activation = self.model_picks[i]["activation"]
            output_layer_activation = self.model_picks[i]["output_layer_activation"]
            l2_r = self.model_picks[i]["l2_r"]
            l1_r = self.model_picks[i]["l1_r"]
            start_neurons = self.model_picks[i]["start_neurons"]
            filters = self.model_picks[i]["filters"]
            kernel_size = self.model_picks[i]["kernel_size"]
            padding = self.model_picks[i]["padding"]
            opt = self.helper.get_optimizer(self.hyper_picks[i]["optimizer"], self.hyper_picks[i]["learning_rate"])
            decay_sequence = None
            num_layers = None
            use_layerwise_dropout_batchnorm = None

            if self.is_dynamic:
                num_layers = self.hyper_picks[i]["num_layers"]
                self.model_picks[i]["decay_sequence"] = self.helper.get_max_decay_sequence(num_layers, 
                                                                                            start_neurons, 
                                                                                            self.model_picks[i]["decay_sequence"], 
                                                                                            self.num_classes)
                decay_sequence = self.model_picks[i]["decay_sequence"]
                use_layerwise_dropout_batchnorm = self.model_picks[i]["use_layerwise_dropout_batchnorm"]
            
            
            current_picks = [model_info, self.hyper_picks[i], self.model_picks[i]]
            pp.pprint(current_picks)
            # Store picked parameters:
            if self.log_data:
                self.results_df = self.store_params_before_fit(current_picks, self.results_df, self.results_file_name)

            _, _, timesteps = self.x_train.shape

            # Generate build model args using the picks from above.
            model_args = self.helper.generate_build_model_args(self.model_nr_type, batch_size, dropout_rate, 
                                                               activation, output_layer_activation,
                                                               l2_r, l1_r, start_neurons, filters, kernel_size, 
                                                               padding, 
                                                               num_layers = num_layers, 
                                                               is_lstm = self.is_lstm, 
                                                               num_classes = self.num_classes,
                                                               decay_sequence = decay_sequence, 
                                                               channels = self.num_channels, 
                                                               timesteps = timesteps,
                                                               use_layerwise_dropout_batchnorm = use_layerwise_dropout_batchnorm,
                                                               )
            # Build model using args generated above
            if self.is_dynamic:  
                model = DynamicModels(**model_args).model
            else:
                model = StaticModels(**model_args).model
            
            # Initializing generators:
            #gen = RamGenerator(self.loadData, self.handler, self.noiseAug)
            train_enq = GeneratorEnqueuer(data_generator(self.x_train, self.y_train, batch_size, self.loadData, self.handler, self.noiseAug, num_channels = num_channels, is_lstm  = self.is_lstm), use_multiprocessing = False)
            val_enq = GeneratorEnqueuer(data_generator(self.x_val, self.y_val,batch_size, self.loadData, self.handler, self.noiseAug, num_channels = num_channels, is_lstm  = self.is_lstm), use_multiprocessing = False)
            train_enq.start(workers = 16, max_queue_size = 15)
            val_enq.start(workers = 16, max_queue_size = 15)
            train_gen = train_enq.get()
            val_gen = train_enq.get()

            # Generate compiler args using picks
            model_compile_args = self.helper.generate_model_compile_args(opt, self.num_classes)
            # Compile model using generated args
            model.compile(**model_compile_args)
            
            print("Starting: ")
            pp.pprint(self.hyper_picks[i])
            print("---------------------------------------------------------------------------------")
            pp.pprint(self.model_picks[i])

            
            # Generate fit args using picks.
            fit_args = self.helper.generate_fit_args(self.train_ds, self.val_ds, batch_size, 
                                                     epoch, val_gen, use_tensorboard = self.use_tensorboard, 
                                                     use_liveplots = self.use_liveplots, 
                                                     use_custom_callback = self.use_custom_callback,
                                                     use_early_stopping = self.use_early_stopping,
                                                     use_reduced_lr = self.use_reduced_lr)
            try:
                # Fit the model using the generated args
                model.fit(train_gen, **fit_args)
                
                # Evaluate the fitted model on the validation set
                loss, accuracy, precision, recall = model.evaluate(x=val_gen,
                                                                   steps=self.helper.get_steps_per_epoch(self.loadData.val, batch_size))
                # Record metrics for train
                metrics = {}
                metrics['val'] = {  "val_loss" : loss,
                                    "val_accuracy" : accuracy,
                                    "val_precision": precision,
                                    "val_recall" : recall}
                current_picks.append(metrics['val'])
                
                # Evaluate the fitted model on the train set
                train_loss, train_accuracy, train_precision, train_recall = model.evaluate(x=train_gen,
                                                                                            steps=self.helper.get_steps_per_epoch(self.loadData.train, batch_size))
                train_enq.stop()
                val_enq.stop()
                metrics['train'] = { "train_loss" : train_loss,
                                    "train_accuracy" : train_accuracy,
                                    "train_precision": train_precision,
                                    "train_recall" : train_recall}
                current_picks.append(metrics['train'])
                
                _ = self.helper.evaluate_model(model, self.x_val, self.y_val, self.loadData.label_dict, num_channels = self.num_channels, plot = False, run_evaluate = False)
                train_enq.stop()
                val_enq.stop()
                gc.collect()
                
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                del model, train_gen, val_gen, train_enq, val_enq
                if self.log_data:
                    self.results_df = self.store_metrics_after_fit(metrics, self.results_df, self.results_file_name)

            except Exception as e:
                print(str(e))
                print("Something went wrong while training the model (most likely)")

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
       
    
    """
    def fit_from_index(self, results_df, index):
        values = list(results_df.iloc[index][0:13])
        keys = list(results_df.columns[0:13])
        params = {keys[i]: values[i] for i in range(len(keys))}

        build_model_args = self.helper.generate_build_model_args(self.model_type, int(params['batch_size']), 
                                                                 float(params['dropout_rate']), params['activation'], 
                                                                 params['output_layer_activation'], float(params['l2_r']), 
                                                                 float(params['l1_r']), int(params['start_neurons']),
                                                                 int(params['filters']), int(params['kernel_size']),
                                                                 params['padding'], num_layers = int(params['num_layers']), 
                                                                 decay_divisor = int(params['decay_divisor']), 
                                                                 num_classes =  self.num_classes)
        # Build model using args generated above
        model = Models(**build_model_args).model

        # Generate generator args using picks.
        gen_args = self.helper.generate_gen_args(int(params['batch_size']), self.detrend, 
                                                 use_scaler = self.use_scaler, scaler = self.scaler, 
                                                 use_time_augmentor = self.use_time_augmentor,
                                                 timeAug = self.timeAug,
                                                 use_noise_augmentor = self.use_noise_augmentor, 
                                                 noiseAug = self.noiseAug)

        # Initiate generators using the args
        train_gen = self.dataGen.data_generator(self.train_ds, **gen_args)
        val_gen = self.dataGen.data_generator(self.val_ds, **gen_args)
        test_gen = self.dataGen.data_generator(self.test_ds, **gen_args)

        # Generate compiler args using picks
        opt = self.helper.getOptimizer(params['optimizer'], float(params['learning_rate']))
        model_compile_args = self.helper.generate_model_compile_args(opt, self.num_classes)
        # Compile model using generated args
        model.compile(**model_compile_args)

        # Generate fit args using picks.
        fit_args = self.helper.generate_fit_args(self.train_ds, self.val_ds, int(params['batch_size']), 
                                                 int(params['epochs']), test_gen, use_tensorboard = self.use_tensorboard, 
                                                 use_liveplots = True, 
                                                 use_custom_callback = self.use_custom_callback,
                                                 use_early_stopping = self.use_early_stopping)
        # Fit the model using the generated args
        model_fit = model.fit(train_gen, **fit_args)
        
        helper.plot_confusion_matrix(model, test_gen, self.test_ds, int(params['batch_size']), self.num_classes)

        # Evaluate the fitted model on the test set
        loss, accuracy, precision, recall = model.evaluate_generator(generator=test_gen,
                                                                   steps=self.helper.get_steps_per_epoch(self.test_ds, 
                                                                                                         int(params['batch_size'])))

        pp = pprint.PrettyPrinter(indent=4)
        print(f'Test loss: {loss}')
        print(f'Test accuracy: {accuracy}')
        print(f'Test precision: {precision}')
        print(f'Test recall: {recall}')
        return model
    """
    
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
       



    
    
    
    
    
    
    
    
    
    
    
    