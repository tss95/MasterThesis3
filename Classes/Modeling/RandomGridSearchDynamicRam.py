import numpy as np
import pandas as pd
import h5py

import sklearn as sk
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ModelCheckpoint

import keras
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from tensorflow.keras import regularizers
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import ParameterGrid




import tensorflow as tf

from .DynamicModels import DynamicModels


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
import os


from livelossplot import PlotLossesKeras
import random
import pprint
import re
import json

base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3.0/Classes'

class RandomGridSearchDynamicRam(GridSearchResultProcessor):
    
    hyper_grid = {
            "num_layers" : [2,3,4,5,6], 
            "batch_size" : [8, 16, 32, 64, 128, 256],
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
    

    def __init__(self, loadData, train_ds, val_ds, test_ds, model_type, detrend, use_scaler, use_time_augmentor, use_noise_augmentor,
                 use_minmax, use_highpass, n_picks, hyper_grid=hyper_grid, model_grid=model_grid, use_tensorboard = False, 
                 use_liveplots = True, use_custom_callback = False, use_early_stopping = False, highpass_freq = 0.1, 
                 start_from_scratch = True, is_lstm = False, num_channels = 3):
        
        self.loadData = loadData
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.model_nr_type = model_type
        self.num_classes = len(set(self.loadData.label_dict.values()))
        self.detrend = detrend
        self.use_scaler = use_scaler
        self.use_noise_augmentor = use_noise_augmentor
        self.use_time_augmentor = use_time_augmentor
        self.use_minmax = use_minmax
        self.use_highpass = use_highpass
        self.n_picks = n_picks
        self.hyper_grid = hyper_grid
        self.hyper_grid["num_channels"] = [num_channels]
        self.model_grid = model_grid
        self.use_tensorboard = use_tensorboard
        self.use_liveplots = use_liveplots
        self.use_custom_callback = use_custom_callback
        self.use_early_stopping = use_early_stopping
        self.highpass_freq = highpass_freq
        self.start_from_scratch = start_from_scratch
        self.is_lstm = is_lstm
        self.num_channels = num_channels

        print("Note that highpass and detrend have not been implemented yet in this class.")
        
        self.helper = HelperFunctions()
        self.ramGen = RamGenerator(self.loadData)
        self.handler = DataHandler(self.loadData)
        self.is_dynamic = False
        if type(self.model_nr_type) == str:
            self.is_dynamic = True
        if self.loadData.earth_explo_only:
            self.full_ds = np.concatenate((self.loadData.noise_ds, self.loadData.full_ds))
        else:
            self.full_ds = self.loadData.full_ds


    
    def fit(self):
        self.timeAug, self.scaler, self.noiseAug = self.init_preprocessing(self.use_time_augmentor, 
                                                                           self.use_scaler, 
                                                                           self.use_noise_augmentor)
        
        # Create name of results file, get initiated results df, either brand new or continue old.
        self.results_file_name = self.get_results_file_name()
        self.hyper_picks = self.get_n_params_from_list(list(ParameterGrid(self.hyper_grid)), self.n_picks)
        self.model_picks = self.get_n_params_from_list(list(ParameterGrid(self.model_grid)), self.n_picks)
        self.results_df = self.initiate_results_df(self.results_file_name, self.num_classes, self.start_from_scratch, self.hyper_picks[0], self.model_picks[0])
        pp = pprint.PrettyPrinter(indent=4)

        # Preprocessing and loading all data to RAM:
        ramLoader = RamLoader(self.handler, self.timeAug, self.scaler, self.noiseAug)
        self.x_train, self.y_train = ramLoader.load_to_ram(self.train_ds, self.is_lstm, self.num_channels)
        self.x_val, self.y_val = ramLoader.load_to_ram(self.val_ds, self.is_lstm, self.num_channels)

        for i in range(len(self.hyper_picks)):
            model_info = {"model_nr_type" : self.model_nr_type, "index" : i}
            print(f"Model nr {i + 1} of {len(self.hyper_picks)}")   
            print(i+1)
            # Translate picks to a more readable format:
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
            opt = self.helper.getOptimizer(self.hyper_picks[i]["optimizer"], self.hyper_picks[i]["learning_rate"])
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
            print(current_picks)
            # Store picked parameters:
            self.results_df = self.store_params_before_fit(current_picks, self.results_df, self.results_file_name)
            
            # Generate build model args using the picks from above.
            model_args = self.helper.generate_build_model_args(self.model_nr_type, batch_size, dropout_rate, 
                                                               activation, output_layer_activation,
                                                               l2_r, l1_r, start_neurons, filters, kernel_size, 
                                                               padding, num_layers = num_layers, num_classes = self.num_classes,
                                                               decay_sequence = decay_sequence, channels = self.num_channels,
                                                               use_layerwise_dropout_batchnorm = use_layerwise_dropout_batchnorm,
                                                               is_lstm = self.is_lstm)
            # Build model using args generated above
            if self.is_dynamic:  
                model = DynamicModels(**model_args).model
            else:
                model = StaticModels(**model_args).model

            # Initializing generators:
            gen = RamGenerator(self.loadData)
            train_gen = gen.data_generator(self.x_train, self.y_train, batch_size)
            val_gen = gen.data_generator(self.x_val, self.y_val, batch_size)

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
                                                     use_early_stopping = self.use_early_stopping)
            # Fit the model using the generated args
            model_fit = model.fit(train_gen, **fit_args)
            
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
            train_loss, train_accuracy, train_precision, train_recall = model.evaluate_generator(generator=train_gen,
                                                                                        steps=self.helper.get_steps_per_epoch(self.train_ds,
                                                                                                                              batch_size))
            metrics['train'] = { "train_loss" : train_loss,
                                 "train_accuracy" : train_accuracy,
                                 "train_precision": train_precision,
                                 "train_recall" : train_recall}
            current_picks.append(metrics['train'])
            self.results_df = self.store_metrics_after_fit(metrics, self.results_df, self.results_file_name)
            
        min_loss, max_accuracy, max_precision, max_recall = self.find_best_performers(self.results_df)
        self.print_best_performers(min_loss, max_accuracy, max_precision, max_recall)
        return self.results_df, min_loss, max_accuracy, max_precision, max_recall


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

    def get_n_params_from_list(self, grid_list, n_picks):
        picks = []
        while (n_picks != 0):
            grid_length = len(grid_list)
            rand_int = random.randint(0,grid_length-1)
            picks.append(grid_list[rand_int])
            del grid_list[rand_int]
            n_picks -= 1
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
       