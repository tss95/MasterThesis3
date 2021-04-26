import numpy as np
import pandas as pd
import h5py
import gc
import traceback

import sklearn as sk
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import GeneratorEnqueuer

import os
base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3'
os.chdir(base_dir)

from Classes.Modeling.DynamicModels import DynamicModels
from Classes.Modeling.TrainSingleModel import TrainSingleModel
from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.Modeling.GridSearchResultProcessor import GridSearchResultProcessor
from Classes.DataProcessing.ts_RamGenerator import data_generator


import sys


import random
import pprint
import re
import json




class TrainSingleModelRam(TrainSingleModel):
    
    def __init__(self, noiseAug, helper, loadData, 
                 model_type, num_channels, use_tensorboard, use_liveplots, use_custom_callback,
                 use_early_stopping, use_reduced_lr, ramLoader, log_data = True, start_from_scratch = False, beta = 1):        
        self.num_channels = num_channels
        self.beta = beta
        
        self.noiseAug = noiseAug
        self.helper = helper
        self.loadData = loadData
        self.ramLoader = ramLoader
        
        self.use_tensorboard = use_tensorboard
        self.use_liveplots = use_liveplots
        self.use_custom_callback = use_custom_callback
        self.use_early_stopping = use_early_stopping
        self.use_reduced_lr = use_reduced_lr
        
        self.num_classes = len(set(self.loadData.label_dict.values()))
        
        self.model_type = model_type
        self.log_data = log_data
        self.start_from_scratch = start_from_scratch
        self.resultsProcessor = GridSearchResultProcessor(self.num_classes, self.model_type, self.loadData, self.ramLoader, self.use_early_stopping, self.num_channels, self.beta)
        super().__init__(self.resultsProcessor)
    
    
    def create_enqueuer(self, X, y, batch_size, noiseAug, num_channels):
        norm_scale = False
        if self.ramLoader.scaler_name == "normalize":
            norm_scale = True
        enq = GeneratorEnqueuer(data_generator(X, y, batch_size, noiseAug, num_channels = num_channels, is_lstm  = True, norm_scale = norm_scale), 
                                use_multiprocessing = False)
        return enq

    def prep_and_fit_model(self, model, x_train, y_train, x_val, y_val, workers, max_queue_size, meier_mode = False, **p):
        print("Before starting enquers and generators")
        tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        print(f"------------------RAM usage: {used_m}/{tot_m} (Free: {free_m})------------------")
        train_enq = self.create_enqueuer(x_train, y_train, p["batch_size"], self.noiseAug, num_channels = self.num_channels)
        val_enq = self.create_enqueuer(x_val, y_val, p["batch_size"], self.noiseAug, num_channels = self.num_channels)
        train_enq.start(workers = workers, max_queue_size = max_queue_size)
        val_enq.start(workers = workers, max_queue_size = max_queue_size)
        return self.fit_model(model, train_enq, val_enq, y_val, meier_mode, **p)



    def metrics_producer(self, model, x_train, y_train, x_val, y_val, workers, max_queue_size, meier_mode = False,**p):
        print("Finished training, starting evaluation")
        tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        print(f"------------------RAM usage: {used_m}/{tot_m} (Free: {free_m})------------------")
        print("Evaluating validation:")
        val_enq = self.create_enqueuer(x_val, y_val, p["batch_size"], self.noiseAug, self.num_channels)
        val_enq.start(workers = workers, max_queue_size = max_queue_size)
        val_gen = val_enq.get()
        val_eval = model.evaluate(x = val_gen, batch_size = p["batch_size"],
                                  steps = self.helper.get_steps_per_epoch(self.loadData.val, p["batch_size"]),
                                  return_dict = True)
        val_enq.stop()
        del val_enq, val_gen
        val_conf, _, val_acc, val_precision, val_recall, val_fscore = self.helper.evaluate_generator(model, x_val, y_val, p["batch_size"], self.loadData.label_dict, self.num_channels, self.noiseAug, self.ramLoader.scaler_name, self.num_classes, beta = self.beta)       
        metrics = {}
        metrics['val'] = {  "val_loss" : val_eval["loss"],
                            "val_accuracy" : val_acc,
                            "val_precision": val_precision,
                            "val_recall" : val_recall,
                            f"val_f{self.beta}" : val_fscore}

        
        print("Evaluating train:")
        train_enq = self.create_enqueuer(x_train, y_train, p["batch_size"], self.noiseAug, self.num_channels)
        train_enq.start(workers = workers, max_queue_size = max_queue_size)
        train_gen = train_enq.start()
        train_eval = model.evaluate(x = train_gen, batch_size = p["batch_size"],
                                    steps = self.helper.get_steps_per_epoch(self.loadData.train, p["batch_size"]),
                                    return_dict = True)
                
        train_enq.stop()
        del train_enq, train_gen
        _, _, train_acc, train_precision, train_recall, train_fscore = self.helper.evaluate_generator(model, x_train, y_train, p["batch_size"], self.loadData.label_dict, self.num_channels, self.noiseAug, self.ramLoader.scaler_name, self.num_classes, beta = self.beta)
        metrics['train'] = { "train_loss" : train_eval["loss"],
                            "train_accuracy" : train_acc,
                            "train_precision": train_precision,
                            "train_recall" : train_recall,
                            f"train_f{self.beta}" : train_fscore}
        print("Finished evaluation")
        tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        print(f"------------------RAM usage: {used_m}/{tot_m} (Free: {free_m})------------------")
        return metrics, val_conf
    
    def run(self, x_train, y_train, x_val, y_val, x_test, y_test, workers, max_queue_size, evaluate_train = False, evaluate_val = False, evaluate_test = False, meier_mode = False, index = None, **p):
        if self.log_data and self.results_df is None and self.results_file_name is None:
            self.p = p
            self.create_result_file()
        input_shape = (x_train.shape[2], x_train.shape[1])
        model = self.create_and_compile_model(input_shape, index = index, meier_mode = meier_mode, **p)
        model = self.prep_and_fit_model(model, x_train, y_train, x_val, y_val, workers, max_queue_size, meier_mode = meier_mode, **p)
        if self.log_data and self.results_df is not None and self.results_file_name != None:
            metrics, val_conf = self.metrics_producer(model, x_train, y_train, x_val, y_val, max(1, int(workers//2)), int(max_queue_size), meier_mode, **p)
            self.results_df = self.resultsProcessor.store_metrics_after_fit(metrics, val_conf, self.results_df, self.results_file_name)
            print(self.results_df.iloc[-1])
        if evaluate_train:
            print("Unsaved train eval:")
            self.helper.evaluate_generator(model, x_train, y_train, p['batch_size'], self.loadData.label_dict, self.num_channels, self.noiseAug, self.ramLoader.scaler_name, self.num_classes, beta = self.beta)
        if evaluate_val:
            print("Unsaved val eval:")
            self.helper.evaluate_generator(model, x_val, y_val, p["batch_size"],self.loadData.label_dict, self.num_channels, self.noiseAug, self.ramLoader.scaler_name,  self.num_classes, beta = self.beta)
        if evaluate_test:
            print("Unsaved test eval:")
            self.helper.evaluate_generator(model, x_test, y_test, p["batch_size"], self.loadData.label_dict, self.num_channels, self.noiseAug, self.ramLoader.scaler_name,  self.num_classes, beta = self.beta)
        gc.collect()
        return model, self.results_df