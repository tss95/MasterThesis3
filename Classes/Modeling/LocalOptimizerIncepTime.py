
import sys
import pandas as pd
import pprint

from sklearn.model_selection import ParameterGrid
from itertools import chain
import tensorflow as tf
from tensorflow.keras import mixed_precision

from Classes.DataProcessing.LoadData import LoadData
from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.DataProcessing.DataHandler import DataHandler

from Classes.DataProcessing.RamLoader import RamLoader
from Classes.DataProcessing.RamGenerator import RamGenerator
from Classes.Modeling.InceptionTimeModel import InceptionTimeModel
from Classes.Modeling.GridSearchResultProcessor import GridSearchResultProcessor
from Classes.Modeling.LocalOptimizer import LocalOptimizer

import time
import datetime

class LocalOptimizerIncepTime(LocalOptimizer):

    def __init__(self, loadData, scaler_name, use_time_augmentor, use_noise_augmentor, filter_name,
                 use_tensorboard, use_liveplots, use_custom_callback, use_early_stopping, band_min, band_max, 
                 highpass_freq, use_reduced_lr, num_channels, depth, quick_mode = False, 
                 continue_from_result_file = False, result_file_name = "", start_grid = []):
        super().__init__(loadData, scaler_name, use_time_augmentor, use_noise_augmentor, filter_name,
                        use_tensorboard, use_liveplots, use_custom_callback, use_early_stopping, band_min, band_max, 
                        highpass_freq, use_reduced_lr, num_channels, depth, quick_mode, 
                        continue_from_result_file, result_file_name, start_grid)
        self.model_nr_type = "InceptionTime"
        
    
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
            tf.keras.backend.clear_session()
            mixed_precision.set_global_policy('mixed_float16')

            print(f"Model nr {i + 1} of {len(search_space)}")

            pp.pprint(search_space[i])

            batch_size = search_space[i]["batch_size"]
            epochs = search_space[i]["epochs"]
            learning_rate = search_space[i]["learning_rate"]
            opt = self.helper.get_optimizer(search_space[i]["optimizer"], learning_rate)
            
            use_residuals = search_space[i]["use_residuals"]
            use_bottleneck = search_space[i]["use_bottleneck"]
            nr_modules = search_space[i]["nr_modules"]
            kernel_size = search_space[i]["kernel_size"]
            bottleneck_size = search_space[i]["bottleneck_size"]
            num_filters =  search_space[i]["num_filters"]
            shortcut_activation = search_space[i]["shortcut_activation"]
            module_activation = search_space[i]["module_activation"]
            module_output_activation = search_space[i]["module_output_activation"]
            output_activation = search_space[i]["output_activation"]

            reg_shortcut = search_space[i]["reg_shortcut"]
            reg_module = search_space[i]["reg_module"]
            l1_r = search_space[i]["l1_r"]
            l2_r = search_space[i]["l2_r"]

            if log_data:

                self.results_df = self.store_params_before_fit_opti(search_space[i], self.results_df, self.result_file_name)
            
            # Generate build model args using the picks from above.
            _, channels, timesteps = self.handler.get_trace_shape_no_cast(self.loadData.train, self.use_time_augmentor)
            input_shape = (self.num_channels, timesteps)

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
            train_gen = gen.data_generator(self.x_train, self.y_train, batch_size, self.num_channels)
            val_gen = gen.data_generator(self.x_val, self.y_val, batch_size, self.num_channels)

            # Generate fit args using picks.
            fit_args = self.helper.generate_fit_args(self.loadData.train, self.loadData.val, batch_size, 
                                                     epochs, val_gen, use_tensorboard = self.use_tensorboard, 
                                                     use_liveplots = self.use_liveplots, 
                                                     use_custom_callback = self.use_custom_callback,
                                                     use_early_stopping = self.use_early_stopping,
                                                     use_reduced_lr = self.use_reduced_lr)

            # Fit the model using the generated args
            try:
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
                
                # Evaluate the fitted model on the train set
                # Likely very redundant
                train_loss, train_accuracy, train_precision, train_recall = model.evaluate(x=train_gen,
                                                                                            steps=self.helper.get_steps_per_epoch(self.loadData.train,
                                                                                                                                batch_size))
                metrics['train'] = { "train_loss" : train_loss,
                                    "train_accuracy" : train_accuracy,
                                    "train_precision": train_precision,
                                    "train_recall" : train_recall}
                if log_data:
                    self.results_df = self.store_metrics_after_fit(metrics, self.results_df, self.result_file_name)

            except Exception as e:
                print(e)
                print("Error (hopefully) occured during training.")
                continue
        self.run_exhaustive_mode(optimize_metric, nr_candidates, metric_gap, log_data)

            
        



    def run_quick_mode(self, optimize_metric, nr_candidates):
        raise Exception("Quick mode has not yet been implemented")


    def create_search_grid(self, main_grid):
        # This is the least robust function in this class. 
        return {'batch_size' : self.create_batch_params(main_grid['batch_size']),
                     'epochs' : self.create_epochs_params(main_grid['epochs']),
                     'learning_rate' : self.create_learning_rate_params(main_grid['learning_rate']),
                     'optimizer' : self.create_optimizer_params(main_grid['optimizer']),
                     'bottleneck_size' : self.create_bottleneck_size(main_grid['bottleneck_size']),
                     'kernel_size' : self.create_kernel_and_filter_params(main_grid['kernel_size']),
                     'l1_r' : self.create_reg_params(main_grid['l1_r']),
                     'l2_r' : self.create_reg_params(main_grid['l2_r']),
                     'module_activation' : self.create_activation_params(main_grid['module_activation'], include_linear = True),
                     'module_output_activation' : self.create_activation_params(main_grid['module_output_activation'], include_linear = True),
                     'nr_modules' : self.create_nr_modules_params(main_grid['nr_modules']),
                     'num_filters' : self.create_kernel_and_filter_params(main_grid['num_filters']),
                     'output_activation' : self.create_output_activation(main_grid['output_activation']),
                     'reg_module' : self.create_boolean_params(main_grid['reg_module']),
                     'reg_shortcut' : self.create_boolean_params(main_grid['reg_shortcut']),
                     'shortcut_activation' : self.create_activation_params(main_grid['shortcut_activation'], include_linear = False),
                     'use_bottleneck' : self.create_boolean_params(main_grid['use_bottleneck']),
                     'use_residuals' : self.create_boolean_params(main_grid['use_residuals'])}

    def create_nr_modules_params(self, center):
        max_modules = 30
        new_nr_modules = [center - 6, center - 3, center + 3, center + 6]
        for i in range(len(new_nr_modules)):
            new_nr_modules[i] = min(max(new_nr_modules[i], 1), max_modules)
        return list(set(new_nr_modules))
    
    def create_kernel_and_filter_params(self, current):
        max_size = 120
        new_kernels = [current - 20, current - 10, current - 2, current + 2, current + 10, current + 20]
        for i, kern in enumerate(new_kernels):
            new_kernels[i] = min(max(kern, 3), max_size)
        return list(set(new_kernels))

    def create_bottleneck_size(self, current_nr):
        max_nr = 100
        new_bottleneck = [current_nr - 4, current_nr - 2, current_nr + 2, current_nr + 4]
        for i, neck in enumerate(new_bottleneck):
            new_bottleneck[i] = min(max(neck, 2), max_nr)
        return list(set(new_bottleneck))