import numpy as np
import os

base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3'
os.chdir(base_dir)

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

class LocalOptimizer(GridSearchResultProcessor):

    """
    This class functions as an heuristic that will attempt to reach a local minima. The class can either start off an existing search, or can start its own. The process looks a little like this:
    1. Select the best model from existing result file (if using a file)
    2. Use the best model / start model as the foundation. Create a search space around this in terms of hyperparameters.
    3. Do a narrow search on the generated search space. If quick_mode = True, then if a better model is found during the narrow search, replace the base model with this and return to step 2. 
    4. Repeat steps 1-3
    

    Notes:
    
    - Would like this to be as robust as possible, and not dependent on InceptionTime. Want to be able to use this class for any model really.
        - This can be challenging when creating dictionaries, as annoyingly, the models use 2 seperate dictionaries for initilization.
        SOLVED
    
    Drawbacks, potential points of failure:
     - The filtering method is very simple, and assumes that less than half of the good models are buggy. This is definitely not necessarily the case, and will cause this class to potentially try to optimize a lost cause. 
         APPEARS TO BE A tf_nightly VERSION ISSUE
     - The way results are processed, requires a model_grid and a hyper_grid. This design choice is the root of sooo many problems, and may lead to a less than robust implementation of this class. This can lead to different versions. Potential soution: Use this class as a parent class, and have children objects that are specialized for each type of model. 


    TODO: Implement method to remove already trained model in the search_space. 
    
    """

    def __init__(self, loadData, use_scaler, use_time_augmentor, use_noise_augmentor, use_minmax, filter_name,
                 use_tensorboard, use_liveplots, use_custom_callback, use_early_stopping, band_min, band_max, highpass_freq,
                 use_reduced_lr, num_channels, depth, quick_mode = False, continue_from_result_file = False, 
                 result_file_name = "", start_grid = []):
        
        self.loadData = loadData
        self.num_classes = len(set(self.loadData.label_dict.values()))
        self.use_scaler = use_scaler
        self.use_time_augmentor = use_time_augmentor
        self.use_noise_augmentor = use_noise_augmentor
        self.use_minmax = use_minmax
        self.filter_name = filter_name
        self.use_tensorboard = use_tensorboard
        self.use_liveplots = use_liveplots
        self.use_custom_callback = use_custom_callback
        self.use_early_stopping = use_early_stopping
        self.band_min = band_min
        self.band_max = band_max
        self.highpass_freq = highpass_freq
        self.use_reduced_lr = use_reduced_lr
        self.num_channels = num_channels

        self.depth = depth
        self.quick_mode = quick_mode
        self.continue_from_result_file = continue_from_result_file
        self.result_file_name = result_file_name
        self.start_grid = start_grid

        self.helper = HelperFunctions()
        self.handler = DataHandler(self.loadData)

        self.current_best_model = None
        self.current_best_metrics = None

    def run(self, optimize_metric = ['val_accuracy', 'val_f1'], nr_candidates = 10, metric_gap = 0.1, log_data = True, skip_to_index = 0):
        """
        Self explanatory

        PARAMS:
        --------------
        result_file_name: (str) Name of the file to be used. If continue_from_result == False, then this will not be used
        num_classes: (int)
        optimize_metric: [string, string] Optimization criteria. First element will be most significant.
        nr_candidates: (int) Number of model candidates that will be considered in the first step sort.

        """
        if self.quick_mode:
            if self.continue_from_result_file:
                print(f"Quick mode, starting of result file: {self.result_file_name}")
                self.run_quick_mode(optimize_metric, nr_candidates, metric_gap, log_data)
                
            else:
                raise Exception("Not continuing training from result file is not yet implemented. Suspected to be unused.")
        else:
            if self.continue_from_result_file:
                print(f"Exhaustive mode, starting of result file: {self.result_file_name}")
                self.run_exhaustive_mode(optimize_metric, nr_candidates, metric_gap, log_data, skip_to_index)
            else:
                raise Exception("Not continuing training from result file is not yet implemented. Suspected to be unused.")
        return

    """
    def determine_model(self, result_file_name):
         name_list = result_file_name.split('_')
         if "InceptionTime" in name_list:
             return LocalOptimizerIncepTime(self.loadData, self.detrend, self.use_scaler, self.use_time_augmentor,
                         self.use_noise_augmentor, self.use_minmax, self.use_highpass, self.use_tensorboard,
                         self.use_liveplots, self.use_custom_callback, self.use_early_stopping, 
                         self.highpass_freq, self.use_reduced_lr, self.num_channels, self.depth, 
                         self.quick_mode, self.continue_from_result_file, self.result_file_name, self.start_grid)
         else:
            raise Excpetion("Other models have not yet been implemented in this class")
    """
    def quick_mode(self, result_file_name, num_classes, optimize_metric, metric_gap):
        pass

    def get_best_model(self, result_file_name, num_classes, optimize_metric, nr_candidates, metric_gap):
        if  "val_loss" in optimize_metric and metric_gap != None:
            raise Exception("Cannot use the metric_gap method when optimizing for loss")

        # Clear nan values
        self.clear_nans(result_file_name)
        results_df = self.get_results_df_by_name(result_file_name)
        df_f1 = results_df.copy()
        # Add f1 stats
        df_f1 = self.add_f1_stats(df_f1)
        # Sort by sort conditions
        sorted_df = self.sort_df(df_f1, optimize_metric)
        highest_first_metric = sorted_df[optimize_metric[0]].head(1).iloc[0]
        print(highest_first_metric)
        if metric_gap != None:
            sorted_df = sorted_df[sorted_df['val_f1'] > highest_first_metric - metric_gap]
    
        # Get the top nr_candidates
            best_initial_candidates = sorted_df.copy().head(nr_candidates)
            return best_initial_candidates.head(1)
        else:
            best_initial_candidates = sorted_df.copy().head(nr_candidates)
            # Attempt to only select models which have the best f1 score, and first part of the sort conditions
            # This is due to (likely) bug that has some models perform really well in one metric, but terrible in other metrics. The working assumption is that models with high f1, are good.
            # TODO: Consider just switching the optimizer metrics here. Without the current BUG with strange training metrics (and inconsistent metrics wrt. the confusion matrix) this is a good opportunity to optimize with two metrics.
            best_initial_sorted_by_f1 = self.sort_df(best_initial_candidates, ['val_f1', optimize_metric[0]])
            # Select nr_candidates//2 of these models, and then resort them by their primary condition.
            reduced_sorted_by_f1 = best_initial_sorted_by_f1.head(nr_candidates//2)
            best_secondary_sorted_by_conditions = self.sort_df(reduced_sorted_by_f1, optimize_metric)
            # At this point we should have filtered out bad outlier models, and be left with good candidates. 
            # We now select the best model according to the sort condidtions.
            best_model = best_secondary_sorted_by_conditions.head(1)

            return best_model

    
    def add_f1_stats(self, df_f1):
        df_f1.columns=df_f1.columns.str.strip()
        all_train_precision = df_f1['train_precision']
        all_train_recall = df_f1['train_recall']
        all_val_precision = df_f1['val_precision']
        all_val_recall = df_f1['val_recall']
        f1_train = self.create_f1_list(all_train_precision, all_train_recall)
        f1_val = self.create_f1_list(all_val_precision, all_val_recall)
        df_f1['train_f1'] = f1_train
        df_f1['val_f1'] = f1_val
        return df_f1

    

    def f1_score(self, precision, recall):
        f1 = 2*((precision*recall)/(precision + recall))
        return f1

    def create_f1_list(self, precision_df, recall_df):
        f1 = []
        for i in range(len(precision_df)):
            f1.append(self.f1_score(precision_df.loc[i], recall_df.loc[i]))
        return f1

        
    def sort_df(self, df, sort_conditions):
        ascending = False
        if sort_conditions == ['val_loss', 'train_loss'] or sort_conditions == ['train_loss', 'val_loss']:
            ascending = True
        if 'val_loss' in sort_conditions and 'train_loss' not in sort_conditions:
            raise Exception("Problematic sorting criteria. Cannot determine if sorting should be ascending or descending. A solution for this needs to be implemented in order for this to work")
        return df.sort_values(by=sort_conditions, axis = 0, ascending = ascending)

    """
    def convert_best_model_to_main_grid(self, best_model):
        model_dict = self.row_to_dict(best_model)


    def row_to_dict(self, model_df):
        keys = list(model_df.keys())
        # Assumes 10 columns dedicated to results and the rest to hyperparams
        hyper_keys = keys[:len(keys) - 10]
        model_dict = model_df[:len(hyper_keys)].to_dict()
        #del model_dict['index']
        return model_dict
    """

    def delete_metrics(self, best_model_df):
        best_model_df = best_model_df[best_model_df.columns[:len(best_model_df.columns) - 10]]
        return best_model_df

    def adapt_best_model_dict(self, best_model_dict):
        print(best_model_dict)
        if 'num_channels' in best_model_dict:
            del best_model_dict['num_channels']
        return {key:[value] for (key,value) in best_model_dict.items()}

    def create_search_grid(self, main_model_grid):
        # Handle hyperparameters that are the same for all models
        param_grid = main_model_grid.copy()
        scaler = range(-4, 4, 2)
    
    
    def create_batch_params(self, batch_center):
        max_batch_size = 4096
        new_params = [batch_center//4, batch_center//2, batch_center*2, batch_center*4]
        for i, batch_size in enumerate(new_params):
            new_params[i] = min(batch_size, max_batch_size)
        return list(set(new_params))
    
    def create_learning_rate_params(self, learning_rate_center):
        min_learning_rate = 0.00001
        new_learning_params = [learning_rate_center*10**2, learning_rate_center*10**1, (learning_rate_center*10)/2, learning_rate_center / 2, learning_rate_center*10**(-1), learning_rate_center*10**(-2)]
        for i, rate in enumerate(new_learning_params):
            new_learning_params[i] = max(rate, min_learning_rate)
        return list(set(new_learning_params))

    def create_epochs_params(self, epoch_center):
        max_epochs = 150
        new_epochs = [epoch_center - 20, epoch_center -10, epoch_center + 10, epoch_center +20]
        for i in range(len(new_epochs)):
            new_epochs[i] = min(max(new_epochs[i], 10), max_epochs)
        return list(set(new_epochs))
    
    def create_optimizer_params(self, current_optimizer):
        options = ["adam", "rmsprop", "sgd"]
        del options[options.index(current_optimizer)]
        return options

    def create_activation_params(self, current_activation, include_linear):
        if include_linear:
            options = ["linear", "relu", "softmax", "tanh", "sigmoid"]
        else: 
            options = ["relu", "softmax", "tanh", "sigmoid"]
        del options[options.index(current_activation)]
        return options

    def create_reg_params(self, current_reg):
        max_reg = 0.3
        if current_reg == 0.0:
            current_reg = 0.01
        new_reg = [current_reg*10**2, current_reg*10, (current_reg*10)/2, current_reg/2, current_reg*10**(-1), current_reg*10**(-2)]
        for i in range(len(new_reg)):
            new_reg[i] = min(new_reg[i], max_reg)
        return list(set(new_reg))

    def create_boolean_params(self, current_bool):
        if current_bool:
            return [False, False]
        else:
            return [True, True]

    def create_output_activation(self, current):
        return [current]

    def get_metrics(self, model, optimize_metric):
        print(model)
        print(optimize_metric)
        return model[optimize_metric[0]].iloc[0], model[optimize_metric[1]].iloc[0]

    def create_search_space(self, main_grid, search_grid):
        key_list = list(main_grid.keys())
        np.random.shuffle(key_list)
        search_list = []
        for key in key_list:
            if len(search_grid[key]) > 1:
                one_model = main_grid.copy()
                one_model[key] = search_grid[key]
                key_grid = list(ParameterGrid(one_model))
                search_list.append(key_grid)
            else:
                continue
        search_list = list(chain.from_iterable(search_list))
        pprint.pprint(search_list)
        return search_list

    def is_new_model_better(self, previous_best_metrics, current_best_metrics):
        if previous_best_metrics == current_best_metrics:
            print("The models have the same metrics. Assumed to be the same models.")
            return False
        if previous_best_metrics[0] > current_best_metrics[0]:
            print("The first optimizer metric is worse on the new model by ", current_best_metrics[0] - previous_best_metric[0])
            if previous_best_metrics[1] < current_best_metrics[1]:
                print("The second optimizer metric is better on the new model by ", current_best_metrics[1] - previous_best_metrics[1])
                # TODO: Consider a better way to handle this. There could possibly be a solution where if the second metric is better, but not the first, the first metric cannot be worse by some condition, e.g. 0.05.
                # An attempt is implemented below, but by now means perfect. 
                if previous_best_metric[0] - current_best_metrics[0] > 0.05:
                    return False
                return True
        else:
            if current_best_metrics[0] < 0.5 or current_best_metrics[1] < 0.5:
                return False
            return True
