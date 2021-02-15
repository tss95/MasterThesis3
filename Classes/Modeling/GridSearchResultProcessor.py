from Classes.DataProcessing.LoadData import LoadData
import os
import sys
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

import os
base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3'
os.chdir(base_dir)

from GlobalUtils import GlobalUtils
utils = GlobalUtils()

class GridSearchResultProcessor():

    def __init__(self):
        pass
    
    def create_results_df(self, hyper_picks, model_picks):
        hyper_keys = list(hyper_picks.keys())
        model_keys = list(model_picks.keys())
        metrics_train_keys = ["train_loss", "train_accuracy", "train_precision", "train_recall"]
        metrics_val_keys = ["val_loss", "val_accuracy", "val_precision", "val_recall"]
        header = np.concatenate((hyper_keys, model_keys, metrics_train_keys, metrics_val_keys))
        results_df = pd.DataFrame(columns = header)
        return results_df
    
    def initiate_results_df(self, file_name, num_classes, start_from_scratch, hyper_picks, model_picks):
        if start_from_scratch:
            self.clear_results_df(file_name)
            return self.create_results_df(hyper_picks, model_picks)
        else:
            if self.does_result_exist(file_name):
                file_name = file_name.split('/')[-1]
                results_df = self.get_results_df_by_name(file_name, num_classes)
                return results_df
            else:
                return self.create_results_df(hyper_picks, model_picks)

        
    def does_result_exist(self, file_name):
        return isfile(file_name)
        
        
    
    def save_results_df(self, results_df, file_name):
        results_df.to_csv(file_name, mode = 'w', index=False)
    
    def clear_results_df(self, file_name):
        path = self.get_results_file_path()
        file = f"{path}/{file_name}"
        if os.path.isfile(file):
            f = open(file, "w+")
            f.close()
        
    
    def get_results_file_name(self, narrow = False):
        file_name = f"{self.get_results_file_path()}/results_{self.model_nr_type}"
        if narrow:
            file_name = f"{file_name}_NARROW"
        if self.loadData.earth_explo_only:
            file_name = f"{file_name}_earthExplo"
        if self.loadData.noise_earth_only:
            file_name = f"{file_name}_noiseEarth"
        if self.loadData.noise_not_noise:
            file_name = f"{file_name}_noiseNotNoise"
        if self.detrend:
            file_name = f"{file_name}_detrend"
        if self.use_time_augmentor:
            file_name = f"{file_name}_timeAug"
        if self.use_scaler:
            if self.use_minmax:
                file_name = f"{file_name}_mmscale"
            else: 
                file_name = f"{file_name}_sscale"
        if self.use_noise_augmentor:
            file_name = f"{file_name}_noiseAug"
        if self.use_early_stopping:
            file_name = f"{file_name}_earlyS"
        if self.use_highpass:
            file_name = f"{file_name}_highpass-{self.highpass_freq}"
        file_name = file_name + ".csv"
        return file_name
    
    def get_results_file_path(self):
        file_path = f'{utils.base_dir}/GridSearchResults/{self.num_classes}_classes'
        return file_path
    
    def store_params_before_fit(self, current_picks, results_df, file_name):

        hyper_params = current_picks[1]
        model_params = current_picks[2]
        picks = []
        for key in list(hyper_params.keys()):
            picks.append(hyper_params[key])
        for key in list(model_params.keys()):
            picks.append(model_params[key])
        nr_fillers = len(results_df.columns) - len(picks)
        for i in range(nr_fillers):
            picks.append(np.nan)
        temp_df = pd.DataFrame(np.array(picks).reshape(1,len(results_df.columns)), columns = results_df.columns)
        results_df = results_df.append(temp_df, ignore_index = True)
        for idx, column in enumerate(results_df.columns):
            if idx >= len(picks):
                results_df[column] = results_df[column].astype('float')
        self.save_results_df(results_df, file_name)
        return results_df


    def store_metrics_after_fit(self, metrics, results_df, file_name):
        results_df = results_df.replace('nan', np.nan)
        unfinished_columns = results_df.columns[results_df.isna().any()].tolist()
        for column in unfinished_columns:
            results_df.iloc[-1, results_df.columns.get_loc(column)] = metrics[column.split('_')[0]][column]
        self.save_results_df(results_df, file_name)
        return results_df
        """
        metrics_train = metrics[1]
        metrics_val = metrics[0]
        finished_train = False
        # Get list of columns containing nan values
        results_df = results_df.replace('nan', np.nan)
        unfinished_columns = results_df.columns[results_df.isna().any()].tolist()
        # Iterate through every unfinished column and change values
        for idx, column in enumerate(unfinished_columns):
            print(column)
            if not finished_train:
                # Change value at column using the metrics
                results_df.iloc[-1, results_df.columns.get_loc(column)] = metrics_train[column]
                if idx == len(metrics_train)-1:
                    finished_train = True
            else:
                # Change value at column using the metrics
                results_df.iloc[-1, results_df.columns.get_loc(column)] = metrics_val[column]
        self.save_results_df(results_df, file_name)
        return results_df
        """
    
    def find_best_performers(self, results_df):
        train_loss_index = results_df.columns.get_loc('train_loss')
        metrics_df = results_df[results_df.columns[train_loss_index:]]
        min_loss = {'train_loss' : min(metrics_df['train_loss']), 'val_loss' : min(metrics_df['val_loss']), 
                    'train_index' : metrics_df[metrics_df['train_loss'] == min(metrics_df['train_loss'])].index[0], 
                    'val_index' : metrics_df[metrics_df['val_loss'] == min(metrics_df['val_loss'])].index[0]}

        max_accuracy = {'train_accuracy' : max(metrics_df['train_accuracy']), 'val_accuracy' : max(metrics_df['val_accuracy']), 
                        'train_index' : metrics_df[metrics_df['train_accuracy'] == max(metrics_df['train_accuracy'])].index[0], 
                        'val_index' : metrics_df[metrics_df['val_accuracy'] == max(metrics_df['val_accuracy'])].index[0]}

        max_precision = {'train_precision' : max(metrics_df['train_precision']), 'val_precision' : max(metrics_df['val_precision']), 
                         'train_index' : metrics_df[metrics_df['train_precision'] == max(metrics_df['train_precision'])].index[0], 
                         'val_index' : metrics_df[metrics_df['val_precision'] == max(metrics_df['val_precision'])].index[0]}

        max_recall = {'train_recall' : max(metrics_df['train_recall']), 'val_recall' : max(metrics_df['train_recall']), 
                      'train_index' : metrics_df[metrics_df['train_recall'] == max(metrics_df['train_recall'])].index[0], 
                      'val_index' : metrics_df[metrics_df['val_recall'] == max(metrics_df['val_recall'])].index[0]}

        return min_loss, max_accuracy, max_precision, max_recall
    
    def get_results_df_by_name(self, file_name, num_classes):
        file_path = f"{utils.base_dir}/GridSearchResults/{num_classes}_classes"
        loaded_df = pd.read_csv(file_path+'/'+file_name)
        return loaded_df
    