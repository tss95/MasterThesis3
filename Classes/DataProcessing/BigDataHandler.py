import h5py
import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split


class BigDataHandler():
    
    def __init__(self, data_loader):
        self.loader = data_loader
        self.label_dict = {'earthquake':0, 'noise': 1}
        self.source_path = self.loader.source_path
        self.seed = self.loader.seed
    
    def create_train_val_test(self, name_label, val_test_size = 0.1, val_test_prop = 0.5, seed = None, shuffle = False):
        train, val_test = train_test_split(name_label, test_size = val_test_size, random_state = seed, shuffle = shuffle)
        val, test = train_test_split(val_test, test_size = val_test_prop, random_state = seed, shuffle = shuffle)
        return train, val, test
    
    def name_to_trace(self, name):
        return np.transpose(self.loader.data_file.get(name)[:])
    
    def csv_to_trace_label(self, data_file, info_file, index):
        name = info_file['trace_name'][index]
        event = data_file.get('data').get(name)[:]
        label = self.get_label_by_name(name)
        return event, label
    
    def get_csv_row_by_name(self, event_name):
        return self.loader.df_csv[self.loader.df_csv['trace_name'] == event_name].values
    
    def get_trace_shape(self, df):
        some_name = df[0][0]
        trace = self.name_to_trace(some_name)
        num_channels, num_timesteps = trace.shape
        return num_channels, num_timesteps
    
    def batch_to_trace_binary_label(self, batch):
        names = batch[:,0]
        labels = batch[:,1]
        batch_trace = np.empty((len(batch), 3, 6000))
        batch_info = np.empty((len(batch), 1))
        for idx, name in enumerate(names):
            batch_trace[idx] = self.name_to_trace(name)
            batch_info[idx] = self.label_dict.get(labels[idx])
        return batch_trace, batch_info
        
    def transform_batch(self, scaler, batch_X):
        transformed_X = batch_X
        for i in range(len(batch_X)):
            transformed_X[i] = scaler.transform(batch_X[i])
        return transformed_X     
        