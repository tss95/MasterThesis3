
import numpy as np
import pandas as pd
import json
import h5py
import seaborn as sns
import os
import csv
import sys

base_dir = 'C:\Documents\Thesis_ssd\MasterThesis'
os.chdir(base_dir)

class LoadData():
    
    def __init__(self, num_classes = 3, isBalanced = True):
        self.isBalanced = isBalanced
        self.data_path = 'data_tord_may2020'
        self.csv_folder = 'csv_folder'
        self.num_classes = num_classes
        if self.num_classes == 3:
            self.full_data_csv = f'{base_dir}/{self.csv_folder}/balanced_csv_3_class.csv'
        else:
            self.full_data_csv = f'{base_dir}/{self.csv_folder}/balanced_csv_2_class.csv'
        if isBalanced:
            self.root_sub = 'balanced'
        else:
            self.root_sub = 'raw'
        self.train_csv = f'{base_dir}/{self.csv_folder}/{self.num_classes}_classes/{self.root_sub}/train_set.csv'
        self.val_csv = f'{base_dir}/{self.csv_folder}/{self.num_classes}_classes/{self.root_sub}/validation_set.csv'
        self.test_csv = f'{base_dir}/{self.csv_folder}/{self.num_classes}_classes/{self.root_sub}/test_set.csv'
    
    def getCsvs(self):
        return self.full_data_csv, self.train_csv, self.val_csv, self.test_csv
    
    def getDatasets(self, shuffle = False):
        self.full_ds = self.load_dataset(self.full_data_csv, shuffle) 
        self.train_ds = self.load_dataset(self.train_csv, shuffle)
        self.val_ds = self.load_dataset(self.val_csv, shuffle)
        self.test_ds = self.load_dataset(self.test_csv, shuffle)
        return self.full_ds, self.train_ds, self.val_ds, self.test_ds
    
    def load_dataset(self, data_csv, shuffle = False):
        columns = ["path", "label"]
        df = pd.read_csv(data_csv, names = columns)
        if shuffle: 
            df = df.sample(frac = 1)
        return df.values
        
        