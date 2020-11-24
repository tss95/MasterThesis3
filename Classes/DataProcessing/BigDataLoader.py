import h5py
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import sys

from .BigDataHandler import BigDataHandler
from .Balancer import Balancer

class BigDataLoader:
    def __init__(self, upsample = False, downsample = False, frac_diff = 1, test = False, seed = None):
        self.upsample = upsample
        self.downsample = downsample
        self.balance = self.upsample and self.downsample
        self.seed = seed
        self.test = test
        self.source_path = "F:\Thesis_ssd"
        
        #TODO: Make these two lines more generic
        self.filename = f"{self.source_path}\LargeDataset\merge.hdf5"
        self.csv_file = f"{self.source_path}\LargeDataset\merge.csv"
        
        self.info_file = self.parse_csv(self.csv_file)
        self.data_file = h5py.File(self.filename, 'r').get('data')
        
        self.event_names = list(self.data_file.keys())
        if self.test:
            np.random.shuffle(self.event_names)
            self.event_names = self.event_names[0:int(len(self.event_names)*0.02)]
        self.name_label = self.create_name_label_array(self.event_names)
        self.handler = BigDataHandler(self)
        self.balancer = Balancer(self, self.handler)
        
        if upsample or downsample:
            self.name_label = self.balancer.balance_dataset(self.name_label, downsample, upsample, frac_diff = frac_diff)
    
    def parse_csv(self, csv_file):
        col_names = pd.read_csv(csv_file, nrows=0).columns
        non_string_or_time = {
                      'receiver_latititude' : float,
                      'receiver_longitude' : float,
                      'p_weight' : float,
                      'p_travel_secs' : float,
                      'source_latitude' : float,
                      'source_longitude' : float,
                      'source_magnitude' : float,
                      'source_distance_deg' : float,
                      'source_distance_km' : float,
                      'back_azimuth_deg' : float,
                      'snr_db' : object,
                      'code_end_sample' : object}
        non_string_or_time.update({col: str for col in col_names if col not in non_string_or_time})
        df = pd.read_csv(self.csv_file, dtype = non_string_or_time)
        return df
           

    def get_label_by_name(self, event_name):
        # Fastest way to get label
        labels = {'EV': 'earthquake' , 'NO' : 'noise'}
        return labels[event_name.split('_')[-1]]


    def create_name_label_array(self, name_list):
        name_label = np.empty((len(name_list), 2), dtype='<U32')
        for idx, name in enumerate(name_list):
            name_label[idx] = [name, self.get_label_by_name(name)]
        return name_label

    def get_dataset_distribution(self, name_labels):
        labels = [x[1] for x in name_labels]
        uniques, counts = np.unique(labels, return_counts = True)
        return uniques, counts