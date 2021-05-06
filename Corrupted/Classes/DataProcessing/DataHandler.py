import numpy as np
import pandas as pd
import json
import h5py
import matplotlib.pyplot as plt
from obspy import Stream, Trace, UTCDateTime
import os
import csv
import seaborn as sns
import time
import tables
import random

import tensorflow as tf
from .LoadData import LoadData
from .TimeAugmentor import TimeAugmentor

class DataHandler():
    
    def __init__(self, loadData):
        self.loadData = loadData
        self.label_dict = self.loadData.get_label_dict()
        
    def get_trace_shape_no_cast(self, ds, use_time_augmentor):
        num_ds = len(ds)
        if use_time_augmentor:
            return num_ds, 3, 6000
        else:
            with h5py.File(ds[0][0], 'r') as dp:
                trace_shape = dp.get('traces').shape
            return num_ds, trace_shape[0], trace_shape[1]

    def path_to_trace(self, path):
        with h5py.File(path, 'r') as dp:
            trace_array = np.array(dp.get('traces'))
            info = np.array(dp.get('event_info'))
            info = str(info)    
            info = info[2:len(info)-1]
            info = json.loads(info)
        return trace_array, info
    
    def batch_to_trace(self, batch):
        path_array = batch[:,0]
        _, channels, datapoints = self.get_trace_shape_no_cast(batch, False)
        batch_trace = np.empty((len(batch), channels, datapoints))
        batch_label = []
        for idx, path in enumerate(path_array):
            batch_trace[idx] = self.path_to_trace(path)[0]
            batch_label.append(self.label_dict.get(batch[idx][1]))
        return batch_trace, np.array(batch_label)
    
    def batch_to_aug_trace(self, batch, timeAug):
        path_array = batch[:,0]
        batch_trace = np.empty((len(batch), 3, 6000))
        batch_label = np.empty((len(batch), 1))
        for idx, path in enumerate(path_array):
            batch_trace[idx] = timeAug.augment_event(path, batch[idx][2])
            batch_label[idx] = self.label_dict.get(batch[idx][1])
        return batch_trace, batch_label
    
    def transform_batch(self, scaler, batch_X):
        transformed_X = batch_X
        for i in range(len(batch_X)):
            transformed_X[i] = scaler.transform(batch_X[i])
        return transformed_X
    
    def apply_filter(self, trace, info, filter_name, highpass_freq = 1.0, band_min = 2.0, band_max = 4.0):
        station = info['trace_stats']['station']
        channels = info['trace_stats']['channels']
        sampl_rate = info['trace_stats']['sampling_rate']
        starttime = info['trace_stats']['starttime']
        trace_BHE = Trace(data=trace[0], header ={'station' : station,
                                                  'channel' : channels[0],
                                                  'sampling_rate' : sampl_rate,
                                                  'starttime' : starttime})
        trace_BHN = Trace(data=trace[1], header ={'station' : station,
                                                  'channel' : channels[1],
                                                  'sampling_rate' : sampl_rate,
                                                  'starttime' : starttime})
        trace_BHZ = Trace(data=trace[2], header ={'station' : station,
                                                  'channel' : channels[2],
                                                  'sampling_rate' : sampl_rate,
                                                  'starttime' : starttime})
        stream = Stream([trace_BHE, trace_BHN, trace_BHZ])
        stream.detrend('demean')
        if filter_name == "highpass":
            stream.taper(max_percentage=0.05, type='cosine')
            stream.filter('highpass', freq = highpass_freq)
        if filter_name == "bandpass":
            stream.taper(max_percentage=0.05, type='cosine')
            stream.filter('bandpass', freqmin=band_min, freqmax=band_max)
        return np.array(stream)
                      
    def convert_to_tensor(self, value, dtype_hint = None, name = None):
        tensor = tf.convert_to_tensor(value, dtype_hint, name)
        return tensor
    

    