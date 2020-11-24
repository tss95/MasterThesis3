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

class DataHandler(LoadData):
    
    def __init__(self):
        super().__init__()
        self.label_dict = {'earthquake':0, 'noise':1, 'explosion':2, 'induced':3}
        
    def get_trace_shape_no_cast(self, ds):
        num_ds = len(ds)
        with h5py.File(ds[0][0], 'r') as dp:
            trace_shape = dp.get('traces').shape
        return num_ds, trace_shape[0], trace_shape[1]

    def path_to_trace(self, path):
        with h5py.File(path, 'r') as dp:
            trace_array = np.array(dp.get('traces'))
            info = np.array(dp.get('event_info'))
            info = json.loads(str(info))
        return trace_array, info
    
    def batch_to_trace(self, batch):
        path_array = batch[:,0]
        _, channels, datapoints = self.get_trace_shape_no_cast(batch)
        batch_trace = np.empty((len(batch), channels, datapoints))
        batch_info = []
        for idx, path in enumerate(path_array):
            batch_trace[idx] = self.path_to_trace(path)[0]
            batch_info.append(self.label_dict.get(batch[idx][1]))
        return batch_trace, np.array(batch_info)
    
    def transform_batch(self, scaler, batch_X):
        transformed_X = batch_X
        for i in range(len(batch_X)):
            transformed_X[i] = scaler.transform(batch_X[i])
        return transformed_X
    
    def detrend_highpass_batch_trace(self, batch_trace, detrend, use_highpass, highpass_freq = 0.1):
        output = batch_trace
        for idx, trace in enumerate(batch_trace):
            trace_BHE = Trace(data=trace[0])
            trace_BHN = Trace(data=trace[1])
            trace_BHZ = Trace(data=trace[2])
            stream = Stream([trace_BHE, trace_BHN, trace_BHZ])
            if detrend:
                stream.detrend('demean')
            if use_highpass:
                stream.taper(max_percentage=0.05, type='cosine')
                stream.filter('highpass', freq = highpass_freq)
            output[idx] = np.array(stream)
        return output
                      
    def convert_to_tensor(self, value, dtype_hint = None, name = None):
        tensor = tf.convert_to_tensor(value, dtype_hint, name)
        return tensor
    
    def detrend_trace(self, trace):
        trace_BHE = Trace(data=trace[0])
        trace_BHN = Trace(data=trace[1])
        trace_BHZ = Trace(data=trace[2])
        stream = Stream([trace_BHE, trace_BHN, trace_BHZ])
        stream.detrend('demean')
        return np.array(stream)
    
    def highpass_filter(self, trace, highpass_freq):
        trace_BHE = Trace(data=trace[0])
        trace_BHN = Trace(data=trace[1])
        trace_BHZ = Trace(data=trace[2])
        stream = Stream([trace_BHE, trace_BHN, trace_BHZ])
        stream.taper(max_percentage=0.05, type='cosine')
        stream.filter('highpass', freq = highpass_freq)
        return np.array(stream)
    
    def get_class_array(self, ds, num_classes = 3):
        class_array = np.zeros((len(path),num_classes))
        for idx, path, label in enumerate(ds):
            if label == "explosion":
                class_array[idx][0] = 1
            if label == "earthquake":
                class_array[idx][1] = 1
            if label == "noise":
                class_array[idx][2] = 1
        return class_array