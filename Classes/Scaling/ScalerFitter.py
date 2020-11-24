import numpy as np
import pandas as pd
import json
import h5py
import matplotlib.pyplot as plt
from obspy import Stream, Trace, UTCDateTime
import os
import csv
from keras.utils import np_utils
import seaborn as sns
import time
import tables
import random

import tensorflow as tf


class ScalerFitter():
    
        def __init__(self, train_ds, scaler):
            self.train_ds = train_ds
            self.scaler = scaler
            
        def subsample(self, ds, shuffle = False, subsample_rate = 0.2):
            num_samples, channels, timesteps = self.get_trace_shape_no_cast(ds)
            num_samples = int(num_samples*subsample_rate)
            if shuffle:
                np.random.shuffle(ds)
            subsample_X = np.empty((num_samples, channels, timesteps))
            subsample_y = np.empty((num_samples,1), dtype=np.dtype('U100'))
            for i in range(num_samples):
                subsample_X[i] = self.path_to_trace(ds[i][0])[0]
                subsample_y[i] = ds[i][1]
            return subsample_X, subsample_y
        
        def transform_subsample(self, ds, subsample_rate = 0.2, shuffle = False, detrend = False):
            subsamples_X, subsamples_y = self.subsample(ds, shuffle, subsample_rate)
            for i in range(len(subsamples_X)):
                subsamples_X[i] = self.scaler.transform(subsamples_X[i])
            return subsamples_X, subsamples_y
        
        
        def transform_sample(self, sample_X):
            return self.scaler.transform(sample_X)
                          
        def get_trace_shape_no_cast(self, ds):
            num_ds = len(ds)
            with h5py.File(ds[0][0], 'r') as dp:
                trace_shape = dp.get('traces').shape
            return num_ds, trace_shape[0], trace_shape[1]

        def path_to_trace(self, path):
            trace_array = np.empty((3,6001))
            with h5py.File(path, 'r') as dp:
                trace_array[:3] = dp.get('traces')
                info = np.array(dp.get('event_info'))
                info = json.loads(str(info))
            return trace_array, info
        
        def detrend_trace(self, trace):
            trace_BHE = Trace(data=trace[0])
            trace_BHN = Trace(data=trace[1])
            trace_BHZ = Trace(data=trace[2])
            stream = Stream([trace_BHE, trace_BHN, trace_BHZ])
            stream.detrend('demean')
            return np.array(stream)