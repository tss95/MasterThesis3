import numpy as np
import pandas as pd
from tensorflow.keras import utils
from obspy import Stream, Trace, UTCDateTime

import os
import sys
classes_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3'
os.chdir(classes_dir)
from Classes.DataProcessing.LoadData import LoadData
from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.DataProcessing.DataHandler import DataHandler
from Classes.DataProcessing.DataGenerator import DataGenerator
from Classes.DataProcessing.TimeAugmentor import TimeAugmentor
from Classes.DataProcessing.NoiseAugmentor import NoiseAugmentor
from Classes.Scaling.ScalerFitter import ScalerFitter
from Classes.Scaling.MinMaxScalerFitter import MinMaxScalerFitter
from Classes.Scaling.StandardScalerFitter import StandardScalerFitter

class RamLoader:
    def __init__(self, loadData, handler, use_time_augmentor = False, use_noise_augmentor = False, use_scaler = False, use_minmax = False, 
                use_highpass = False, highpass_freq = 0.1, detrend = False, load_test_set = False):
        self.loadData = loadData
        self.handler = handler
        self.full_ds, self.train_ds, self.val_ds, self.test_ds = self.loadData.get_datasets()
        self.use_time_augmentor = use_time_augmentor
        self.use_noise_augmentor = use_noise_augmentor
        self.use_scaler = use_scaler
        self.use_minmax = use_minmax
        self.use_highpass = use_highpass
        self.highpass_freq = highpass_freq
        self.detrend = detrend
        self.load_test_set = load_test_set
        self.num_classes = len(set(handler.loadData.label_dict.values()))

    

    """
    Time, filter, scaler, noise:
    1. timeaug
    2. filter
    3 scaler
    4. noise
    Time, scaler, noise:
    1. timeaug
    2. scaler
    3. noise
    filter, scaler, noise:
    1. filter
    2. scaler
    3. noise
    

    PROBLEMS: 
    - Scaling is different if a filter is used. If a filter is used, we need to load the data into ram,
    detrend/filter it, as well as time augment it, prior to fitting the scaler.

    - Initially this class was initiated once for each dataset. Since we are fitting timeAug and scalers
    within this class now, we need to do everything in on initiation of the class

    These solutions should be resolved.
    """
    
    def fit_timeAug(self):
        timeAug = None
        if self.use_time_augmentor:
            if self.loadData.earth_explo_only:
                full_and_noise_ds = np.concatenate((self.loadData.full_ds, self.loadData.noise_ds))
                timeAug = TimeAugmentor(self.handler, full_and_noise_ds, seed = self.loadData.seed)
            else:
                timeAug = TimeAugmentor(self.handler, self.full_ds, seed = self.loadData.seed)
            timeAug.fit()
            print("\n")
        return timeAug
    
    def fit_scaler(self, traces):
        scaler = None
        if self.use_scaler:
            if self.use_minmax:
                scaler = MinMaxScalerFitter(self.train_ds, self.timeAug).fit_scaler_ram(traces)
            else:
                scaler = StandardScalerFitter(self.train_ds, self.timeAug).fit_scaler_ram(traces)
        print("\n")
        return scaler

    def fit_noiseAug(self, noise_ds, scaler, loadData, timeAug):
        noiseAug = None
        use_scaler = False
        if scaler != None:
            use_scaler = True
        if self.use_noise_augmentor:
            noiseAug = NoiseAugmentor(noise_ds, self.use_highpass, self.detrend, use_scaler, scaler, loadData, timeAug, self.highpass_freq)
        return noiseAug
            

    
    def stage_one_load(self, ds):
        loaded_label = np.empty((len(ds), 1))
        loaded_trace = np.empty((self.handler.get_trace_shape_no_cast(ds, self.use_time_augmentor)))
        num_events = len(ds)
        for i in range(num_events):
            self.progress_bar_1(i+1, num_events)
            loaded_label[i] = self.handler.label_dict.get(ds[i][1])
            # timeAug, highpass and detrend.
            if (self.use_highpass or self.detrend) or self.use_time_augmentor:
                if (self.use_highpass or self.detrend) and self.use_time_augmentor:
                    loaded_trace[i] = self.timeAug.augment_event(ds[i][0], ds[i][2])
                    loaded_trace[i] = self.detrend_highpass(loaded_trace[i], self.detrend, self.use_highpass, self.highpass_freq)
                if not (self.use_highpass or self.detrend):
                    loaded_trace[i] = self.timeAug.augment_event(ds[i][0], ds[i][2])
                if not self.use_time_augmentor:
                    loaded_trace[i] = self.handler.path_to_trace(ds[i][0])
                    loaded_trace[i] = self.detrend_highpass(loaded_trace[i], self.detrend, self.use_highpass, self.highpass_freq)
            else:
                loaded_trace[i] = self.handler.path_to_trace(ds[i][0])
        print("\n")
        return loaded_trace, loaded_label
    
    def stage_two_load(self, traces, labels, is_lstm, num_channels):
        num_samples = traces.shape[0]
        if self.use_scaler:
            for i in range(num_samples):
                self.progress_bar_2(i+1, num_samples)
                traces[i] = self.scaler.transform(traces[i])
            print("\n")
        traces = traces[:][:,0:num_channels]
        if is_lstm:
            traces = np.reshape(traces, (traces.shape[0], 
                                         traces.shape[2], 
                                         traces.shape[1]))
        labels = utils.to_categorical(labels, self.num_classes, dtype=np.int8)
        if self.num_classes == 2:
            labels = labels[:,1]
            labels = np.reshape(labels, (labels.shape[0],1))
        return traces


    def load_to_ram(self, is_lstm, num_channels = 3):
        # Starting with fitting potential time augmentor
        self.timeAug = self.fit_timeAug()
        # Step one, load traces and apply time augmentation and/or detrend/highpass
        train_trace, train_label = self.stage_one_load(self.train_ds)
        val_trace, val_label = self.stage_one_load(self.val_ds)
        if self.load_test_set:
            test_trace, test_label = self.stage_one_load(self.test_ds)
        # The scaler is dependent on timeAug and highpasss/detrend, so must now fit the scaler:
        self.scaler = self.fit_scaler(train_trace)
        # Using the fitted scaler, we transform the traces:
        train_trace = self.stage_two_load(train_trace, train_label, is_lstm, num_channels)
        val_trace = self.stage_two_load(val_trace, val_label, is_lstm, num_channels)
        if self.load_test_set:
            test_trace = self.stage_two_load(test_trace, test_label, is_lstm, num_channels)
        self.noiseAug = self.fit_noiseAug(self.loadData.noise_ds, self.scaler, self.loadData, self.timeAug)
        print("Completed loading to RAM")
        if self.load_test_set:
            return train_trace, train_label, val_trace, val_label, test_trace, test_label, self.timeAug, self.scaler, self.noiseAug
        return train_trace, train_label, val_trace, val_label, self.timeAug, self.scaler, self.noiseAug
    
    
    def detrend_highpass(self, trace, detrend, use_highpass, highpass_freq):
        trace_BHE = Trace(data=trace[0])
        trace_BHN = Trace(data=trace[1])
        trace_BHZ = Trace(data=trace[2])
        stream = Stream([trace_BHE, trace_BHN, trace_BHZ])
        if detrend:
            stream.detrend('demean')
        if use_highpass:
            stream.taper(max_percentage=0.05, type='cosine')
            stream.filter('highpass', freq = highpass_freq)
            #stream.filter('bandpass', freqmin = 3.0, freqmax = 5.0)
        return np.array(stream)

    def progress_bar_1(self, current, total, barLength = 40):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('Stage 1 loading to RAM: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

    def progress_bar_2(self, current, total, barLength = 40):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('Stage 2 loading to RAM: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')


