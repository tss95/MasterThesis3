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
                filter_name = None, band_min = 2.0, band_max = 4.0, highpass_freq = 0.1, load_test_set = False):
        self.loadData = loadData
        self.handler = handler
        self.full_ds, self.train_ds, self.val_ds, self.test_ds = self.loadData.get_datasets()
        self.use_time_augmentor = use_time_augmentor
        self.use_noise_augmentor = use_noise_augmentor
        self.use_scaler = use_scaler
        self.use_minmax = use_minmax
        self.filter_name = filter_name
        self.band_min = band_min
        self.band_max = band_max
        self.highpass_freq = highpass_freq
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
            noiseAug = NoiseAugmentor(noise_ds, self.filter_name, use_scaler, scaler, loadData, timeAug, 
                                      band_min = self.band_min, band_max = self.band_max, 
                                      highpass_freq = self.highpass_freq)
        return noiseAug
            
    def get_substage(self, substage):
        if substage == 0:
            return "training set"
        if substage == 1:
            return "validation set"
        if substage == 2:
            return "test set"
        return ""
    
    def stage_one_load(self, ds, substage):
        loaded_label = np.empty((len(ds), 1))
        loaded_trace = np.empty((self.handler.get_trace_shape_no_cast(ds, self.use_time_augmentor)))
        num_events = len(ds)
        bar_text =  self.stage_one_text(substage)
        for i in range(num_events):
            self.progress_bar(i+1, num_events, bar_text)
            loaded_label[i] = self.handler.label_dict.get(ds[i][1])
            # timeAug, highpass and detrend.
            if self.filter_name != None or self.use_time_augmentor:
                if self.filter_name != None and self.use_time_augmentor:
                    loaded_trace[i] = self.timeAug.augment_event(ds[i][0], ds[i][2])
                    info = self.handler.path_to_trace(ds[i][0])[1]
                    loaded_trace[i] = self.apply_filter(loaded_trace[i], info, self.filter_name, highpass_freq = self.highpass_freq, band_min = self.band_min, band_max = self.band_max)
                if self.filter_name == None:
                    loaded_trace[i] = self.timeAug.augment_event(ds[i][0], ds[i][2])
                if not self.use_time_augmentor:
                    loaded_trace[i] = self.handler.path_to_trace(ds[i][0])[0]
                    info = self.handler.path_to_trace(ds[i][0])[1]
                    loaded_trace[i] = self.apply_filter(loaded_trace[i], info, self.filter_name, highpass_freq = self.highpass_freq, band_min = self.band_min, band_max = self.band_max)
            else:
                loaded_trace[i] = self.handler.path_to_trace(ds[i][0])[0]
        print("\n")
        return loaded_trace, loaded_label
    
    def stage_one_text(self, substage):
        bar_text = f"Stage one loading {self.get_substage(substage)}"
        if self.filter_name != None or self.use_time_augmentor:
            bar_text = bar_text + ", "
            if self.filter_name != None:
                bar_text += self.filter_name
            if self.filter_name != None and self.use_time_augmentor:
                bar_text += " and "
            if self.use_time_augmentor:
                bar_text += "timeAug"
        return bar_text
    
    def stage_two_load(self, traces, labels, is_lstm, num_channels, substage):
        num_samples = traces.shape[0]
        bar_text = self.stage_two_text(substage)
        if self.use_scaler:
            for i in range(num_samples):
                self.progress_bar(i+1, num_samples, bar_text)
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
    
    def stage_two_text(self, substage):
        bar_text = f"Stage two loading {self.get_substage(substage)}, labels"
        if self.use_scaler:
            if self.use_minmax:
                bar_text = bar_text + " and minmax"
            else:
                bar_text = bar_text + " and sscaler"
        return bar_text
        


    def load_to_ram(self, is_lstm, num_channels = 3):
        # Starting with fitting potential time augmentor
        self.timeAug = self.fit_timeAug()
        # Step one, load traces and apply time augmentation and/or detrend/highpass
        train_trace, train_label = self.stage_one_load(self.train_ds, 0)
        val_trace, val_label = self.stage_one_load(self.val_ds, 1)
        if self.load_test_set:
            test_trace, test_label = self.stage_one_load(self.test_ds, 2)
        # The scaler is dependent on timeAug and highpasss/detrend, so must now fit the scaler:
        self.scaler = self.fit_scaler(train_trace)
        # Using the fitted scaler, we transform the traces:
        train_trace = self.stage_two_load(train_trace, train_label, is_lstm, num_channels, 0)
        val_trace = self.stage_two_load(val_trace, val_label, is_lstm, num_channels, 1)
        if self.load_test_set:
            test_trace = self.stage_two_load(test_trace, test_label, is_lstm, num_channels, 2)
        self.noiseAug = self.fit_noiseAug(self.loadData.noise_ds, self.scaler, self.loadData, self.timeAug)
        print("Completed loading to RAM")
        if self.load_test_set:
            return train_trace, train_label, val_trace, val_label, test_trace, test_label, self.timeAug, self.scaler, self.noiseAug
        return train_trace, train_label, val_trace, val_label, self.timeAug, self.scaler, self.noiseAug
    
    
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

    def progress_bar(self, current, total, text, barLength = 40):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('%s: [%s%s] %d %%' % (text, arrow, spaces, percent), end='\r')



