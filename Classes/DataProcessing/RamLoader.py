import numpy as np
import pandas as pd
from tensorflow.keras import utils
from obspy import Stream, Trace, UTCDateTime
import time

import os
import sys
classes_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3'
os.chdir(classes_dir)
#from Classes.DataProcessing.LoadData import LoadData
from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.DataProcessing.DataHandler import DataHandler
from Classes.DataProcessing.DataGenerator import DataGenerator
from Classes.DataProcessing.TimeAugmentor import TimeAugmentor
from Classes.DataProcessing.NoiseAugmentor import NoiseAugmentor
from Classes.Scaling.ScalerFitter import ScalerFitter
from Classes.Scaling.MinMaxScalerFitter import MinMaxScalerFitter
from Classes.Scaling.StandardScalerFitter import StandardScalerFitter
from Classes.Scaling.RobustScalerFitter import RobustScalerFitter

class RamLoader:
    def __init__(self, loadData, handler, use_time_augmentor = False, use_noise_augmentor = False, scaler_name = None, 
                filter_name = None, band_min = 2.0, band_max = 4.0, highpass_freq = 0.1, load_test_set = False, meier_load = False):
        self.loadData = loadData
        self.handler = handler
        self.train_ds, self.val_ds, self.test_ds = self.loadData.get_datasets()
        self.noise_ds = self.loadData.noise_ds
        self.use_time_augmentor = use_time_augmentor
        self.use_noise_augmentor = use_noise_augmentor
        self.scaler_name = scaler_name
        self.filter_name = filter_name
        self.band_min = band_min
        self.band_max = band_max
        self.highpass_freq = highpass_freq
        self.load_test_set = load_test_set
        self.num_classes = len(set(handler.loadData.label_dict.values()))
        self.meier_load = meier_load

    

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
    
    def fit_timeAug(self, ds, dataset_name):
        timeAug = None
        if self.use_time_augmentor:
            timeAug = TimeAugmentor(self.handler, ds, dataset_name = dataset_name, seed = self.loadData.seed)
            timeAug.fit()
            print("\n")
        return timeAug
    
    
    
    def load_to_ram(self):
        start = time.time()
        # Starting with fitting potential time augmentor
        self.train_timeAug = self.fit_timeAug(self.train_ds, "train")
        self.val_timeAug = self.fit_timeAug(self.val_ds, "validation")
        if self.load_test_set:
            self.test_timeAug = self.fit_timeAug(self.test_ds, "test")
            
        # If we want to use noise augmentor while we arent classifying noise,
        # we need to fit time augmentor for the noise samples.
        if self.loadData.earth_explo_only and self.use_noise_augmentor:
            self.noise_timeAug = self.fit_timeAug(self.noise_ds, "noise set")
            
        # Step one, load traces and apply time augmentation and/or detrend/highpass
        train_trace, train_label = self.stage_one_load(self.train_ds, self.train_timeAug, 0)
        val_trace, val_label = self.stage_one_load(self.val_ds, self.val_timeAug, 1)
        if self.load_test_set:
            test_trace, test_label = self.stage_one_load(self.test_ds, self.test_timeAug, 2)
        
        
        # The scaler is dependent on timeAug and highpasss/detrend, so must now fit the scaler:
        self.scaler = self.fit_scaler(train_trace, self.train_timeAug)
        
        # When using noise augmentor and earth_explo_only, a scaler for the noise data needs to be fitted.
        # This is temporarily memory expensive with the current solution.
        if self.use_noise_augmentor and self.loadData.earth_explo_only:
            noise_trace, _ = self.stage_one_load(self.noise_ds, self.noise_timeAug, 3)
            self.noise_scaler = self.fit_scaler(noise_trace, self.noise_timeAug)
            if self.scaler_name == "robust":
                for idx, trace in enumerate(noise_trace):
                    trace = self.noise_scaler.fit_transform(trace)
            self.noiseAug = self.fit_noiseAug(self.loadData, noise_trace)
            del noise_trace
            
        
        # Using the fitted scaler, we transform the traces:
        train_trace, train_label = self.stage_two_load(train_trace, train_label, 0, self.meier_load)
        val_trace, val_label = self.stage_two_load(val_trace, val_label, 1, self.meier_load)
        if self.load_test_set:
            test_trace = self.stage_two_load(test_trace, test_label, 2, self.meier_load)
        
        if self.use_noise_augmentor and not self.loadData.earth_explo_only:
            # Need to get only the noise traces:
            if (self.meier_load and self.num_classes == 2) or self.num_classes > 2:
                noise_indexes = np.where(train_label[:,self.loadData.label_dict["noise"]] == 1)
            if not self.meier_load and self.num_classes == 2:
                noise_indexes = np.where(train_label == self.loadData.label_dict["noise"])
            noise_traces = train_trace[noise_indexes]
            self.noiseAug = self.fit_noiseAug(self.loadData, noise_traces)
        print("\n")
        print("Completed loading to RAM")
        end = time.time()
        print(f"Process took {int((end-start))} seconds.")
        if self.load_test_set:
            return train_trace, train_label, val_trace, val_label, test_trace, test_label, self.noiseAug
        return train_trace, train_label, val_trace, val_label, self.noiseAug
    
    
    
    
    
    def fit_scaler(self, traces, timeAug):
        scaler = None
        if self.scaler_name != None:
            if self.scaler_name == "minmax":
                scaler = MinMaxScalerFitter(self.train_ds, timeAug)
                scaler.fit_scaler_ram(traces)
                scaler = scaler.scaler
            elif self.scaler_name == "standard":
                scaler = StandardScalerFitter(self.train_ds, timeAug)
                scaler.fit_scaler_ram(traces)
                scaler = scaler.scaler
            elif self.scaler_name == "robust":
                scaler = RobustScalerFitter(self.train_ds, timeAug)
                print("Fit process of robust scaler skipped as unecessary.")
            elif self.scaler_name != "minmax" or self.scaler_name != "standard" or self.scaler_name != "robust":
                raise Exception(f"{self.scaler_name} is not implemented.")
            print("\n")
        return scaler

    def fit_noiseAug(self, loadData, noise_traces):
        noiseAug = None
        if self.use_noise_augmentor:
            noiseAug = NoiseAugmentor(loadData, noise_traces)
        return noiseAug
            
    def get_substage(self, substage):
        if substage == 0:
            return "training set"
        if substage == 1:
            return "validation set"
        if substage == 2:
            return "test set"
        if substage == 3:
            return "noise set"
        return ""
    
    def stage_one_load(self, ds, timeAug, substage):
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
                    loaded_trace[i] = timeAug.augment_event(ds[i][0], ds[i][2])
                    info = self.handler.path_to_trace(ds[i][0])[1]
                    loaded_trace[i] = self.apply_filter(loaded_trace[i], info, self.filter_name, highpass_freq = self.highpass_freq, band_min = self.band_min, band_max = self.band_max)
                if self.filter_name == None:
                    loaded_trace[i] = timeAug.augment_event(ds[i][0], ds[i][2])
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
    
    def stage_two_load(self, traces, labels, substage, meier_load):
        num_samples = traces.shape[0]
        bar_text = self.stage_two_text(substage)
        if self.scaler_name != None:
            if self.scaler_name != "robust":
                for i in range(num_samples):
                    self.progress_bar(i+1, num_samples, bar_text)
                    traces[i] = self.scaler.transform(traces[i])
                print("\n")
            else:
                for i in range(num_samples):
                    self.progress_bar(i+1, num_samples, bar_text)
                    traces[i] = self.scaler.fit_transform_trace(traces[i])
                print("\n")

        labels = utils.to_categorical(labels, self.num_classes, dtype=np.int8)
        if self.num_classes == 2 and not meier_load:
            labels = labels[:,1]
            labels = np.reshape(labels, (labels.shape[0],1))
        return traces, labels
    
    def stage_two_text(self, substage):
        bar_text = f"Stage two loading {self.get_substage(substage)}, labels"
        if self.scaler_name != None:
            bar_text = bar_text + f" and {self.scaler_name} scaler"

        return bar_text
        

    
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



