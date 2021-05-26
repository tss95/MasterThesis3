import numpy as np
import pandas as pd
from tensorflow.keras import utils
from obspy import Stream, Trace, UTCDateTime
import time
import datetime

import os
import sys
classes_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3'
os.chdir(classes_dir)
#from Classes.DataProcessing.LoadData import LoadData
from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.DataProcessing.DataHandler import DataHandler
from Classes.DataProcessing.TimeAugmentor import TimeAugmentor
from Classes.DataProcessing.NoiseAugmentor import NoiseAugmentor
from Classes.Scaling.ScalerFitter import ScalerFitter
from Classes.Scaling.MinMaxScalerFitter import MinMaxScalerFitter
from Classes.Scaling.StandardScalerFitter import StandardScalerFitter
from Classes.Scaling.RobustScalerFitter import RobustScalerFitter
from Classes.Scaling.DataNormalizer import DataNormalizer

class RamLessLoader:
    """
    Class responsible for fitting preprocessing elements prior to use. Does not load data to RAM. Has not been used in the project, and so is not tested.

    PARAMETERS:
    ------------------------------------------------------------------------

    loadData: (object)           Fitted LoadData object.
    handler: (object)            DataHandler object. Holds functions which are relevant to the loading and handling of the recordings.
    use_time_augmentor: (bool)   Boolean for whether or not to use time augmentation. 
    use_noise_augmentor: (bool)  Boolean for whether or not to use noise augmentation.
    scaler_name: (str)           String representing the name of the scaler type to be used in the preprocessing.
    filter_name: (str)           Name of the digital filter to be used. Will use default filter values unless otherwised specified.
    band_min: (float)            Minimum frequency parameter for Bandpass filter
    band_max: (float)            Maximum frequency parameter for Bandpass filter
    highpass_freq: (float)       Corner frequency for Highpass filter.
    load_test_set: (bool)        Whether or not to load the test set.
    meier_load: (bool)           True if training/evaluating Meier et al.'s model.
    """




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
        self.num_classes = len(set(self.loadData.label_dict.values()))
        self.meier_load = meier_load

    
    def fit_timeAug(self, ds, dataset_name):
        timeAug = None
        if self.use_time_augmentor:
            timeAug = TimeAugmentor(self.handler, ds, dataset_name = dataset_name, seed = self.loadData.seed)
            timeAug.fit()
            print("\n")
        return timeAug
    
    def fit(self):
        _, num_channels, timesteps = self.handler.get_trace_shape_no_cast(self.loadData.train, self.use_time_augmentor)
        self.input_shape = (timesteps, num_channels)
        self.create_y()
        if self.loadData.noise_not_noise:
            return self.fit_noise_not_noise()
        if self.loadData.earth_explo_only:
            return self.fit_earth_explo_only()
        else:
            raise Exception("Loading to ram for this type of data has not been implemented.")

    def fit_scaler(self):
        if self.scaler_name == "normalize":
            print("No fitting necessary for normalize scalar.")
        if self.scaler_name != None and self.scaler_name != "normalize":
            num_samples = len(self.train_ds)
            for idx, path_label_red in enumerate(self.train_ds):
                self.progress_bar(idx + 1, num_samples, f"Fitting {self.scaler_name} scaler:")
                path = path_label_red[0]
                label = path_label_red[1]
                red_i = path_label_red[2]
                # Load event to RAM and apply time aug and filter
                loaded_trace, _ = self.timeAug_and_filter(self.train_timeAug, path, label, red_i)
                # Now to fit scaler:
                self.scaler.partial_fit_ramless(loaded_trace)
        self.scaler = self.scaler.scaler
            

    def fit_noiseAug(self, noise_ds, timeAug, scaler):
        noiseAug = NoiseAugmentor(self.loadData, None)
        noiseAug.get_noise_mean_std_ramless(noise_ds, timeAug, self, scaler)
        return noiseAug

    def fit_noise_not_noise(self):
        # This part is the same for RamLoader:
        start = time.time()
        self.train_timeAug = self.fit_timeAug(self.train_ds, "train")
        self.val_timeAug = self.fit_timeAug(self.val_ds, "validation")
        if self.load_test_set:
            self.test_timeAug = self.fit_timeAug(self.test_ds, "test")
        # This is where things start to change. We only need to care about the training set. Everything else will be handled in the generators.
        # We need to fit scaler and noiseAug seperately.
        self.scaler = self.get_scaler()
        if self.scaler is not None:
            self.fit_scaler()

        self.noiseAug = None
        if self.use_noise_augmentor:
            noise_ds = self.train_ds[self.train_ds[:,1] == "noise"]
            self.noiseAug = self.fit_noiseAug(noise_ds, self.train_timeAug, self.scaler)
        print("\n")
        end = time.time()
        print(f"Process took {datetime.timedelta(seconds=end-start)} seconds.")



    def fit_earth_explo_only(self):
        start = time.time()
        if self.use_time_augmentor:
            self.noise_timeAug = self.fit_timeAug(self.noise_ds, "noise set")
        self.train_timeAug = self.fit_timeAug(self.train_ds, "train")
        self.val_timeAug = self.fit_timeAug(self.val_ds, "validation")
        if self.load_test_set:
            self.test_timeAug = self.fit_timeAug(self.test_ds, "test")

        self.scaler = self.get_scaler()
        if self.scaler is not None:
            self.fit_scaler()
        self.noiseAug = None
        if self.use_noise_augmentor:
            self.noiseAug = self.fit_noiseAug(self.noise_ds, self.noise_timeAug, self.scaler)
        print("\n")
        end = time.time()
        print(f"Process took {datetime.timedelta(seconds=end-start)} seconds.")    
    
    def get_scaler(self):
        scaler = None
        if self.scaler_name == "minmax":
            scaler = MinMaxScalerFitter()
        elif self.scaler_name == "standard":
            scaler = StandardScalerFitter()
        elif self.scaler_name == "robust":
            scaler = RobustScalerFitter()
        elif self.scaler_name == "normalize":
            scaler = DataNormalizer()
        elif self.scaler_name != "minmax" or self.scaler_name != "standard" or self.scaler_name != "robust":
            raise Exception(f"{self.scaler_name} is not implemented.")
        print("\n")
        return scaler


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
    
    def timeAug_and_filter(self, timeAug, path, label, red_i):
        loaded_label = self.handler.label_dict.get(label)
        if self.filter_name != None or self.use_time_augmentor:
            if self.filter_name != None and self.use_time_augmentor:
                loaded_trace = timeAug.augment_event(path, red_i)
                info = self.handler.path_to_trace(path)[1]
                loaded_trace = self.apply_filter(loaded_trace, info, self.filter_name, highpass_freq = self.highpass_freq, band_min = self.band_min, band_max = self.band_max)
            if self.filter_name == None:
                loaded_trace = timeAug.augment_event(path, red_i)
            if not self.use_time_augmentor:
                loaded_trace, info = self.handler.path_to_trace(path)
                loaded_trace = self.apply_filter(loaded_trace, info, self.filter_name, highpass_freq = self.highpass_freq, band_min = self.band_min, band_max = self.band_max)
        else:
            loaded_trace = self.handler.path_to_trace(path)[0]
        return loaded_trace, loaded_label
    
    
    def scaler_transform_trace(self, trace):
        if self.scaler_name != None:
            if self.scaler_name != "normalize":
                trace = np.transpose(self.scaler.transform(np.transpose(trace)))
            else:
                trace = self.scaler.fit_transform_trace(trace)
        return trace

        

    
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
    
    def create_y(self):
        self.y_train = self.convert_labels(self.train_ds, self.num_classes, self.meier_load)
        self.y_val = self.convert_labels(self.val_ds, self.num_classes, self.meier_load)
        if self.load_test_set:
            self.y_test = self.convert_labels(self.test_ds, self.num_classes, self.meier_load)

    def convert_labels(self, ds, num_classes, meier_load):
        labels = ds[:,1]
        y = np.empty((len(ds), 1))
        for i in range(len(ds)):
            y[i] = self.handler.label_dict.get(labels[i])
        y = utils.to_categorical(y, self.num_classes, dtype = np.int8)
        if self.num_classes == 2 and not meier_load:
            y = y[:,1]
            y = np.reshape(y, (y.shape[0], 1))
        return y 


    def load_batch(self, batch, timeAug, batch_traces):
        for i in range(len(batch)):
            path, label, red_i = batch[i]
            batch_traces[i], _ = self.timeAug_and_filter(timeAug, path, label, red_i)
            batch_traces[i] = self.scaler_transform_trace(batch_traces[i])
        return batch_traces
        


    def progress_bar(self, current, total, text, barLength = 40):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('%s: [%s%s] %d %%' % (text, arrow, spaces, percent), end='\r')



