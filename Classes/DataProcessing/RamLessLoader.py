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
    This class will be used in two cases: 
        1. When training a model with more data than RAM capacity
        2. When preprocessing the test data, two different versions of this class needs to be fitted. Will likely be fitted with a large amount of data.
             - The test data will likely be treated as validation data for this class. 
    
    In essence, this class will function as the holder of preprocessing variables and objects, and will be passed into the generators in some way.


    The fitting of the time augmentor will be the same as the orginal RamLoader.
    Due to dependency issues, I need to consider what is the most efficient way of doing this:
     - Time aug and the filters are independent of everything.
     - Normalize scaler does not fit to the data, and will be handled directly in the generator.
     - The other scalers will need:
        1. Load each datapoint
        2. Apply time aug
        3. Apply filter (if necessary)
        4. partial_fit
        5. Returned fitted scaler
     - NoiseAug needs:
        1. Load each datapoint
        2. Apply time aug
        3. Apply filter
        4. Apply scaler
        5. fit noiseAug
        6. Return noiseAug

    Observations:
    For noise-not-nosie:
     - other than time-aug, every other preprocessing step which requires fitting, can be done partially and exclusively on the training set. This way theres no repeated loading of the data.
     - This is not entierly true. The scaler needs to be fitted completely before actually scaling any data.
     - The validation and test data will all be transformed in the generators

     For earth explo:
      - We can fit time aug the same way we already do.
      - Then we can fit the training set preprocessors.
      - Finally we can fit the noise augmentor
    """
    
    def fit_timeAug(self, ds, dataset_name):
        timeAug = None
        if self.use_time_augmentor:
            timeAug = TimeAugmentor(self.handler, ds, dataset_name = dataset_name, seed = self.loadData.seed)
            timeAug.fit()
            print("\n")
        return timeAug
    
    def fit(self):
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
            noise_ds = self.train_ds[self.train_ds[:,] == "noise"]
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
            scaler = MinMaxScalerFitter().scaler
        elif self.scaler_name == "standard":
            scaler = StandardScalerFitter().scaler
        elif self.scaler_name == "robust":
            scaler = RobustScalerFitter().scaler
        elif self.scaler_name == "normalize":
            scaler = DataNormalizer().scaler
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
    


    def preprocess_data(self, path_label_red, timeAug, batch_traces, batch_labels):
        for i in range(path_label_red):
            path, label, red_i = path_label_red[i]
            batch_traces[i], batch_labels[i] = self.timeAug_and_filter(timeAug, path, label, red_i)
            batch_traces[i] = self.scaler_transform_trace(batch_traces[i])
        return batch_traces, batch_labels
        


    def progress_bar(self, current, total, text, barLength = 40):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('%s: [%s%s] %d %%' % (text, arrow, spaces, percent), end='\r')



