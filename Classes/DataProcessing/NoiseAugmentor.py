import numpy as np
from tensorflow.keras import utils
import random
from obspy import Stream, Trace, UTCDateTime

from .LoadData import LoadData
from .DataHandler import DataHandler
from .HelperFunctions import HelperFunctions


class NoiseAugmentor(DataHandler):
    # TODO: Consider this: https://stackoverflow.com/questions/47324756/attributeerror-module-matplotlib-has-no-attribute-plot
    # How does SNR impact the use of this class???
    def __init__(self, ds, filter_name, use_scaler, scaler, loadData, timeAug, band_min = 2.0, band_max = 4.0, highpass_freq = 1):
        super().__init__(loadData)
        self.loadData = loadData
        self.ds = ds
        self.filter_name = filter_name
        self.use_scaler = use_scaler
        self.scaler = scaler
        self.timeAug = timeAug
        self.band_min = band_min
        self.band_max = band_max
        self.highpass_freq = highpass_freq
        self.helper = HelperFunctions()
        if self.loadData.earth_explo_only or self.loadData.noise_earth_only or self.loadData.noise_not_noise:
            self.noise_ds = self.loadData.noise_ds
        else:
            self.noise_ds = self.get_noise(self.ds)
            
        self.noise_mean, self.noise_std = self.get_noise_mean_std(self.noise_ds)
        
    def create_noise(self, mean, std, sample_shape):
        noise = np.random.normal(mean, std, (sample_shape))
        return noise
    
    def batch_augment_noise(self, X, mean, std):
        noise = self.create_noise(mean, std, X.shape)
        X = X + noise
        return X
    

    def get_noise(self, ds):
        noise_ds = ds[ds[:,1] == "noise"]
        return np.array(noise_ds)
    """
    def get_noise_mean_std(self, noise_ds, use_scaler, scaler):
        noise_mean = 0
        noise_std = 0
        nr_noise = len(noise_ds)
        idx = 0
        for path, label, redundancy_index in noise_ds:
            if self.timeAug != None and use_scaler:
                X = scaler.transform(self.batch_to_aug_trace(np.array([[path, label, redundancy_index]]), self.timeAug)[0][0])
            elif use_scaler:
                X = scaler.transform(self.path_to_trace(path)[0])
                noise_mean += np.mean(X)
                noise_std += np.std(X)
            elif self.timeAug != None:
                X = self.batch_to_aug_trace([path, label, redundancy_index], self.timeAug)[0][0]
                noise_mean += np.mean(X)
                noise_std += np.std(X)
            else:
                X = self.path_to_trace(path)[0]
                noise_mean += np.mean(X)
                noise_std += np.std(X)
            self.progress_bar(idx, nr_noise)
            idx += 1
        noise_mean = noise_mean/len(noise_ds)
        noise_std = noise_std/len(noise_ds)
        return noise_mean, noise_std
    """
    
    
    
    def get_noise_mean_std(self, noise_ds):
        noise_mean = 0
        noise_std = 0
        nr_noise = len(noise_ds)
        idx = 0
        for path, label, redundancy_index in noise_ds:
            if self.timeAug != None:
                X = self.batch_to_aug_trace(np.array([[path, label, redundancy_index]]), self.timeAug)[0][0]
            else:
                X = self.path_to_trace(path)[0]
            if self.filter_name != None:
                info = self.path_to_trace(path)[1]
                X = self.apply_filter(X, info, self.filter_name, highpass_freq = self.highpass_freq, band_min = self.band_min, band_max = self.band_max)
            if self.use_scaler:
                X = self.scaler.transform(X)
            noise_mean += np.mean(X)
            noise_std += np.std(X)
            self.helper.progress_bar(idx, nr_noise ,"Fitting noise augmentor")
            idx += 1
        noise_mean = noise_mean/nr_noise
        noise_std = noise_std/nr_noise
        return noise_mean, noise_std

    