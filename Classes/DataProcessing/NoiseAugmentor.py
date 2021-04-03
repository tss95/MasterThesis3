import numpy as np
import random
from Classes.DataProcessing.DataHandler import DataHandler
from Classes.DataProcessing.HelperFunctions import HelperFunctions

class NoiseAugmentor(DataHandler):
    # TODO: Consider this: https://stackoverflow.com/questions/47324756/attributeerror-module-matplotlib-has-no-attribute-plot
    # How does SNR impact the use of this class???

    def __init__(self, loadData, traces):
        super().__init__(loadData)
        self.loadData = loadData
        self.helper = HelperFunctions()
        self.noise_mean, self.noise_std = self.get_noise_mean_std_ram(traces)
        
    def create_noise(self, mean, std, sample_shape):
        noise = np.random.normal(mean, std, (sample_shape))
        return noise
    
    def batch_augment_noise(self, X, mean, std):
        noise = self.create_noise(mean, std, X.shape)
        return X + noise

    def get_noise_mean_std_ram(self, noise_traces):
        noise_mean = 0
        noise_std = 0
        nr_noise = len(noise_traces)
        for idx, trace in enumerate(noise_traces):
            noise_mean += np.mean(trace)
            noise_std += np.std(trace)
            self.helper.progress_bar(idx+1, nr_noise ,"Fitting noise augmentor")
        noise_mean = noise_mean/nr_noise
        noise_std = noise_std/nr_noise
        return noise_mean, noise_std

    """
    def __init__(self, ds, filter_name, scaler_name, scaler, loadData, timeAug, band_min = 2.0, band_max = 4.0, highpass_freq = 1):
        super().__init__(loadData)
        self.loadData = loadData
        self.ds = ds
        self.filter_name = filter_name
        self.scaler_name = scaler_name
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
            
        self.noise_mean, self.noise_std = self.get_noise_mean_std_ram(self.noise_ds)
        
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
            if self.scaler_name != None:
                if self.scaler_name != "robust":
                    X = self.scaler.transform(X)
                else:
                    X = self.scaler.fit_transform_trace(X)
            noise_mean += np.mean(X)
            noise_std += np.std(X)
            idx += 1
            self.helper.progress_bar(idx, nr_noise ,"Fitting noise augmentor")
            
        noise_mean = noise_mean/nr_noise
        noise_std = noise_std/nr_noise
        return noise_mean, noise_std

    
            
"""

    