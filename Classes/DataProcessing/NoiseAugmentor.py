import numpy as np
from tensorflow.keras import utils
import random
from obspy import Stream, Trace, UTCDateTime

from .LoadData import LoadData
from .DataHandler import DataHandler


class NoiseAugmentor(DataHandler):
    # TODO: Consider this: https://stackoverflow.com/questions/47324756/attributeerror-module-matplotlib-has-no-attribute-plot
    # How does SNR impact the use of this class???
    def __init__(self, ds, use_highpass, detrend, use_scaler, scaler, loadData, timeAug, highpass_freq = 0.1):
        super().__init__(loadData)
        self.loadData = loadData
        self.ds = ds
        self.use_highpass = use_highpass
        self.detrend = detrend
        self.use_scaler = use_scaler
        self.scaler = scaler
        self.timeAug = timeAug
        self.highpass_freq = highpass_freq
        if self.loadData.earth_explo_only or self.loadData.noise_earth_only or self.loadData.noise_not_noise:
            self.noise_ds = self.loadData.noise_ds
        else:
            self.noise_ds = self.get_noise(self.ds)
            
        self.noise_mean, self.noise_std = self.get_noise_mean_std(self.noise_ds, self.use_highpass, self.detrend, self.highpass_freq)
        
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
    
    
    
    def get_noise_mean_std(self, noise_ds, use_highpass, detrend, highpass_freq):
        noise_mean = 0
        noise_std = 0
        nr_noise = len(noise_ds)
        idx = 0
        for path, label, redundancy_index in noise_ds:
            if self.timeAug != None:
                X = self.batch_to_aug_trace(np.array([[path, label, redundancy_index]]), self.timeAug)[0][0]
            else:
                X = self.path_to_trace(path)[0]
            if self.use_highpass:
                X = self.detrend_highpass(X, detrend, use_highpass, highpass_freq)
            if self.use_scaler:
                X = self.scaler.transform(X)
            noise_mean += np.mean(X)
            noise_std += np.std(X)
            self.progress_bar(idx, nr_noise)
            idx += 1
        noise_mean = noise_mean/nr_noise
        noise_std = noise_std/nr_noise
        return noise_mean, noise_std

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
        return np.array(stream)

    def progress_bar(self, current, total, barLength = 20):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('Fitting noise progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')