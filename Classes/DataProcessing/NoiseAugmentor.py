import numpy as np
from keras.utils import np_utils
import random

from .LoadData import LoadData
from .DataHandler import DataHandler


class NoiseAugmentor(DataHandler):
    # TODO: Consider this: https://stackoverflow.com/questions/47324756/attributeerror-module-matplotlib-has-no-attribute-plot
    # How does SNR impact the use of this class???
    def __init__(self, ds, use_scaler, scaler):
        super().__init__()
        self.ds = ds
        self.use_scaler = use_scaler
        self.scaler = scaler
        self.noise_ds = self.get_noise(self.ds)
        self.noise_mean, self.noise_std = self.get_noise_mean_std(self.noise_ds, self.use_scaler, self.scaler)
        
    def create_noise(self, mean, std, sample_shape):
        noise = np.random.normal(mean, std, (sample_shape))
        return noise
    
    def batch_augment_noise(self, X, mean, std):
        noise = self.create_noise(mean, std, X.shape)
        X = X + noise
        return X
    
    def get_noise(self, ds):
        noise_ds = []
        for path, label in ds:
            if label == "noise":
                noise_ds.append([path,label])
        return np.array(noise_ds)

    def get_noise_mean_std(self, noise_ds, use_scaler, scaler):
        noise_mean = 0
        noise_std = 0
        for path, label in noise_ds:
            if use_scaler:
                X = scaler.transform(self.path_to_trace(path)[0])
                noise_mean += np.mean(X)
                noise_std += np.std(X)
            else:
                X = self.path_to_trace(path)[0]
                noise_mean += np.mean(X)
                noise_std += np.std(X)
        noise_mean = noise_mean/len(noise_ds)
        noise_std = noise_std/len(noise_ds)
        return noise_mean, noise_std

