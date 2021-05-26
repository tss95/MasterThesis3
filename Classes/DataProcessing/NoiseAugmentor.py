import numpy as np
import random
from Classes.DataProcessing.DataHandler import DataHandler
from Classes.DataProcessing.HelperFunctions import HelperFunctions

class NoiseAugmentor(DataHandler):
    """
    The class responsible for augmenting noise onto the waveforms.

    PARAMETERS:
    -------------------------------------------------------------------------------------------
    loadData: (object)           Fitted LoadData object.
    traces: (np.array)           Optional. If not None, then the augmentor will fit the already loaded waveforms. If not, the user needs to call get_noise_mean_std_ramless to fit.
    """

    def __init__(self, loadData, traces):
        super().__init__(loadData)
        self.loadData = loadData
        self.helper = HelperFunctions()
        if traces is not None:
            self.noise_mean, self.noise_std = self.get_noise_mean_std_ram(traces)
        else:
            print("RAM-less noise augmentor initiated.")
            print("\n")
        
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

    def get_noise_mean_std_ramless(self, noise_ds, timeAug, ramLessLoader, scaler):
        noise_mean = 0
        noise_std = 0
        nr_noise = len(noise_ds)
        for i in range(nr_noise):
            self.helper.progress_bar(i + 1, nr_noise, "Fitting noise augmentor")
            loaded_trace, _ = ramLessLoader.timeAug_and_filter(timeAug, noise_ds[i][0], noise_ds[i][1], noise_ds[i][2])
            loaded_trace = ramLessLoader.scaler_transform_trace(loaded_trace)
            noise_mean += np.mean(loaded_trace)
            noise_std += np.std(loaded_trace)
        self.noise_mean = noise_mean/nr_noise
        self.noise_std = noise_std/nr_noise


    