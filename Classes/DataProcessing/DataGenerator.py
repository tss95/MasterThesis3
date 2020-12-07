import numpy as np
from keras.utils import np_utils
import random
from sklearn.preprocessing import LabelEncoder

from .LoadData import LoadData
from .DataHandler import DataHandler
from .NoiseAugmentor import NoiseAugmentor

class DataGenerator(DataHandler):
    
    def __init__(self, loadData):
        super().__init__(loadData)
        self.num_classes = len(set(loadData.label_dict.values()))
   
   
    def data_generator(self, ds, batch_size, detrend = False, use_scaler = False, scaler = None, 
                       use_time_augmentor = True, timeAug = None, use_noise_augmentor = False, 
                       noiseAug = None, use_highpass = False, highpass_freq = 0.49):
        
        num_samples, channels, timesteps = self.get_trace_shape_no_cast(ds, use_time_augmentor)
        num_samples = len(ds)
        while True:
            for offset in range(0, num_samples, batch_size):
                # Get the samples you'll use in this batch
                self.batch_samples = np.empty((batch_size,3), dtype = np.ndarray)
                
                # Handle what happens when asking for a batch but theres no more new data
                if offset+batch_size > num_samples:
                    overflow = offset + batch_size - num_samples
                    self.batch_samples[0:batch_size-overflow] = ds[offset:offset+batch_size]
                    i_start = random.randint(0, num_samples-overflow)
                    self.batch_samples[batch_size-overflow:batch_size] = ds[i_start:i_start+overflow]           
                else:
                    self.batch_samples = ds[offset:offset+batch_size]
                # Preprocessinng
                X, y = self.preprocessing(self.batch_samples, detrend, use_highpass, 
                                          use_scaler, scaler, use_time_augmentor, timeAug, 
                                          use_noise_augmentor, noiseAug, highpass_freq)
                try:
                        y = np_utils.to_categorical(y, self.num_classes, dtype=np.int8)
                        if self.num_classes == 2:
                            y = y[:,1]
                except:
                    raise Exception(f'Error when doing to_categorical. Inputs are y: {y} and num_classes: {self.num_classes}') 
                yield X, y
    
    def preprocessing(self, batch_samples, detrend, use_highpass, use_scaler, scaler, use_time_augmentor, timeAug, 
                      use_noise_augmentor, noiseAug, highpass_freq = 0.1):
        if use_time_augmentor:
            batch_trace, batch_label = self.batch_to_aug_trace(batch_samples, timeAug)
        else:
            batch_trace, batch_label = self.batch_to_trace(batch_samples)
        if use_scaler:
            batch_trace = self.transform_batch(scaler, batch_trace)
        if use_noise_augmentor:
            batch_trace = noiseAug.batch_augment_noise(batch_trace, 0, noiseAug.noise_std/10)
        if detrend or use_highpass:
            batch_trace = self.detrend_highpass_batch_trace(batch_trace, detrend, use_highpass, highpass_freq)
        return batch_trace, batch_label
        

    
            