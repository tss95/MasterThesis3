import numpy as np
from keras.utils import np_utils
import random

from .LoadData import LoadData
from .DataHandler import DataHandler
from .NoiseAugmentor import NoiseAugmentor

class DataGenerator(DataHandler):
    
    def __init__(self, loadData):
        super().__init__(loadData)
   
   
    def data_generator(self, ds, batch_size, test = False, detrend = False, num_classes = 3, 
                       use_scaler = False, scaler = None, use_noise_augmentor = False, augmentor = None,
                       use_highpass = False, highpass_freq = 0.49):
        
        num_samples, channels, timesteps = self.get_trace_shape_no_cast(ds)
        num_samples = len(ds)
        if test:
            num_samples = int(num_samples * 0.1)
        while True:
            for offset in range(0, num_samples, batch_size):
                # Get the samples you'll use in this batch
                self.batch_samples = np.empty((batch_size,2), dtype = np.ndarray)
                
                # Handle what happens when asking for a batch but theres no more new data
                if offset+batch_size > num_samples and not test:
                    overflow = offset + batch_size - num_samples
                    self.batch_samples[0:batch_size-overflow] = ds[offset:offset+batch_size]
                    i_start = random.randint(0, num_samples-overflow)
                    self.batch_samples[batch_size-overflow:batch_size] = ds[i_start:i_start+overflow]           
                else:
                    self.batch_samples = ds[offset:offset+batch_size]
                # Preprocessinng
                X, y = self.preprocessing(self.batch_samples, detrend, use_highpass, 
                                          use_scaler, scaler, use_noise_augmentor, augmentor, highpass_freq)
                try:
                    y = np_utils.to_categorical(y, num_classes, dtype=np.int64)
                except:
                    raise Exception(f'Error when doing to_categorical. Inputs are y: {y} and num_classes: {num_classes}')               
                yield X, y
    
    def preprocessing(self, batch_samples, detrend, use_highpass, use_scaler, scaler, use_noise_augmentor, augmentor, highpass_freq = 0.1):
        batch_trace, batch_label = self.batch_to_trace(batch_samples)
        if use_scaler:
            batch_trace = self.transform_batch(scaler, batch_trace)
        if use_noise_augmentor:
            batch_trace = augmentor.batch_augment_noise(batch_trace, 0, augmentor.noise_std/10)
        if detrend or use_highpass:
            batch_trace = self.detrend_highpass_batch_trace(batch_trace, detrend, use_highpass, highpass_freq)
        return batch_trace, batch_label
        

    
            