from keras.utils import np_utils
import numpy as np
import pandas as pd

from .BigDataLoader import BigDataLoader
from .BigDataHandler import BigDataHandler
from .BigScalerFitter import BigScalerFitter
from .BigStandardScalerFitter import BigStandardScalerFitter

class BigDataGenerator():
    
    def __init__(self, data_loader):
        self.loader = data_loader
        self.handler = self.loader.handler
   
   
    def data_generator(self, ds, batch_size, use_scaler = False, scaler = None):
        channels, timesteps = self.handler.get_trace_shape(ds)
        num_samples = len(ds)
        while True:
            for offset in range(0, num_samples, batch_size):
                # Get the samples you'll use in this batch
                self.batch_samples = np.empty((batch_size,2), dtype = np.ndarray)
                
                # Handle what happens when asking for a batch but theres no more new data
                if offset+batch_size > num_samples:
                    overflow = offset + batch_size - num_samples
                    self.batch_samples[0:batch_size-overflow] = ds[offset:offset+batch_size]
                    i_start = random.randint(0, num_samples-overflow)
                    self.batch_samples[batch_size-overflow:batch_size] = ds[i_start:i_start+overflow]           
                else:
                    self.batch_samples = ds[offset:offset+batch_size]
                # Preprocessinng
                X, y = self.preprocessing(self.batch_samples, use_scaler, scaler)
                try:
                    y = np_utils.to_categorical(y, len(np.unique(y)), dtype=np.int64)
                except:
                    raise Exception(f'Error when doing to_categorical. Inputs are y: {y} and num_classes: {len(np.unique(y))}')               
                yield X, y
    
    def preprocessing(self, batch_samples, use_scaler, scaler):
        batch_trace, batch_label = self.handler.batch_to_trace_binary_label(batch_samples)
        if use_scaler:
            batch_trace = self.handler.transform_batch(scaler, batch_trace)
        return batch_trace, batch_label
        