import numpy as np
import random

import os
import sys
import threading

from .DataHandler import DataHandler

class RamGenerator(DataHandler):


    class threadsafe_iter:

        def __init__(self, it):
            self.it = it
            self.lock = threading.Lock()

        def __iter__(self):
            return self

        def next(self):
            with self.lock:
                return self.it.next()

    """
    After further inspection in RamGeneratorDevelop.ipynb, this class behaves as expected.
    """
    
    def __init__(self, loadData, handler, noiseAug = None):
        super().__init__(loadData)
        self.handler = handler
        self.num_classes = len(set(loadData.label_dict.values()))
        self.noiseAug = noiseAug

    def threadsafe_generator(f):
        """A decorator that takes a generator function and makes it thread-safe.
        """
        def g(*a, **kw):
            return threadsafe_iter(f(*a, **kw))
        return g
        
    @threadsafe_generator
    def data_generator(self, traces, labels, batch_size, num_channels = 3, is_lstm = False):
        """
        Creates a generator object which yields two arrays. One array for waveforms, and one array for labels
        """
        # Number of samples 
        num_samples = len(labels)
        while True:
            # Loop which goes from 0 to num_samples, jumping n number for each loop, where n is equal to batch_size
            for offset in range(0, num_samples, batch_size):
                # Initiates the arrays.
                batch_traces = np.empty((batch_size, traces.shape[1], traces.shape[2]))
                batch_labels = np.empty((batch_size, 1))
                # If condition that handles what happens when the funcion has been called k times, and k*batch_size > num_samples.
                # This makes sure that the shape of the arrays remain the same, even though there arent enough events to fill an entire batch.
                # when this condition is true, it will be the last iteration of the loop, so at next call the iterator will start at 0 again.
                if offset + batch_size > num_samples:
                    overflow = offset + batch_size - num_samples

                    batch_traces[0:batch_size - overflow] = traces[offset:(offset+batch_size) - overflow]
                    batch_labels[0:batch_size - overflow] = labels[offset:(offset+batch_size) - overflow]

                    i_start = random.randint(0, num_samples-overflow)
                    batch_traces[batch_size - overflow:batch_size] = traces[i_start:i_start + overflow]
                    batch_labels[batch_size - overflow:batch_size] = labels[i_start:i_start + overflow]
                # Regular trucking along here
                else:
                    batch_traces = traces[offset:offset + batch_size]
                    batch_labels = labels[offset:offset + batch_size]

                # Adds a little noise to each event, as a regulatory measure
                if self.noiseAug != None:
                    batch_traces = self.preprocess_data(batch_traces)

                batch_traces = batch_traces[:][:,0:num_channels]
                if is_lstm:
                    batch_traces = np.reshape(batch_traces, (batch_traces.shape[0], batch_traces.shape[2], batch_traces.shape[1]))

                yield batch_traces, batch_labels
                
    def preprocess_data(self, traces):
        return self.noiseAug.batch_augment_noise(traces, 0, self.noiseAug.noise_std/10)