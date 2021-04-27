import numpy as np
import random

import os
import sys
import threading

#from .DataHandler import DataHandler
from tensorflow.keras.utils import GeneratorEnqueuer
from tensorflow.keras.utils import Sequence
import time

class threadsafe_iter:

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


        
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

#@threadsafe_generator    
def data_generator(traces, labels, batch_size, noiseAug, num_channels = 3, is_lstm = False, norm_scale = False):
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

            batch_traces = traces[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]

            # Adds a little noise to each event, as a regulatory measure
            if noiseAug != None:
                # Since normalize scaler is independent it will scale noise proportionally less than non noise events
                if not norm_scale:
                    batch_traces = preprocess_data(batch_traces, noiseAug, 1/15)
                else:
                    # Choose 1/15, but this is just some number. Potential for improvement by tuning.
                    # Chaned to 1/20 as the validation results were soooo affected by the augmentation
                    batch_traces = preprocess_data(batch_traces, noiseAug, 1/20)

            batch_traces = batch_traces[:][:,0:num_channels]
            if is_lstm:
                batch_traces = np.reshape(batch_traces, (batch_traces.shape[0], batch_traces.shape[2], batch_traces.shape[1]))

            yield batch_traces, batch_labels

def preprocess_data(traces, noiseAug, std_frac):
    return noiseAug.batch_augment_noise(traces, 0, noiseAug.noise_std*std_frac)


@threadsafe_generator
def modified_data_generator(traces, labels, batch_size, noiseAug, num_channels = 3, is_lstm = False, norm_scale = False):
    """
    Creates a generator object which yields two arrays. One array for waveforms, and one array for labels
    """
    # Number of samples 
    num_samples = len(labels)
    print("USING MODIFIED GENERATOR!! BE CAREFUL")
    while True:
        # Loop which goes from 0 to num_samples, jumping n number for each loop, where n is equal to batch_size
        for offset in range(0, num_samples, batch_size):
            # Initiates the arrays.
            batch_traces = np.empty((batch_size, traces.shape[1], traces.shape[2]))
            batch_labels = np.empty((batch_size, 2))

            
            
            batch_traces = traces[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]

            # Adds a little noise to each event, as a regulatory measure
            if noiseAug != None:
                if not norm_scale:
                    batch_traces = preprocess_data(batch_traces, noiseAug, 1/15)
                else:
                    batch_traces = preprocess_data(batch_traces, noiseAug, 1/20)

            batch_traces = batch_traces[:][:,0:num_channels]
            if is_lstm:
                batch_traces = np.reshape(batch_traces, (batch_traces.shape[0], batch_traces.shape[2], batch_traces.shape[1]))

            yield batch_traces, batch_labels

@threadsafe_generator
def ramless_data_generator(ds, y, batch_size, ramLessLoader, timeAug, num_channels, input_shape, is_lstm = True, norm_scale = False):
    num_samples = len(y)
    timesteps, channels = input_shape
    while True:
        # Loop which goes from 0 to num_samples, jumping n number for each loop, where n is equal to batch_size
        for offset in range(0, num_samples, batch_size):
            batch_traces = load_batch(ds[offset:offset + batch_size], timeAug, (batch_size, channels, timesteps), ramLessLoader)
            batch_labels = y[offset:offset + batch_size]

            # Adds a little noise to each event, as a regulatory measure
            if ramLessLoader.noiseAug is not None:
                # Since normalize scaler is independent it will scale noise proportionally less than non noise events
                if not norm_scale:
                    batch_traces = preprocess_data(batch_traces, ramLessLoader.noiseAug, 1/15)
                else:
                    # Choose 1/15, but this is just some number. Potential for improvement by tuning.
                    # Changed to 1/20 as the validation results were soooo affected by the augmentation
                    batch_traces = preprocess_data(batch_traces, ramLessLoader.noiseAug, 1/20)

            batch_traces = batch_traces[:][:,0:num_channels]
            if is_lstm:
                batch_traces = np.reshape(batch_traces, (batch_traces.shape[0], batch_traces.shape[2], batch_traces.shape[1]))

            yield batch_traces, batch_labels



def load_batch(batch, timeAug, shape, ramLessLoader):
    return ramLessLoader.load_batch(batch, timeAug, np.empty((shape)))


def get_rambatch(num_workers, max_queue_size, **kwargs):
    # Code similar to: https://github.com/watersink/Character-Segmentation/blob/master/data_generator.py
    try:
        enq = GeneratorEnqueuer(data_generator(**kwargs), use_multiprocessing = False)
        enq.start(workers = num_workers, max_queue_size = max_queue_size)
        gen_output = None
        while True:
            while enq.is_running():
                if not enq.queue.empty():
                    gen_output = enq.get()
                    break
                else:
                    time.sleep(0.01)
            yield gen_output
            gen_output = None
    finally:
        if enq is not None:
            enq.stop()

def get_ramlessbatch(num_workers, max_queue_size, **kwargs):
    # Code similar to: https://github.com/watersink/Character-Segmentation/blob/master/data_generator.py
    try:
        enq = GeneratorEnqueuer(ramless_data_generator(**kwargs), use_multiprocessing = False)
        enq.start(workers = num_workers, max_queue_size = max_queue_size)
        gen_output = None
        while True:
            while enq.is_running():
                if not enq.queue.empty():
                    gen_output = enq.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield gen_output
            gen_output = None
    finally:
        if enq is not None:
            enq.stop()
        
    
    