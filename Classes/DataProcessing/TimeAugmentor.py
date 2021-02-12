import numpy as np
from tensorflow.keras import utils
import math
import random
import datetime
from dateutil import parser
import time
import h5py
import json
from .LoadData import LoadData


class TimeAugmentor():

    """
    fitted_dict = {-paths-:{'initial_index' : int
                            'random_start_index' : [int list of length of the largest redundancy index of the path + 1]}}
    
    """


    
    def __init__(self, DataHandler, ds, seed = None):
        self.handler = DataHandler
        self.ds = ds
        self.fitted_dict = {}
        self.seed = seed

    """
    def fit(self):
        time_start = time.time()
        path_red_ds = self.ds[:,[0,2]]
        len_ds = len(path_red_ds)
        _,_,pre_length = self.handler.get_trace_shape_no_cast(self.ds, False)
        post_length = 6000
        np.random.seed(0)
        print("Issues will occur if upsampling of explosions or noise is implemented")
        explo_ds = self.ds[self.ds[:,1] == "explosion"]
        earth_ds = self.ds[self.ds[:,1] == "earthquake"]
        noise_ds = self.ds[self.ds[:,1] == "noise"]
        max_redundancy = max(earth_ds[:,2])
        #fitted_dict = dict.fromkeys(set(self.ds[:,0]))
        fitted_dict = {}
        if len(explo_ds) > 0:
            i = 0
            explo_len = len(set(explo_ds[:,0]))
            for path in set(explo_ds[:,0]):
                random_start_index = np.random.randint(0,4500, 1)
                initial_index, info = self.find_initial_event_index(path)
                fitted_dict[path] = {'initial_index': initial_index,
                                     'random_start_index' : random_start_index}
                self.progress_bar(i + 1, explo_len)
                i += 1
        print("Finished explosions")
        if len(noise_ds) > 0:
            i = 0
            noise_len = len(set(noise_ds[:,0]))
            for path in set(noise_ds[:,0]):
                random_start_index = np.random.randint(0,4500, 1)
                initial_index, info = self.find_initial_event_index(path)
                fitted_dict[path] = {'initial_index': initial_index,
                                     'random_start_index' : random_start_index}
                self.progress_bar(i + 1, noise_len)
                i += 1
        print("Finished noise")
        if len(earth_ds) > 0:
            i = 0
            earth_len = len(set(earth_ds[:,0]))
            for path in set(earth_ds[:,0]):
                random_start_index = np.random.randint(0,4500, max_redundancy)
                initial_index, info = self.find_initial_event_index(path)
                fitted_dict[path] = {'initial_index': initial_index,
                                     'random_start_index' : random_start_index}
                self.progress_bar(i + 1, earth_len)
                i += 1

        print("Finished earthquakes")
        self.fitted_dict = fitted_dict
        time_end = time.time()
        print(f"Fit process completed after {time_end - time_start} seconds. Total datapoints fitted: {len(path_red_ds)}.")
        print(f"Average time per datapoint: {(time_end - time_start) / len(path_red_ds)}")

    """

            
    def augment_event(self, path, redundancy_index):
        trace, info = self.handler.path_to_trace(path)
        fit = self.fitted_dict[path]
        augmented_trace = np.empty((3, 6000))
        
        random_start_index = fit['random_start_index'][int(redundancy_index)]
        initial_index = fit['initial_index']
        interesting_part_length = trace.shape[1] - initial_index
        missing_length = (augmented_trace.shape[1] - random_start_index) - interesting_part_length
        
        for i in range(augmented_trace.shape[0]):
            augmented_trace[i] = self.fill_start(trace, augmented_trace, random_start_index, initial_index, i)
            augmented_trace[i] = self.fill_interesting_part(trace, augmented_trace, random_start_index, interesting_part_length, initial_index, i)
            if missing_length > 0:
                # missing_length was intereting_part_length. Why? Error?
                augmented_trace[i] = self.fill_lacking_ends(trace, augmented_trace, random_start_index, missing_length, i)
        return augmented_trace
    
    def fill_start(self, trace, augmented_trace, random_start_index, initial_index, i_channel):
        if random_start_index < initial_index:
            augmented_trace[i_channel][0:random_start_index] = trace[i_channel][0:random_start_index]
            return augmented_trace[i_channel]
        else:
            augmented_trace[i_channel][0:initial_index] = trace[i_channel][0:initial_index]
            trace_interval_start = trace.shape[1] - (random_start_index - initial_index)
            trace_interval_end = trace.shape[1]
            augmented_trace[i_channel][initial_index:random_start_index] = trace[i_channel][trace_interval_start:trace_interval_end]
            return augmented_trace[i_channel]

    def fill_interesting_part(self, trace, augmented_trace, random_start_index, interesting_length, initial_index, i_channel):
        aug_interval_end = min(random_start_index + interesting_length, augmented_trace.shape[1])
        trace_interval_end = min(initial_index + interesting_length, initial_index + (augmented_trace.shape[1] - random_start_index))
        augmented_trace[i_channel][random_start_index:aug_interval_end] = trace[i_channel][initial_index:trace_interval_end]
        return augmented_trace[i_channel]
        
    def fill_lacking_ends(self, trace, augmented_trace, random_start_index, missing_length, i_channel):
        fill_interval_start = random_start_index
        fill_interval_end = random_start_index + missing_length
        augmented_trace[i_channel][augmented_trace.shape[1] - missing_length:augmented_trace.shape[1]] = trace[i_channel][fill_interval_start:fill_interval_end]
        return augmented_trace[i_channel]
    

    def find_initial_event_index(self, path):
        info = self.path_to_info(path)
        start_time = parser.isoparse(info['trace_stats']['starttime']).replace(tzinfo=None)
        if info['analyst_pick_time'] != None:
            event_time = parser.isoparse(info['analyst_pick_time']).replace(tzinfo=None)
            uncertainty = 0
        else:
            event_time = parser.isoparse(info['est_arrivaltime_arces']).replace(tzinfo=None)
            uncertainty = 0
            if 'origins' in info:
                if 'time_errors' in info['origins'][0]:
                    uncertainty = min(float(info['origins'][0]['time_errors']['uncertainty']), 15)
        sampling_rate = info['trace_stats']['sampling_rate']
        relative_seconds = (event_time - start_time).total_seconds()
        # Problem with uncertainty: Some events have very large uncertainty.
        # This can be so high that the interesting event could have potentially occured prior to the recording.          
        initial_index = max(math.floor((relative_seconds-uncertainty)*sampling_rate),0)
        return initial_index, info
    
    def path_to_info(self, path):
        with h5py.File(path, 'r') as dp:
            info = np.array(dp.get('event_info'))
            # This is a mess, but for some reason it works with this shitty code.
            info = str(info)
            info = info[2:len(info)-1]
            info = json.loads(info)
        return info
        
    
    def progress_bar(self, current, total, barLength = 20):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('Fitting time augmentor: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')     
        
# Old fit implementation. Inconsistency in performance cause of rewrite. This method is more memory efficient, and more robust.

    def fit(self):
        time_start = time.time()
        path_red_ds = self.ds[:,[0,2]]
        len_ds = len(path_red_ds)
        _,_,pre_length = self.handler.get_trace_shape_no_cast(self.ds, False)
        post_length = 6000
        np.random.seed(self.seed)
        gen = self.np_generator(path_red_ds)
        for idx in range(len_ds):
            path_red = next(gen)
            path = path_red[0]
            red = int(path_red[1])
            self.progress_bar(idx + 1, len_ds)
            if path in self.fitted_dict:
                if red + 1 <= len(self.fitted_dict[path]['random_start_index']):
                    continue
                else:
                    random_start_index = np.random.randint(0,4500, red + 1)
                    self.fitted_dict[path]['random_start_index'] = random_start_index
            else:
                random_start_index = np.random.randint(0, 4500, red+1)
                initial_index, info = self.find_initial_event_index(path)
                self.fitted_dict[path] = { 'initial_index' : initial_index,
                                           'random_start_index' : random_start_index}
            idx += 1
        time_end = time.time()
        print(f"Fit process completed after {time_end - time_start} seconds. Total datapoints fitted: {len(path_red_ds)}.")
        print(f"Average time per datapoint: {(time_end - time_start) / len(path_red_ds)}")
           
    def np_generator(self, path_red_ds):
        for row in path_red_ds:
            yield row
