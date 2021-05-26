import numpy as np
import math
from dateutil import parser
import time
import h5py
import json

class TimeAugmentor():

    """
    The class responsible for time augmentation of the waveforms. The core of the process is in the fitted_dict:
    fitted_dict = {-paths-:{'initial_index' : int
                            'random_start_index' : [int list of length of the largest redundancy index of the path + 1]}}.
    When fitted, this object can augment events very quickly.

    Note: The speed of the fitting process depends on hash collisions. The fitting process is generally quick for a smaller dataset. 
          Consider making this class more robust to inaccurately labeled start times.
    
    PRAMETERS:
    ----------------------------------------------------------------------------------
    handler: (object)          DataHandler object. Holds functions which are relevant to the loading and handling of the recordings.
    ds: (np.array)             Array holding the list of recordings. 
    dataset_name: (str)        Name of the dataset to be fitted. Used in progress bar.
    fitted_dict: (dict)        Dictionary where the key is the path of the recording.
    seed: (int)                Seed. Makes sure the augmentation is the same every time.
    """
    
    def __init__(self, DataHandler, ds, dataset_name, seed = None):
        self.handler = DataHandler
        self.ds = ds
        self.dataset_name = dataset_name
        self.fitted_dict = {}
        self.seed = seed

            
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
                augmented_trace[i] = self.fill_lacking_ends(trace, augmented_trace, random_start_index, missing_length, i)
        return augmented_trace
    
    def fill_start(self, trace, augmented_trace, random_start_index, initial_index, i_channel):
        if random_start_index < initial_index:
            augmented_trace[i_channel][0:random_start_index] = trace[i_channel][0:random_start_index]
            return augmented_trace[i_channel]
        else:
            # Fill the beginning of the waveform with noise leading up to the event
            augmented_trace[i_channel][0:initial_index] = trace[i_channel][0:initial_index]
            trace_interval_start = trace.shape[1] - (random_start_index - initial_index)
            trace_interval_end = trace.shape[1]
            # Fill gap between the beginning and the interesting event with slice of the end.
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
        initial_index = max(math.floor((relative_seconds-uncertainty)*sampling_rate),0)
        return initial_index, info
    
    def path_to_info(self, path):
        with h5py.File(path, 'r') as dp:
            info = np.array(dp.get('event_info'))
            info = str(info)
            info = info[2:len(info)-1]
            info = json.loads(info)
        return info
        
    
    def progress_bar(self, current, total, text, barLength = 40):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('%s: [%s%s] %d %%' % (text, arrow, spaces, percent), end='\r')  
        
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
            self.progress_bar(idx + 1, len_ds, f"Fitting {self.dataset_name} time augmentor")
            if path in self.fitted_dict:
                if red + 1 <= len(self.fitted_dict[path]['random_start_index']):
                    continue
                else:
                    # This random number interval might be the most important parameter in this entire project.
                    # This is because the second batch of data all appear to start very close to the 1 minute mark, and have a duration of interest longer than 1 minute.
                    random_start_index = np.random.randint(0,2000, red + 1)
                    self.fitted_dict[path]['random_start_index'] = random_start_index
            else:
                random_start_index = np.random.randint(0, 2000, red+1)
                initial_index, info = self.find_initial_event_index(path)
                self.fitted_dict[path] = { 'initial_index' : initial_index,
                                           'random_start_index' : random_start_index}
            idx += 1
        time_end = time.time()
        print("\n")
        print(f"Fit process completed after {time_end - time_start} seconds. Total datapoints fitted: {len(path_red_ds)}.")
        print(f"Average time per datapoint: {(time_end - time_start) / len(path_red_ds)}")
           
    def np_generator(self, path_red_ds):
        for row in path_red_ds:
            yield row
