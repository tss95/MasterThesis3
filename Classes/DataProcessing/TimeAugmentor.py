import numpy as np
from keras.utils import np_utils
import math
import random
import datetime
from dateutil import parser

from .LoadData import LoadData


class TimeAugmentor():
    
    def __init__(self, DataHandler, ds, seed = None):
        self.handler = DataHandler
        self.ds = ds
        self.fitted_dict = {}
        self.seed = seed
             
    def fit(self):
        path_ds = self.ds[:,0]
        len_ds = len(path_ds)
        _,_,pre_length = self.handler.get_trace_shape_no_cast(self.ds, False)
        post_length = 6000
        if self.seed != None:
            np.random.seed(self.seed)
        for idx, path in enumerate(path_ds):
            self.progress_bar(idx + 1, len_ds)
            initial_index, info = self.find_initial_event_index(path)
            random_start_index = np.random.randint(0, 5000)
            interesting_part_length = pre_length - initial_index
            # Handling what happens when the duration of the interesting event is shorter than what is needed to fill the array:
            ideal_length = post_length - random_start_index
            missing_length = ideal_length - interesting_part_length
            # Only interesting for events where we need to fill in at the end:
            # Problem occurs if initial_index - missing_length  < 0.
            if initial_index - missing_length <= 0:
                filler_index = 0
            else: 
                filler_index_start = np.random.randint(0, (initial_index - missing_length))
            filler_index_end = filler_index_start + missing_length
            # First index of what requires more filling
            required_fill_index_start = post_length - missing_length
            self.fitted_dict[path] = { 'initial_index' : initial_index,
                                       'random_start_index' : random_start_index,
                                       'interesting_part_length' : interesting_part_length,
                                       'missing_length' : missing_length,
                                       'filler_index_start' : filler_index_start,
                                       'filler_index_end' : filler_index_end,
                                       'required_fill_index_start' : required_fill_index_start}
            
            
            
            
    def augment_event(self, path):
        trace, info = self.handler.path_to_trace(path)
        fit = self.fitted_dict[path]
        augmented_trace = np.empty((3, 6000))
        for i in range(augmented_trace.shape[0]):
            augmented_trace[i] = self.fill_start(trace, augmented_trace, fit['random_start_index'], fit['initial_index'], i)
            augmented_trace[i] = self.fill_interesting_part(trace, augmented_trace, fit['random_start_index'], fit['interesting_part_length'], fit['initial_index'], i)
            if fit['missing_length'] > 0:
                augmented_trace[i] = self.fill_lacking_ends(trace, augmented_trace, fit['random_start_index'], fit['interesting_part_length'], i)
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
        _, info = self.handler.path_to_trace(path)
        start_time = parser.isoparse(info['trace_stats']['starttime']).replace(tzinfo=None)
        if info['analyst_pick_time'] != None:
            event_time = parser.isoparse(info['analyst_pick_time']).replace(tzinfo=None)
        else:
            event_time = parser.isoparse(info['est_arrivaltime_arces']).replace(tzinfo=None)
        sampling_rate = info['trace_stats']['sampling_rate']
        relative_seconds = (event_time - start_time).total_seconds()
        # Problem with uncertainty: Some events have very large uncertainty.
        # This can be so high that the interesting event could have potentially occured prior to the recording.
        uncertainty = 0
        if 'origins' in info:
            if 'time_errors' in info['origins'][0]:
                uncertainty = float(info['origins'][0]['time_errors']['uncertainty'])
            
        initial_index = max(math.floor((relative_seconds-uncertainty)*sampling_rate),0)

        return initial_index, info
    
    def progress_bar(self, current, total, barLength = 20):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('Fitting time augmentor: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')   