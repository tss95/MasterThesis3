import numpy as np
import pandas as pd
from tensorflow.keras import utils

import os
import sys
classes_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3.0/'
os.chdir(classes_dir)
from Classes.DataProcessing.LoadData import LoadData
from Classes.DataProcessing.HelperFunctions import HelperFunctions
from Classes.DataProcessing.DataHandler import DataHandler
from Classes.DataProcessing.DataGenerator import DataGenerator
from Classes.DataProcessing.TimeAugmentor import TimeAugmentor
from Classes.Scaling.ScalerFitter import ScalerFitter
from Classes.Scaling.MinMaxScalerFitter import MinMaxScalerFitter
from Classes.Scaling.StandardScalerFitter import StandardScalerFitter

class RamLoader:
    def __init__(self,handler, timeAug = None, scaler = None):
        self.handler = handler
        self.timeAug = timeAug
        self.scaler = scaler
        self.use_time_augmentor = False
        self.num_classes = len(set(handler.loadData.label_dict.values()))
        if self.timeAug != None:
            self.use_time_augmentor = True
    
    def load_to_ram(self, ds, is_lstm, num_channels = 3):
        loaded_label = np.empty((len(ds), 1))
        loaded_trace = np.empty((self.handler.get_trace_shape_no_cast(ds, self.use_time_augmentor)))
        print("Starting loading to RAM")
        if self.timeAug != None and self.scaler != None:
            for i in range(len(ds)):
                loaded_trace[i] = self.timeAug.augment_event(ds[i][0], ds[i][2])
                loaded_trace[i] = self.scaler.transform(loaded_trace[i])
                loaded_label[i] = self.handler.label_dict.get(ds[i][1])
        elif self.timeAug != None:
            for i in range(len(ds)):
                loaded_trace[i] = self.timeAug.augment_event(ds[i][0], ds[i][2])
                loaded_label[i] = self.handler.label_dict.get(ds[i][1])
        elif self.scaler != None:
            for i in range(len(ds)):
                loaded_trace[i] = self.handler.path_to_trace(ds[i][0])
                loaded_trace[i] = self.scaler.transform(loaded_trace[i])
                loaded_label[i] = self.handler.label_dict.get(ds[i][1])
        else:
            for i in range(len(ds)):
                loaded_trace[i] = self.handler.path_to_trace(ds[i][0])
                loaded_label[i] = self.handler.label_dict.get(ds[i][1])
        
        loaded_trace = loaded_trace[:][:,0:num_channels]
        if is_lstm:
            loaded_trace = np.reshape(loaded_trace, (loaded_trace.shape[0], loaded_trace.shape[2], loaded_trace.shape[1]))
        loaded_label = utils.to_categorical(loaded_label, self.num_classes, dtype=np.int8)
        if self.num_classes == 2:
            loaded_label = loaded_label[:,1]
            loaded_label = np.reshape(loaded_label, (loaded_label.shape[0],1))
        print("Completed loading to RAM")
        return loaded_trace, loaded_label
