
import numpy as np
import pandas as pd
import json
import h5py
import seaborn as sns
import os
import csv
import sys
from sklearn.model_selection import train_test_split

base_dir = 'F:\Thesis_ssd\MasterThesis3.0'
os.chdir(base_dir)

        
class LoadData():
    
    def __init__(self, earth_explo_only = False, noise_earth_only = False, downsample = False, upsample = False, frac_diff = 1, seed = None, subsample_size = 1):
        self.seed = seed
        np.random.seed(self.seed)
        self.earth_explo_only = earth_explo_only
        self.noise_earth_only = noise_earth_only
        self.downsample = downsample
        self.upsample = upsample
        self.frac_diff = frac_diff
        self.subsample_size = subsample_size
        
        self.csv_folder = os.path.join('F:\\', 'Thesis_ssd','MasterThesis3.0','csv_folder')
        self.data_csv_name = 'event_paths_no_nan_no_induced.csv'
        self.full_ds = self.csv_to_numpy(self.data_csv_name, self.csv_folder)
        
        
        # WARNING: If you ever want to do Noise + Explosion only then you CANNOT set downsample = True. 
        # This will not produce the desired results
        
        if downsample or upsample:
            self.full_ds = self.balance_ds(self.full_ds, self.downsample, self.upsample, frac_diff = self.frac_diff)
            
        
        # Remove uninteresting label from ds but keep noise seperately if removed
        self.refine_full_ds()
        
        # The noise needs to be reduced in order to work properly in noise augmentor (creating training set for noise augmentor).
        if self.earth_explo_only or self.noise_earth_only:
            self.noise_ds, _ = train_test_split(self.noise_ds, test_size = 0.15, random_state = self.seed)
            self.noise_ds = self.noise_ds[0:int(len(self.noise_ds)*self.subsample_size)]
        
        self.train, val_test = train_test_split(self.full_ds, test_size = 0.15, random_state = self.seed)
        self.val, self.test = train_test_split(val_test, test_size = 0.5, random_state = self.seed)
        
        
            
        self.train = self.train[0:int((len(self.train)*self.subsample_size))]
        self.val = self.val[0:int((len(self.val)*self.subsample_size))]
        self.test = self.test[0:int((len(self.test)*self.subsample_size))]
        
        if self.earth_explo_only:
            self.label_dict = {'earthquake' : 0, 'explosion' : 1}
        elif self.noise_earth_only:
            self.label_dict = {'earthquake' : 0, 'noise' : 1}
        else:
            self.label_dict = {'earthquake':0, 'noise':1, 'explosion':2, 'induced':3}
    
    def refine_full_ds(self):
        if self.earth_explo_only or self.noise_earth_only:
            self.noise_ds = np.array(self.full_ds[self.full_ds[:,1] == "noise"])
            if self.earth_explo_only:
                self.full_ds = np.array(self.full_ds[self.full_ds[:,1] != "noise"])
            if self.noise_earth_only:
                self.full_ds = np.array(self.full_ds[self.full_ds[:,1] != "explosion"])
        if self.earth_explo_only and self.noise_earth_only:
            raise Exception("Cannot have both earth_explo_only = True and noise_earth_only = True")
    
    def get_datasets(self):
        return self.train, self.val, self.test  
        
    def csv_to_numpy(self, data_csv, csv_folder):
        with open(csv_folder + '/' + data_csv) as file:
            file_list = np.array(list(file))
            dataset = np.empty((len(file_list), 2), dtype=object)
            for idx, event in enumerate(file_list):
                path, label = event.split(',')
                dataset[idx][0] = path.rstrip()
                dataset[idx][1] = label.rstrip()
            file.close()
        return dataset
    
    def downsample_label(self, target_label, ds, n_samples):
        target_array = np.array([x for x in ds if x[1] == target_label], dtype = object)
        down_ds = np.array([y for y in ds if y[1] != target_label], dtype = object)
        down_ds = np.concatenate((down_ds, target_array[np.random.choice(target_array.shape[0], n_samples, replace = True)]))
        return np.array(down_ds)

    def upsample_label(self, target_label, ds, n_samples):
        target_array = np.array([x for x in ds if x[1] == target_label])
        up_ds = [y for y in ds if y[1] != target_label]
        up_ds = np.concatenate((up_ds, target_array[np.random.choice(target_array.shape[0], n_samples, replace = True)]))
        return np.array(up_ds)

    def frac_diff_n_samples(self, frac_diff, min_counts, max_counts):
        diff = max_counts - min_counts
        n_samples = int(min_counts + diff*frac_diff)
        return n_samples

    def balance_ds(self, ds, downsample, upsample, frac_diff = 0):
        unique_labels, counts = np.unique(ds[:,1], return_counts = True)
        nr_classes = len(unique_labels)
        if downsample:
            # Downsamples by first reducing the largest class, then the second class.
            for i in range(nr_classes-1):
                unique_labels, counts = np.unique(ds[:,1], return_counts = True)
                most_occuring_label = unique_labels[np.where(counts == max(counts))]
                n_samples_frac_diff = self.frac_diff_n_samples(frac_diff, min(counts), max(counts))
                ds = self.downsample_label(most_occuring_label, ds, n_samples_frac_diff)
        if upsample:
            if frac_diff != 0:
                unique_labels, counts = np.unique(ds[:,1], return_counts = True)
                least_occuring_label = unique_labels[np.where(counts == min(counts))]
                n_samples_for_balance = max(counts)
                ds = self.upsample_label(least_occuring_label, ds, n_samples_for_balance)
        np.random.shuffle(ds)
        return ds
    
    
    def get_label_dict(self):
        return self.label_dict
