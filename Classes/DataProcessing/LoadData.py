
import numpy as np
import pandas as pd
import json
import h5py
import seaborn as sns
import os
import csv
import sys
from sklearn.model_selection import train_test_split

base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3.0/'
os.chdir(base_dir)

        
        
class LoadData():
    
    def __init__(self, earth_explo_only = False, noise_earth_only = False, noise_not_noise = False, 
                 downsample = False, upsample = False, frac_diff = 1, seed = None, subsample_size = 1,
                 balance_non_train_set = False, use_true_test_set = False, load_everything = False, 
                 load_first_batch = False, even_balance = False):
        self.seed = seed
        np.random.seed(self.seed)
        self.earth_explo_only = earth_explo_only
        self.noise_earth_only = noise_earth_only
        self.noise_not_noise = noise_not_noise
        self.downsample = downsample
        self.upsample = upsample
        self.frac_diff = frac_diff
        self.subsample_size = subsample_size
        self.balance_non_train_set = balance_non_train_set
        self.use_true_test_set = use_true_test_set
        # If true, then the class distribution will be equal to 1/num_classes.
        self.even_balance = even_balance
        
        
        self.csv_folder = os.path.join('/media/tord/T7', 'Thesis_ssd','MasterThesis3.0','csv_folder')
        self.data_csv_name = 'full_no_test.csv'
        if load_first_batch:
            self.data_csv_name = 'full_batch_1.csv'
            assert not load_everything, "Load everything should be False when using the first batch. A test set has not been generated for this dataset"
        if load_everything:
            self.data_csv_name = 'event_paths_no_nan_no_induced.csv'
            self.full_ds = self.csv_to_numpy(self.data_csv_name, self.csv_folder)
            self.create_label_dict()
        else:
            self.test_csv_name = 'DO_NOT_TOUCH_test_set.csv'
            self.full_ds = self.csv_to_numpy(self.data_csv_name, self.csv_folder)
            self.create_label_dict()
            self.load_data()
            if self.use_true_test_set:
                self.true_test_ds = self.csv_to_numpy(self.test_csv_name, self.csv_folder)
                print("WARNING!")
                print("You are using the true test set.")
                print("THIS SHOULD ONLY BE USED ONCE")
                print("If this is an error, please set use_true_test_set = False and reload the kernel")

            if sum([self.earth_explo_only, self.noise_earth_only, self.noise_not_noise]) > 1:
                raise Exception("Invalid load data arguments.")

    def load_data(self):
        if not self.use_true_test_set:
            if self.balance_non_train_set:
                self.full_ds = self.balance_ds(self.full_ds, self.downsample, self.upsample, frac_diff = self.frac_diff)
                if self.even_balance:
                    self.full_ds = self.even_label_occurances(self.full_ds)
                self.full_ds = self.full_ds[np.random.choice(self.full_ds.shape[0], int(len(self.full_ds)*self.subsample_size), replace = False)]
                self.refine_full_ds()
                self.train, val_test = train_test_split(self.full_ds, test_size = 0.25, random_state = self.seed)
                self.val, self.test = train_test_split(val_test, test_size = 0.4, random_state = self.seed)
                if not self.earth_explo_only:
                    self.noise_ds = self.train[self.train[:,1] == "noise"]
            else:
                
                self.full_ds = self.balance_ds(self.full_ds, False, False, frac_diff = self.frac_diff)
                self.full_ds = self.full_ds[np.random.choice(self.full_ds.shape[0], int(len(self.full_ds)*self.subsample_size), replace = False)]
                if self.earth_explo_only or self.noise_earth_only:
                    if self.earth_explo_only:
                        self.noise_ds = np.array(self.full_ds[self.full_ds[:,1] == "noise"])
                        self.full_ds = np.array(self.full_ds[self.full_ds[:,1] != "noise"])
                        # The noise needs to be reduced in order to work properly in noise augmentor
                        self.noise_ds, _ = train_test_split(self.noise_ds, test_size = 0.25, random_state = self.seed)
                        zero_column = np.zeros((len(self.noise_ds), 1), dtype = np.int8)
                        self.noise_ds = np.hstack((self.noise_ds, zero_column))
                    else:
                        self.full_ds = np.array(self.full_ds[self.full_ds[:,1] != "explosion"])
                self.train, val_test = train_test_split(self.full_ds, test_size = 0.25, random_state = self.seed)
                self.val, self.test = train_test_split(val_test, test_size = 0.4, random_state = self.seed)
                self.train = self.balance_ds(self.train, self.downsample, self.upsample, frac_diff = self.frac_diff)
                if self.even_balance:
                    self.train = self.even_label_occurances(self.train)
                if self.upsample:
                    self.train = self.map_redundancy(self.train)
                else:
                    zero_column = np.zeros((len(self.train), 1), dtype = np.int)
                    self.train = np.hstack((self.train, zero_column))
                zero_val = np.zeros((len(self.val), 1), dtype = np.int)
                zero_test = np.zeros((len(self.test), 1), dtype = np.int)
                self.val = np.hstack((self.val, zero_val), dtype = np.int)
                self.test = np.hstack((self.test, zero_test), dtype = np.int)
                self.full_ds = np.concatenate((self.train, self.val))
                self.full_ds = np.concatenate((self.full_ds, self.test))
                if not self.earth_explo_only:
                    self.noise_ds = self.train[self.train[:,1] == "noise"]
                
        else:
            print("Write this code when you are ready to use the test set.")
            raise Exception("The code has not yet been written for the true test set.")
            
                
    
                
    
    def refine_full_ds(self):
        # Method which removes undesired classes from the dataset. 
        # If noise is removed, a seperate dataset will be created for use in NoiseAugmentation
        # Method also calls for map redundancy, or performs equivalent action for formatting consistancy if unecessary.
        if self.earth_explo_only or self.noise_earth_only:
            if self.earth_explo_only:
                self.noise_ds = np.array(self.full_ds[self.full_ds[:,1] == "noise"])
                self.full_ds = np.array(self.full_ds[self.full_ds[:,1] != "noise"])
                zero_column = np.zeros((len(self.noise_ds), 1), dtype = np.int8)
                self.noise_ds = np.hstack((self.noise_ds, zero_column))
            if self.noise_earth_only:
                self.full_ds = np.array(self.full_ds[self.full_ds[:,1] != "explosion"])
        if self.earth_explo_only and self.noise_earth_only:
            raise Exception("Cannot have both earth_explo_only = True and noise_earth_only = True")
        # Only need to map redundency if upsampling, as upsampling is the cause of redundancy
        if self.upsample:
            self.full_ds = self.map_redundancy(self.full_ds)
        else:
            zero_column = np.zeros((len(self.full_ds), 1), dtype = np.int8)
            self.full_ds = np.hstack((self.full_ds, zero_column))
    
    def create_label_dict(self):
        # Method which produces the dictionary for labels. This is used in order to disguise labels during training.
        if self.earth_explo_only:
            self.label_dict = {'earthquake' : 0, 'explosion' : 1}
        elif self.noise_earth_only:
            self.label_dict = {'earthquake' : 0, 'noise' : 1}
        elif self.noise_not_noise:
            self.label_dict = { 'noise': 0, 'earthquake' : 1, 'explosion' : 1}
        else:
            self.label_dict = {'earthquake' : 0, 'noise' : 1, 'explosion' : 2, 'induced' : 3}
    
    def get_datasets(self):
        return self.full_ds, self.train, self.val, self.test  
        
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
        # Method which produces n_samples of the targeted label, and returns the dataset unchanged but with n_samples of the targeted label.
        target_array = np.array([x for x in ds if x[1] == target_label], dtype = object)
        down_ds = np.array([y for y in ds if y[1] != target_label], dtype = object)
        np.random.seed(self.seed)
        down_ds = np.concatenate((down_ds, target_array[np.random.choice(target_array.shape[0], n_samples, replace = True)]))
        return np.array(down_ds)

    def upsample_label(self, target_label, ds, n_samples):
        # Seemingly equivalent to downsample_label (?)
        target_array = np.array([x for x in ds if x[1] == target_label])
        up_ds = [y for y in ds if y[1] != target_label]
        np.random.seed(self.seed)
        up_ds = np.concatenate((up_ds, target_array[np.random.choice(target_array.shape[0], n_samples, replace = True)]))
        return np.array(up_ds)

    def frac_diff_n_samples(self, frac_diff, min_counts, max_counts):
        # Returns the difference between the most occuring label and the least occuring label, multiplied by some fraction, and added to the number of the least occuring label.
        # Potentially a really stupid idea which is now implemented. Can be omitted by setting frac diff = 1 
        diff = max_counts - min_counts
        n_samples = int(min_counts + diff*frac_diff)
        return n_samples

    def balance_ds(self, ds, downsample, upsample, frac_diff = 0):
        # Downsamples the two most occuring labels, and upsamples the most occuring label.
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
        np.random.seed(self.seed)
        np.random.shuffle(ds)
        return ds

    def even_label_occurances(self, ds):
        # In cases where labels are disguised as something else, this method will even them out so that the label distribution is even. 
        num_classes = len(set(self.label_dict.values()))
        print(num_classes, len(set(self.label_dict.keys())))
        if num_classes != len(set(self.label_dict.keys())):
            ids = self.label_dict.values()
            most_occuring_id = max(ids)
            least_occuring_id = min(ids)
            label_count_dict = {}
            for label, id in self.label_dict.items():
                label_count_dict[label] = len(ds[ds[:,1] == label])
            print(label_count_dict)
            # Want the labels which share id, to combined have the same number of events of the event with the unique id.
            for label, id in self.label_dict.items():
                if id == most_occuring_id:
                    pure_label_ds = ds[ds[:,1] == label]
                    ds_without_label = ds[ds[:,1] != label]
                    ds = np.concatenate((ds_without_label, pure_label_ds[np.random.choice(pure_label_ds.shape[0], label_count_dict[label]//num_classes, replace = False)]))
        return ds
    
    def get_label_dict(self):
        return self.label_dict
    
    def map_redundancy(self, ds):
        # Creates a redundancy index which distinguishes events which are sampled multiple times.
        # Primarily used in timeAugmentation in order to create unique augmentations of otherwise identical events.
        # This only works if we are upsampling EARTHQUAKES (NOTHING ELSE)!
        new_column = np.zeros((len(ds), 1), dtype = np.int8)
        mapped_ds = np.hstack((ds, new_column))
        earth_ds = ds[ds[:,1] == "earthquake"]
        unique_earth_paths = set(earth_ds[:,0])
        nr_unique_earth_paths = len(unique_earth_paths)
        for idx, path in enumerate(unique_earth_paths):
            self.progress_bar(idx + 1, nr_unique_earth_paths)
            nr_repeats = len(earth_ds[earth_ds[:,0] == path])
            label = earth_ds[earth_ds[:,0] == path][0][1]
            repeating_indexes = np.where(ds[ds[:,0] == path][:,0][0] == ds[:,0])[0]
            current_index = 0
            if len(repeating_indexes) > 1:
                for event in earth_ds[earth_ds[:,0] == path]:
                    mapped_ds[repeating_indexes[current_index]][0] = path
                    mapped_ds[repeating_indexes[current_index]][1] = label
                    mapped_ds[repeating_indexes[current_index]][2] = current_index
                    current_index += 1
        return mapped_ds

    def progress_bar(self, current, total, barLength = 40):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('Mapping redundancy: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')
