
import numpy as np
import os
from sklearn.model_selection import train_test_split

base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3/'
os.chdir(base_dir)
from GlobalUtils import GlobalUtils
glob_utils = GlobalUtils()

        
        
class LoadData():
    """
    This class is responsible from loading all the data from the premade csv to numpy arrays.
    It also splits into training/validation/test and performs sampling of the data.
    This class also contains label_dict, which is the translation between the one-hot encoded labels and the
    labels in their text form.
    
    NOTE: The even_label_occurance function is not robust at all. It is used when doing noise-not-noise, and
          its functionality is strictly dependent on label_dict = { 'noise': 0, 'earthquake' : 1, 'explosion' : 1}.
    
    PARAMETERS:
    -----------------------------------------------------------------------------------------------------------------
    earth_explo_only:(Bool)      Determining if only events labeled explosion and earthquake is to be loaded. 
                                 Will also load a training set sized sample of noise events for the noise augmentor.
                                 Splits into train/val/test
    
    noise_earth_only:(Bool)      Intended to behave much like the one above. Currently not in use.
    
    noise_not_noise:(Bool)       Much like earth_explo_only, this boolean loads events of all classes, and splits them
                                 into train/val/test
    
    downsample:(Bool)            Will reduce the most frequent event so that it matches the second most frequent class in number.
    
    upsample:(Bool)              Will sample with replacement the least frequent class to match the second most frequent class in number.
    
    frac_diff:(float)            Ignore this, always set to 1.
    
    seed: (int)                  Seed, makes sure that the shuffling and splits of the data are the same everytime with the same parameters.
    
    subsample_size:(float)       Fraction, (0,1] which will select a subsample of the sets in order to reduce computational resource demand.
    
    balance_non_train_set:(Bool) Dictates whether or not upsampling/downsampling and even_balance should be done on the validation and test sets
    
    use_true_test_set:(Bool)     Whether or not the test set should consist of the isolated training set, or if a pseudo test set should be used
    
    load_everything:(Bool)       Whether or not to load all the data from the second batch of data. Will also load the isolated
                                 test set. Useful when graphing and looking at the whole dataset. NEVER USE FOR MODELING
    
    even_balance:(Bool)          Whether or not to balance the classes so that each class so that the distribution of each class
                                 is all_events/nr_clases.

    load_nukes: (Bool)           Whether or not to load to nuclear explosion waveforms.
    """
    
    def __init__(self, earth_explo_only = False, noise_earth_only = False, noise_not_noise = False, 
                 downsample = False, upsample = False, frac_diff = 1, seed = None, subsample_size = 1,
                 balance_non_train_set = False, use_true_test_set = False, load_everything = False, 
                 load_first_batch = False, even_balance = False, load_nukes = False):

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
        self.load_nukes = load_nukes

        self.csv_folder = glob_utils.csv_dir
        self.data_csv_name = glob_utils.data_csv_name
        if self.load_nukes:
            self.initialize_nuclear_tests()
            return
        if load_first_batch:
            self.data_csv_name = glob_utils.batch_1_csv_name
            assert not load_everything, "Load everything should be False when using the first batch. A test set has not been generated for this dataset"
        if load_everything:
            print("Loading all of second batch. Including the test data.")
            self.data_csv_name = glob_utils.no_nan_no_induced_csv_name
        else:
            if self.use_true_test_set:
                self.test_csv_name = glob_utils.test_csv_name
                self.test_ds = self.csv_to_numpy(self.test_csv_name, self.csv_folder)
                print("WARNING: You are using the true test set.")
                print("If this is an error, please set use_true_test_set = False and reload the kernel")
        if sum([self.earth_explo_only, self.noise_earth_only, self.noise_not_noise]) > 1:
            raise Exception("Invalid load data arguments.")
        self.full_ds = self.csv_to_numpy(self.data_csv_name, self.csv_folder)
        self.create_label_dict()
        self.load_data()
        print("\n")
        if not load_everything and not load_first_batch:
            self.print_data_info()

    def load_data(self):
        if not self.use_true_test_set:
            if self.noise_not_noise:
                self.load_noise_not_noise()
            if self.earth_explo_only:
                self.load_earth_explo_only()
            if self.noise_earth_only:
                raise Exception("Not implemented noise earth only. Seems unecessary")
        else:
            self.load_final_evaluation()


    def initialize_nuclear_tests(self):
        # Need to load data to fit preprocessors
        self.full_ds = self.csv_to_numpy(self.data_csv_name, self.csv_folder)
        noise = self.full_ds[self.full_ds[:,1] == "noise"]
        explosions = self.full_ds[self.full_ds[:,1] == "explosion"]
        earthquakes = self.full_ds[self.full_ds[:,1] == "earthquake"]
        
        train_noise, _ = train_test_split(noise, test_size = 0.15, random_state = self.seed)
        train_explo, _ = train_test_split(explosions, test_size = 0.15, random_state = self.seed)
        train_earth, _ = train_test_split(earthquakes, test_size = 0.15, random_state = self.seed)

        self.train = np.concatenate((train_noise, train_explo, train_earth))
        self.train = self.balance_ds(self.train, self.downsample, self.upsample, frac_diff = self.frac_diff)

        if self.even_balance:
            self.train = self.even_label_occurances(self.train)
        self.train = self.train[np.random.choice(self.train.shape[0], int(len(self.train)*self.subsample_size), replace = False)]
        self.train = self.map_redundancy(self.train, "train")
        self.noise_ds = self.train[self.train[:,1] == "noise"]
        self.val = None
        self.test = None
        nuke_csv_name = glob_utils.nukes_csv_name
        nukes = self.csv_to_numpy(nuke_csv_name, self.csv_folder)
        nukes_ds = np.zeros((nukes.shape[0], nukes.shape[1] + 1), dtype = object)
        nukes_ds[:,:-1] = nukes
        self.nukes_ds = nukes_ds
        self.create_label_dict()
        
    def load_noise_not_noise_true_test(self):
        # Training and validation:
        noise = self.full_ds[self.full_ds[:,1] == "noise"]
        explosions = self.full_ds[self.full_ds[:,1] == "explosion"]
        earthquakes = self.full_ds[self.full_ds[:,1] == "earthquake"]
        
        # Creating unique train/val splits for each class:
        train_noise, val_noise = train_test_split(noise, test_size = 0.15, random_state = self.seed)
        train_explo, val_explo = train_test_split(explosions, test_size = 0.15, random_state = self.seed)
        train_earth, val_earth = train_test_split(earthquakes, test_size = 0.15, random_state = self.seed)
        
        # Combining events. This way we prevent duplicates in other sets.
        self.train = np.concatenate((train_noise, train_explo, train_earth))
        self.val = np.concatenate((val_noise, val_explo, val_earth))
        self.test = self.test_ds
        
        # Up and down sampling 
        self.train = self.balance_ds(self.train, self.downsample, self.upsample, frac_diff = self.frac_diff)
        if self.balance_non_train_set:
            self.val = self.balance_ds(self.val, self.downsample, self.upsample, frac_diff = self.frac_diff)
            self.test = self.balance_ds(self.test, self.downsample, self.upsample, frac_diff = self.frac_diff)
        else:
            # Shuffles the data if not
            self.val = self.balance_ds(self.val, False, False, frac_diff = self.frac_diff)
            self.test = self.balance_ds(self.test, False, False, frac_diff = self.frac_diff)

        if self.even_balance:
            self.train = self.even_label_occurances(self.train)
            if self.balance_non_train_set:
                self.val = self.even_label_occurances(self.val)
                self.test = self.even_label_occurances(self.test)

        self.train = self.train[np.random.choice(self.train.shape[0], int(len(self.train)*self.subsample_size), replace = False)]
        self.val = self.val[np.random.choice(self.val.shape[0], int(len(self.val)*self.subsample_size), replace = False)]
        print("NO SUBSAMPLING DONE ON THE TRUE TEST SET")

        # Mapping redundnad samples for time augmentor
        self.train = self.map_redundancy(self.train, "train")
        self.val = self.map_redundancy(self.val, "validation")
        self.test = self.map_redundancy(self.test, "test")
        self.full_ds = np.concatenate((self.train, self.val, self.test))
        
        self.noise_ds = self.train[self.train[:,1] == "noise"]
        
        
        
    def load_noise_not_noise(self):
        noise = self.full_ds[self.full_ds[:,1] == "noise"]
        explosions = self.full_ds[self.full_ds[:,1] == "explosion"]
        earthquakes = self.full_ds[self.full_ds[:,1] == "earthquake"]
        
        # Unique noise split
        train_noise, val_test_noise = train_test_split(noise, test_size = 0.2, random_state = self.seed)
        val_noise, test_noise = train_test_split(val_test_noise, test_size = 0.4, random_state = self.seed)
        
        #Unique explosion split
        train_explosions, val_test_explosions = train_test_split(explosions, test_size = 0.2, random_state = self.seed)
        val_explosions, test_explosions = train_test_split(val_test_explosions, test_size = 0.4, random_state = self.seed)
        
        #Unique earthquake split
        train_earthquakes, val_test_earthquakes = train_test_split(earthquakes, test_size = 0.2, random_state = self.seed)
        val_earthquakes, test_earthquakes = train_test_split(val_test_earthquakes, test_size = 0.4, random_state = self.seed)
        
        # Combining so that events are not duplicated in the splits
        self.train = np.concatenate((train_noise, train_explosions, train_earthquakes))
        self.val = np.concatenate((val_noise, val_explosions, val_earthquakes))
        self.test = np.concatenate((test_noise, test_explosions, test_earthquakes))
        
        # Combining so that events are not duplicated in the splits
        self.train = np.concatenate((train_noise, train_explosions, train_earthquakes))
        self.val = np.concatenate((val_noise, val_explosions, val_earthquakes))
        self.test = np.concatenate((test_noise, test_explosions, test_earthquakes))
        # Up and down sampling 
        self.train = self.balance_ds(self.train, self.downsample, self.upsample, frac_diff = self.frac_diff)
        if self.balance_non_train_set:
            self.val = self.balance_ds(self.val, self.downsample, self.upsample, frac_diff = self.frac_diff)
            self.test = self.balance_ds(self.test, self.downsample, self.upsample, frac_diff = self.frac_diff)
        else:
            # Shuffles the data if not
            self.val = self.balance_ds(self.val, False, False, frac_diff = self.frac_diff)
            self.test = self.balance_ds(self.test, False, False, frac_diff = self.frac_diff)

        if self.even_balance:
            self.train = self.even_label_occurances(self.train)
            if self.balance_non_train_set:
                self.val = self.even_label_occurances(self.val)
                self.test = self.even_label_occurances(self.test)

        self.train = self.train[np.random.choice(self.train.shape[0], int(len(self.train)*self.subsample_size), replace = False)]
        self.val = self.val[np.random.choice(self.val.shape[0], int(len(self.val)*self.subsample_size), replace = False)]
        self.test = self.test[np.random.choice(self.test.shape[0], int(len(self.test)*self.subsample_size), replace = False)]

        # Mapping redundnad samples for time augmentor
        self.train = self.map_redundancy(self.train, "train")
        self.val = self.map_redundancy(self.val, "validation")
        self.test = self.map_redundancy(self.test, "test")
        self.full_ds = np.concatenate((self.train, self.val, self.test))
        
        self.noise_ds = self.train[self.train[:,1] == "noise"]

        
    def load_earth_explo_only_true_test(self):
        noise = self.full_ds[self.full_ds[:,1] == "noise"]
        explosions = self.full_ds[self.full_ds[:,1] == "explosion"]
        earthquakes = self.full_ds[self.full_ds[:,1] == "earthquake"]
        
       # Creating unique train/val splits for each class:
        train_noise, _ = train_test_split(noise, test_size = 0.15, random_state = self.seed)
        train_explo, val_explo = train_test_split(explosions, test_size = 0.15, random_state = self.seed)
        train_earth, val_earth = train_test_split(earthquakes, test_size = 0.15, random_state = self.seed)
        
        # Combing so that events are not duplciated
        self.train = np.concatenate((train_explo, train_earth))
        self.val = np.concatenate((val_explo, val_earth))
        self.test = self.test_ds[self.test_ds[:,1] != "noise"]
        
        # Up and down sampling 
        self.train = self.balance_ds(self.train, self.downsample, self.upsample, frac_diff = self.frac_diff)
        if self.balance_non_train_set:
            self.val = self.balance_ds(self.val, self.downsample, self.upsample, frac_diff = self.frac_diff)
            self.test = self.balance_ds(self.test, self.downsample, self.upsample, frac_diff = self.frac_diff)
        else:
            # Shuffles the data if not
            self.val = self.balance_ds(self.val, False, False, frac_diff = self.frac_diff)
            self.test = self.balance_ds(self.test, False, False, frac_diff = self.frac_diff)

        if self.even_balance:
            self.train = self.even_label_occurances(self.train)
            if self.balance_non_train_set:
                self.val = self.even_label_occurances(self.val)
                self.test = self.even_label_occurances(self.test)

        self.train = self.train[np.random.choice(self.train.shape[0], int(len(self.train)*self.subsample_size), replace = False)]
        self.val = self.val[np.random.choice(self.val.shape[0], int(len(self.val)*self.subsample_size), replace = False)]
        print("NOT SUBSAMPLING THE TRUE TEST SET.")
        
        # Mapping redundnad samples for time augmentor
        self.train = self.map_redundancy(self.train, "train")
        self.val = self.map_redundancy(self.val, "validation")
        self.test = self.map_redundancy(self.test, "test")
        self.full_ds = np.concatenate((self.train, self.val, self.test))
        
        # Create noise_ds. 
        self.noise_ds = self.train[self.train[:,1] == "noise"]        
        
    def load_final_evaluation(self):
        self.train = self.full_ds
        self.test = self.test_ds
        # Up and down sampling 
        self.train = self.balance_ds(self.train, self.downsample, self.upsample, frac_diff = self.frac_diff)
        if self.even_balance:
            self.train = self.even_label_occurances(self.train)
        self.train = self.train[np.random.choice(self.train.shape[0], int(len(self.train)*self.subsample_size), replace = False)]

        # Mapping redundnad samples for time augmentor
        self.train = self.map_redundancy(self.train, "train")
        self.test = self.map_redundancy(self.test, "test")
        self.val = None
        self.noise_ds = self.train[self.train[:,1] == "noise"]
            
    def load_earth_explo_only(self):
        noise = self.full_ds[self.full_ds[:,1] == "noise"]
        
        explosions = self.full_ds[self.full_ds[:,1] == "explosion"]
        earthquakes = self.full_ds[self.full_ds[:,1] == "earthquake"]
        
        
        # Unique noise split
        train_noise, _ = train_test_split(noise, test_size = 0.2, random_state = self.seed)
        
        #Unique explosion split
        train_explosions, val_test_explosions = train_test_split(explosions, test_size = 0.2, random_state = self.seed)
        val_explosions, test_explosions = train_test_split(val_test_explosions, test_size = 0.4, random_state = self.seed)
        
        #Unique earthquake split
        train_earthquakes, val_test_earthquakes = train_test_split(earthquakes, test_size = 0.2, random_state = self.seed)
        val_earthquakes, test_earthquakes = train_test_split(val_test_earthquakes, test_size = 0.4, random_state = self.seed)
        
        # Combining so that events are not duplicated in the splits
        self.train = np.concatenate((train_explosions, train_earthquakes))
        self.val = np.concatenate((val_explosions, val_earthquakes))
        self.test = np.concatenate((test_explosions, test_earthquakes))
        
        # Up and down sampling 
        self.train = self.balance_ds(self.train, self.downsample, self.upsample, frac_diff = self.frac_diff)
        if self.balance_non_train_set:
            self.val = self.balance_ds(self.val, self.downsample, self.upsample, frac_diff = self.frac_diff)
            self.test = self.balance_ds(self.test, self.downsample, self.upsample, frac_diff = self.frac_diff)
        else:
            # Shuffles the data if not
            self.val = self.balance_ds(self.val, False, False, frac_diff = self.frac_diff)
            self.test = self.balance_ds(self.test, False, False, frac_diff = self.frac_diff)

        if self.even_balance:
            self.train = self.even_label_occurances(self.train)
            if self.balance_non_train_set:
                self.val = self.even_label_occurances(self.val)
                self.test = self.even_label_occurances(self.test)

        self.train = self.train[np.random.choice(self.train.shape[0], int(len(self.train)*self.subsample_size), replace = False)]
        self.val = self.val[np.random.choice(self.val.shape[0], int(len(self.val)*self.subsample_size), replace = False)]
        self.test = self.test[np.random.choice(self.test.shape[0], int(len(self.test)*self.subsample_size), replace = False)]

        # Mapping redundnad samples for time augmentor
        self.train = self.map_redundancy(self.train, "train")
        self.val = self.map_redundancy(self.val, "validation")
        self.test = self.map_redundancy(self.test, "test")
        self.full_ds = np.concatenate((self.train, self.val, self.test))
        
        # Create noise_ds. 
        # Create zero redundancy column
        train_noise = train_noise[np.random.choice(train_noise.shape[0], int(len(train_noise)*self.subsample_size), replace = False)]
        zero_column = np.zeros((len(train_noise), 1), dtype = np.int)
        self.noise_ds = np.hstack((train_noise, zero_column))
    
                
    def create_label_dict(self):
        # Used 
        self.complete_label_dict = {'noise' : 0, 'earthquake' : 1, 'explosion' : 2}
        # Method which produces the dictionary for labels. This is used in order to disguise labels during training.
        if self.earth_explo_only:
            self.label_dict = self.earth_explo_dict()
        elif self.noise_earth_only:
            self.label_dict = {'earthquake' : 0, 'noise' : 1}
        elif self.noise_not_noise:
            self.label_dict = self.noise_not_noise_dict()
        else:
            self.label_dict = {'noise' : 0, 'explosion' : 1, 'earthquake' : 2}
    
    def noise_not_noise_dict(self):
        return {'noise': 0, 'earthquake' : 1, 'explosion' : 1}
    
    def earth_explo_dict(self):
        return {'explosion' : 0, 'earthquake' : 1}

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
            print("Balancing due to disguised labels.")
            print("This functions barely works, and is a piece of shit that should not be trusted. Only works because noise has id: 0")
            ids = self.label_dict.values()
            most_occuring_id = max(ids)
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
    
    def map_redundancy(self, ds, set_name):
        # Creates a redundancy index which distinguishes events which are sampled multiple times.
        # Primarily used in timeAugmentation in order to create unique augmentations of otherwise identical events.
        # This only works if we are upsampling EARTHQUAKES (NOTHING ELSE)!
        # TODO: Check this function and improve it. This is sad.
        new_column = np.zeros((len(ds), 1), dtype = np.int8)
        mapped_ds = np.hstack((ds, new_column))
        earth_ds = ds[ds[:,1] == "earthquake"]
        unique_earth_paths = set(earth_ds[:,0])
        nr_unique_earth_paths = len(unique_earth_paths)
        for idx, path in enumerate(unique_earth_paths):
            self.progress_bar(idx + 1, nr_unique_earth_paths, f"Mapping {set_name} redundancy: ")
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
        print("\n")
        return mapped_ds

    def progress_bar(self, current, total, text, barLength = 40):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('%s: [%s%s] %d %%' % (text, arrow, spaces, percent), end='\r')

    def print_data_info(self):
        if self.earth_explo_only:
            print("Loaded explosion and earthquake dataset:")
        if self.noise_not_noise:
            print("Loaded noise non-noise dataset.")
        if self.use_true_test_set:
            print("Loaded true test set, accompanied by a train set for preprocessing fitting.")
        if self.even_balance:
            print("Evenly balanced among classes in the train set.")
        if self.balance_non_train_set:
            print("As well as non train sets.")
        print("Distribution (Label: (counts, proportion)) of")
        print("Train ds:")
        labels, counts = np.unique(self.train[:,1], return_counts = True)
        print(self.generate_dist_string_EE(labels, counts))
        if not self.use_true_test_set:
            print("Val ds:")
            labels, counts = np.unique(self.val[:,1], return_counts = True)
            print(self.generate_dist_string_EE(labels, counts))
        print("Test ds:")
        labels, counts = np.unique(self.test[:,1], return_counts = True)
        print(self.generate_dist_string_EE(labels, counts))

        
        
    def generate_dist_string_EE(self, labels, counts):
        string = ""
        for i in range(len(labels)):
            string += f"{labels[i]}: ({counts[i]}, {np.round(counts[i]/np.sum(counts), decimals = 4)})  "
            if i != len(labels) - 1:
                string += "|  "
        return string
            
