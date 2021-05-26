import numpy as np
from tensorflow.keras import utils
from obspy import Stream, Trace


import os
classes_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3'
os.chdir(classes_dir)
from Classes.DataProcessing.TimeAugmentor import TimeAugmentor
from Classes.DataProcessing.NoiseAugmentor import NoiseAugmentor
from Classes.Scaling.MinMaxScalerFitter import MinMaxScalerFitter
from Classes.Scaling.StandardScalerFitter import StandardScalerFitter
from Classes.Scaling.RobustScalerFitter import RobustScalerFitter
from Classes.Scaling.DataNormalizer import DataNormalizer
from .decorators import runtime

class RamLoader:
    """
    Class responsible for preprocessing and loading the recordings to RAM.

    PARAMETERS:
    ------------------------------------------------------------------------

    loadData: (object)           Fitted LoadData object.
    handler: (object)            DataHandler object. Holds functions which are relevant to the loading and handling of the recordings.
    use_time_augmentor: (bool)   Boolean for whether or not to use time augmentation. 
    use_noise_augmentor: (bool)  Boolean for whether or not to use noise augmentation.
    scaler_name: (str)           String representing the name of the scaler type to be used in the preprocessing.
    filter_name: (str)           Name of the digital filter to be used. Will use default filter values unless otherwised specified.
    band_min: (float)            Minimum frequency parameter for Bandpass filter
    band_max: (float)            Maximum frequency parameter for Bandpass filter
    highpass_freq: (float)       Corner frequency for Highpass filter.
    load_test_set: (bool)        Whether or not to load the test set.
    meier_load: (bool)           True if training/evaluating Meier et al.'s model.
    """



    def __init__(self, loadData, handler, use_time_augmentor = False, use_noise_augmentor = False, scaler_name = None, 
                filter_name = None, band_min = 2.0, band_max = 4.0, highpass_freq = 0.1, load_test_set = False, meier_load = False):
        self.loadData = loadData
        self.handler = handler
        self.train_ds, self.val_ds, self.test_ds = self.loadData.get_datasets()
        self.noise_ds = self.loadData.noise_ds
        self.use_time_augmentor = use_time_augmentor
        self.use_noise_augmentor = use_noise_augmentor
        self.scaler_name = scaler_name
        self.filter_name = filter_name
        self.band_min = band_min
        self.band_max = band_max
        self.highpass_freq = highpass_freq
        self.load_test_set = load_test_set
        self.num_classes = len(set(handler.loadData.label_dict.values()))
        self.meier_load = meier_load
    
    def fit_timeAug(self, ds, dataset_name):
        timeAug = None
        if self.use_time_augmentor:
            timeAug = TimeAugmentor(self.handler, ds, dataset_name = dataset_name, seed = self.loadData.seed)
            timeAug.fit()
            print("\n")
        return timeAug
    
    def load_to_ram(self):
        if self.loadData.noise_not_noise:
            return self.load_to_ram_noise_not_noise()
        if self.loadData.earth_explo_only:
            return self.load_to_ram_earth_explo_only()
        if self.loadData.use_true_test_set:
            return self.load_to_ram_final_evaluation()
        if self.loadData.load_nukes:
            return self.load_to_ram_nukes()
        else:
            raise Exception("Loading to ram for this type of data has not been implemented.")

    def manually_cut_nuke(self, trace):
        new_trace = np.empty((3, 6000))
        for ch in range(new_trace.shape[0]):
            new_trace[ch] = trace[ch][2000: trace.shape[1]-4001]
        return new_trace

    @runtime
    def load_to_ram_nukes(self):
        print("Initializing loading of the nukes.")
        print("Step 1: Fit augmentors and scalers on training data. As normalized scaler is assumed to be used, only the noise events are loaded.")
        self.train_timeAug = self.fit_timeAug(self.noise_ds, "train")
        train_trace, train_label = self.stage_one_load(self.noise_ds, self.train_timeAug, 0)
        self.scaler = self.fit_scaler(train_trace)
        train_trace, train_label = self.stage_two_load(train_trace, train_label, 0, False)
        self.noiseAug = None
        if self.use_noise_augmentor:
            noise_indexes = np.where(train_label == self.loadData.label_dict["noise"])
            noise_traces = train_trace[noise_indexes]
            self.noiseAug = self.fit_noiseAug(self.loadData, noise_traces)
        print("Step 2: Load and transform the test set, using the previously fitted scaler and augmentors")
        del train_trace, train_label
        self.nuke_ds = self.loadData.nukes_ds
        nuke_trace, nuke_label = self.nuke_stage_one_load(self.nuke_ds)
        nuke_trace, nuke_label = self.stage_two_load(nuke_trace, nuke_label, 4, False)
        return nuke_trace, nuke_label, self.noiseAug


    @runtime
    def load_to_ram_final_evaluation(self):
        """
        This is written to as quickly as possible get the results I need.

        Shortcomings:
        - Does not handle models where different preprocessing steps are necessary.
        - "Meier mode" is not implemented.
        - Use beyond the design with caution.
        """
        print("Initializing loading of the test set")
        print("Step 1: Fit augmentors and scalers on training data")
        self.train_timeAug = self.fit_timeAug(self.train_ds, "train")
        train_trace, train_label = self.stage_one_load(self.train_ds, self.train_timeAug, 0)
        self.scaler = self.fit_scaler(train_trace)
        train_trace, train_label = self.stage_two_load(train_trace, train_label, 0, False)
        self.noiseAug = None
        if self.use_noise_augmentor:
            noise_indexes = np.where(train_label == self.loadData.label_dict["noise"])
            noise_traces = train_trace[noise_indexes]
            self.noiseAug = self.fit_noiseAug(self.loadData, noise_traces)
        print("Step 2: Load and transform the test set, using the previously fitted scaler and augmentors")
        del train_trace, train_label
        self.test_timeAug = self.fit_timeAug(self.test_ds, "test")
        test_trace, test_label = self.stage_one_load(self.test_ds, self.test_timeAug, 2)
        test_trace, test_label = self.stage_two_load(test_trace, test_label, 2, False)
        return test_trace, test_label, self.noiseAug


    @runtime
    def load_to_ram_noise_not_noise(self):
        self.train_timeAug = self.fit_timeAug(self.train_ds, "train")
        self.val_timeAug = self.fit_timeAug(self.val_ds, "validation")
        if self.load_test_set:
            self.test_timeAug = self.fit_timeAug(self.test_ds, "test")
        
        # Step one, load traces and apply time augmentation and/or detrend/highpass
        train_trace, train_label = self.stage_one_load(self.train_ds, self.train_timeAug, 0)
        val_trace, val_label = self.stage_one_load(self.val_ds, self.val_timeAug, 1)
        if self.load_test_set:
            test_trace, test_label = self.stage_one_load(self.test_ds, self.test_timeAug, 2)

        # Step 1.5, fit scaler:
        self.scaler = self.fit_scaler(train_trace)

        train_trace, train_label = self.stage_two_load(train_trace, train_label, 0, self.meier_load)
        val_trace, val_label = self.stage_two_load(val_trace, val_label, 1, self.meier_load)
        if self.load_test_set:
            test_trace, test_label = self.stage_two_load(test_trace, test_label, 2, self.meier_load)
        
        self.noiseAug = None
        if self.use_noise_augmentor:
            # Need to get only the noise traces:
            if (self.meier_load and self.num_classes == 2) or self.num_classes > 2:
                noise_indexes = np.where(train_label[:,self.loadData.label_dict["noise"]] == 1)
            if not self.meier_load and self.num_classes == 2:
                noise_indexes = np.where(train_label == self.loadData.label_dict["noise"])
            noise_traces = train_trace[noise_indexes]
            self.noiseAug = self.fit_noiseAug(self.loadData, noise_traces)
        print("\n")
        if self.load_test_set:
            return train_trace, train_label, val_trace, val_label, test_trace, test_label, self.noiseAug
        return train_trace, train_label, val_trace, val_label, self.noiseAug


    @runtime
    def load_to_ram_earth_explo_only(self):
        if self.use_time_augmentor:
            self.noise_timeAug = self.fit_timeAug(self.noise_ds, "noise set")
        self.train_timeAug = self.fit_timeAug(self.train_ds, "train")
        self.val_timeAug = self.fit_timeAug(self.val_ds, "validation")
        if self.load_test_set:
            self.test_timeAug = self.fit_timeAug(self.test_ds, "test")

        # Step one, load traces and apply time augmentation and/or detrend/highpass
        train_trace, train_label = self.stage_one_load(self.train_ds, self.train_timeAug, 0)
        val_trace, val_label = self.stage_one_load(self.val_ds, self.val_timeAug, 1)
        if self.load_test_set:
            test_trace, test_label = self.stage_one_load(self.test_ds, self.test_timeAug, 2)
        self.scaler = self.fit_scaler(train_trace)
        self.noiseAug = None
        if self.use_noise_augmentor:
            noise_trace, noise_label = self.stage_one_load(self.noise_ds, self.noise_timeAug, 3)
            noise_trace, _ = self.stage_two_load(noise_trace, noise_label, 3, self.meier_load)
            self.noiseAug = self.fit_noiseAug(self.loadData, noise_trace)
            print("\n")
            #del noise_trace, noise_label, self.noise_timeAug

        train_trace, train_label = self.stage_two_load(train_trace, train_label, 0, self.meier_load)
        val_trace, val_label = self.stage_two_load(val_trace, val_label, 1, self.meier_load)
        if self.load_test_set:
            test_trace, test_label = self.stage_two_load(test_trace, test_label, 2, self.meier_load)
        print("\n")
        print("Completed loading to RAM")
        if self.load_test_set:
            return train_trace, train_label, val_trace, val_label, test_trace, test_label, self.noiseAug
        return train_trace, train_label, val_trace, val_label, self.noiseAug

    
    
    def fit_scaler(self, traces):
        scaler = None
        if self.scaler_name != None:
            if self.scaler_name == "minmax":
                scaler = MinMaxScalerFitter()
                scaler.fit_scaler_ram(traces)
                scaler = scaler.scaler
            elif self.scaler_name == "standard":
                scaler = StandardScalerFitter()
                scaler.fit_scaler_ram(traces)
                scaler = scaler.scaler
            elif self.scaler_name == "robust":
                scaler = RobustScalerFitter()
                scaler.fit_scaler_ram(traces)
                scaler = scaler.scaler
            elif self.scaler_name == "normalize":
                scaler = DataNormalizer()
                print("Fit process of normalizer skipped as unecessary")
            elif self.scaler_name != "minmax" or self.scaler_name != "standard" or self.scaler_name != "robust":
                raise Exception(f"{self.scaler_name} is not implemented.")
            print("\n")
        return scaler

    def fit_noiseAug(self, loadData, noise_traces):
        noiseAug = None
        if self.use_noise_augmentor:
            noiseAug = NoiseAugmentor(loadData, noise_traces)
        print("\n")
        return noiseAug
            
    def get_substage(self, substage):
        if substage == 0:
            return "training set"
        if substage == 1:
            return "validation set"
        if substage == 2:
            return "test set"
        if substage == 3:
            return "noise set"
        if substage == 4:
            return "nuclear set"
        return ""
    
    def nuke_stage_one_load(self, ds):
        loaded_label = np.empty((len(ds), 1))
        loaded_trace = np.empty((self.handler.get_trace_shape_no_cast(ds, True)))
        num_events = len(ds)
        bar_text = self.stage_one_text(4)
        for i in range(num_events):
            self.progress_bar(i+1, num_events, bar_text)
            loaded_label[i] = self.handler.label_dict.get(ds[i][1])
            trace = self.handler.path_to_trace(ds[i][0])[0]
            loaded_trace[i] = self.manually_cut_nuke(trace)
        print("\n")
        return loaded_trace, loaded_label


    def stage_one_load(self, ds, timeAug, substage):
        loaded_label = np.empty((len(ds), 1))
        loaded_trace = np.empty((self.handler.get_trace_shape_no_cast(ds, self.use_time_augmentor)))
        num_events = len(ds)
        bar_text =  self.stage_one_text(substage)
        for i in range(num_events):
            self.progress_bar(i+1, num_events, bar_text)
            loaded_label[i] = self.handler.label_dict.get(ds[i][1])
            # timeAug, highpass and detrend.
            if self.filter_name != None or self.use_time_augmentor:
                if self.filter_name != None and self.use_time_augmentor:
                    loaded_trace[i] = timeAug.augment_event(ds[i][0], ds[i][2])
                    info = self.handler.path_to_trace(ds[i][0])[1]
                    loaded_trace[i] = self.apply_filter(loaded_trace[i], info, self.filter_name, highpass_freq = self.highpass_freq, band_min = self.band_min, band_max = self.band_max)
                if self.filter_name == None:
                    loaded_trace[i] = timeAug.augment_event(ds[i][0], ds[i][2])
                if not self.use_time_augmentor:
                    loaded_trace[i] = self.handler.path_to_trace(ds[i][0])[0]
                    info = self.handler.path_to_trace(ds[i][0])[1]
                    loaded_trace[i] = self.apply_filter(loaded_trace[i], info, self.filter_name, highpass_freq = self.highpass_freq, band_min = self.band_min, band_max = self.band_max)
            else:
                loaded_trace[i] = self.handler.path_to_trace(ds[i][0])[0]
        print("\n")
        return loaded_trace, loaded_label
    
    def stage_one_text(self, substage):
        bar_text = f"Stage one loading {self.get_substage(substage)}"
        if self.filter_name != None or self.use_time_augmentor:
            bar_text = bar_text + ", "
            if self.filter_name != None:
                bar_text += self.filter_name
            if self.filter_name != None and self.use_time_augmentor:
                bar_text += " and "
            if self.use_time_augmentor:
                bar_text += "timeAug"
        return bar_text
    
    def stage_two_load(self, traces, labels, substage, meier_load):
        num_samples = traces.shape[0]
        bar_text = self.stage_two_text(substage)
        if self.scaler_name != None:
            if self.scaler_name != "normalize":
                for i in range(num_samples):
                    self.progress_bar(i+1, num_samples, bar_text)
                    traces[i] = np.transpose(self.scaler.transform(np.transpose(traces[i])))
                print("\n")
            else:
                for i in range(num_samples):
                    self.progress_bar(i+1, num_samples, bar_text)
                    traces[i] = self.scaler.fit_transform_trace(traces[i])
                print("\n")
        
        if substage != 3:
            labels = utils.to_categorical(labels, self.num_classes, dtype=np.int8)
            if self.num_classes == 2 and not meier_load:
                labels = labels[:,1]
                labels = np.reshape(labels, (labels.shape[0],1))
        return traces, labels
    
    def stage_two_text(self, substage):
        bar_text = f"Stage two loading {self.get_substage(substage)}, labels"
        if self.scaler_name != None:
            bar_text = bar_text + f" and {self.scaler_name} scaler"

        return bar_text
        

    
    def apply_filter(self, trace, info, filter_name, highpass_freq = 1.0, band_min = 2.0, band_max = 4.0):
        station = info['trace_stats']['station']
        channels = info['trace_stats']['channels']
        sampl_rate = info['trace_stats']['sampling_rate']
        starttime = info['trace_stats']['starttime']
        trace_BHE = Trace(data=trace[0], header ={'station' : station,
                                                  'channel' : channels[0],
                                                  'sampling_rate' : sampl_rate,
                                                  'starttime' : starttime})
        trace_BHN = Trace(data=trace[1], header ={'station' : station,
                                                  'channel' : channels[1],
                                                  'sampling_rate' : sampl_rate,
                                                  'starttime' : starttime})
        trace_BHZ = Trace(data=trace[2], header ={'station' : station,
                                                  'channel' : channels[2],
                                                  'sampling_rate' : sampl_rate,
                                                  'starttime' : starttime})
        stream = Stream([trace_BHE, trace_BHN, trace_BHZ])
        stream.detrend('demean')
        if filter_name == "highpass":
            stream.taper(max_percentage=0.05, type='cosine')
            stream.filter('highpass', freq = highpass_freq)
        if filter_name == "bandpass":
            stream.taper(max_percentage=0.05, type='cosine')
            stream.filter('bandpass', freqmin=band_min, freqmax=band_max)
        return np.array(stream)

    def progress_bar(self, current, total, text, barLength = 40):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('%s: [%s%s] %d %%' % (text, arrow, spaces, percent), end='\r')



