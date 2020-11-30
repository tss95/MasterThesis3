import numpy as np
import pandas as pd
import json
import h5py
import sklearn as sk
import matplotlib.pyplot as plt
from obspy import Stream, Trace, UTCDateTime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import csv
import pylab as pl
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Activation, Conv1D, Dense, Dropout, Flatten, MaxPooling3D, BatchNormalization, InputLayer, LSTM
from keras.layers import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.utils import Sequence
from keras.optimizers import Adam
from tensorflow.keras import regularizers

from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import tensorflow as tf
import datetime
import re
from sklearn.metrics import confusion_matrix
from livelossplot import PlotLossesKeras
modeling_dir = 'C:\Documents\Thesis_ssd\MasterThesis\Classes\Modeling'
os.chdir(modeling_dir)
from CustomCallback import CustomCallback

class HelperFunctions():
    
    def plot_confusion_matrix(self, model, gen, test_ds, batch_size, class_dict, train_ds = None, train_testing = False):
        if not train_testing:
            ds = test_ds
        else:
            ds = train_ds
        steps = len(ds)/batch_size
        predictions = model.predict_generator(gen, steps)
        predicted_classes = self.convert_to_class(predictions)[0:len(ds)]
        true_classes = self.get_class_array(ds, class_dict)
        labels = list(class_dict.keys())[0:true_classes.shape[1]]
        cm = confusion_matrix(true_classes.argmax(axis=1), predicted_classes.argmax(axis=1))
        print(cm)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        return
    
    def evaluate_model(self, model, gen, test_ds, batch_size, class_dict):
        steps = len(test_ds)/batch_size
        print(model.evaluate_generator(gen, steps, verbose = 1))
        # Add this later!!! print(classification_report(, y_pred_bool))
        self.plot_confusion_matrix(model, gen, test_ds, batch_size, class_dict)
        
    def get_steps_per_epoch(self, gen_set, batch_size):
        return int(len(gen_set)/batch_size)
    
    def get_class_array(self, ds, class_dict):
        uniques = np.unique(ds[:,1])
        class_array = np.zeros((len(ds), len(uniques)))
        for idx, path_and_label in enumerate(ds):
            label = path_and_label[1]
            class_array[idx][class_dict.get(label)] = 1
        return class_array
                                
        for idx, path_and_label in enumerate(ds):
            if path_and_label[1] == "earthquake":
                class_array[idx][0] = 1
            elif path_and_label[1] == "noise":
                class_array[idx][1] = 1
            elif path_and_label[1] == "explosion":
                class_array[idx][2] = 1
            else:
                print(f"No class available: {path_and_label[1]}")
                break
        return class_array

    def convert_to_class(self, predictions):
        predicted_classes = np.zeros((predictions.shape))
        for idx, prediction in enumerate(predictions):
            highest_pred = max(prediction)
            highest_pred_index = np.where(prediction == highest_pred)
            predicted_classes[idx][highest_pred_index] = 1
        return predicted_classes
    
    def get_class_distribution_from_csv(self,data_csv):
        with open(data_csv) as file:
            classes, counts = np.unique(file[:,1], return_counts = True)
            file.close()
        return classes, counts
    
    def get_class_distribution_from_ds(self, ds):
        classes, counts = np.unique(ds[:,1], return_counts = True)
        return classes, counts
        
    def batch_class_distribution(self, batch):
        batch_size, nr_classes = batch[1].shape
        class_distribution = np.zeros((1,nr_classes))[0]
        print(class_distribution)
        for sample in batch[1]:
            for idx, i in enumerate(sample):
                if i == 1:
                    class_distribution[idx] += 1
        return class_distribution
    
    def get_trace_shape_no_cast(self, ds):
        num_ds = len(ds)
        with h5py.File(ds[0][0], 'r') as dp:
            trace_shape = dp.get('traces').shape
        return num_ds, trace_shape[0], trace_shape[1]
    
    def generate_build_model_args(self, model_nr, batch_size, dropout_rate, activation, output_layer_activation, l2_r, l1_r, 
                                  start_neurons, filters, kernel_size, padding, 
                                  num_classes = 3, channels = 3, timesteps = 6000):
        return {"model_nr" : model_nr,
                "input_shape" : (batch_size, channels, timesteps),
                "num_classes" : num_classes,
                "dropout_rate" : dropout_rate,
                "activation" : activation,
                "output_layer_activation" : output_layer_activation,
                "l2_r" : l2_r,
                "l1_r" : l1_r,
                "full_regularizer" : True,
                "start_neurons" : start_neurons,
                "filters" : filters,
                "kernel_size" : kernel_size,
                "padding" : padding}
    
    def generate_model_compile_args(self, opt, nr_classes):
        if nr_classes == 2:
            loss = "binary_crossentropy"
        else:
            loss = "categorical_crossentropy"
        return {"loss" : loss,
                      "optimizer" : opt,
                      "metrics" : ["accuracy",
                                   tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None),
                                   tf.keras.metrics.Recall(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)]}
    
    def generate_gen_args(self, batch_size, detrend, use_scaler = False, scaler = None, 
                          use_time_augmentor = False, timeAug = None, use_noise_augmentor = False, 
                          noiseAug = None,  num_classes = 3, use_highpass = False, highpass_freq = 0.3):
        return {"batch_size" : batch_size,
                    "detrend" : detrend,
                    "use_scaler" : use_scaler,
                    "scaler" : scaler,
                    "use_time_augmentor": use_time_augmentor,
                    "timeAug" : timeAug,
                    "use_noise_augmentor" : use_noise_augmentor,
                    "noiseAug" : noiseAug,
                    "num_classes" : num_classes,
                    "use_highpass" : use_highpass,
                    "highpass_freq" : highpass_freq}
    
    def generate_fit_args(self, train_ds, val_ds, batch_size, epoch, val_gen, use_tensorboard, use_liveplots, use_custom_callback, use_early_stopping):
        callbacks = []
        if use_liveplots:
            callbacks.append(PlotLossesKeras())
        if use_tensorboard:
            log_dir = "tensorboard_dir/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks.append(tensorboard_callback)
        if use_custom_callback:
            custom_callback = CustomCallback(data_gen)
            callbacks.append(custom_callback)
        if use_early_stopping:
            earlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)
            callbacks.append(earlystop)
        
        return {"steps_per_epoch" : self.get_steps_per_epoch(train_ds, batch_size),
                        "epochs" : epoch,
                        "validation_data" : val_gen,
                        "validation_steps" : self.get_steps_per_epoch(val_ds, batch_size),
                        "verbose" : 1,
                        "use_multiprocessing" : False, 
                        "workers" : 1,
                        "callbacks" : callbacks
                       }
    
    def getOptimizer(self, optimizer, learning_rate):
        if optimizer == "adam":
            return keras.optimizers.Adam(learning_rate=learning_rate)
        if optimizer == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        if optimizer == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise Exception(f"{optimizer} not implemented into getOptimizer")    
    
    def plot_event(trace, info):
        start_time = info['trace_stats']['starttime']
        channels = info['trace_stats']['channels']
        sampl_rate = info['trace_stats']['sampling_rate']
        station = info['trace_stats']['station']

        trace_BHE = Trace(
        data=trace[0],
        header={
            'station': station,
            'channel': channels[0],
            'sampling_rate': sampl_rate,
            'starttime': start_time})
        trace_BHN = Trace(
            data=trace[1],
            header={
                'station': station,
                'channel': channels[1],
                'sampling_rate': sampl_rate, 
                'starttime': start_time})
        trace_BHZ = Trace(
            data=trace[2],
            header={
                'station': station,
                'channel': channels[2],
                'sampling_rate': sampl_rate,
                'starttime': start_time})
        stream = Stream([trace_BHE, trace_BHN, trace_BHZ])
        stream.plot()

    def get_n_points_with_highest_training_loss(self, train_ds, n, full_logs):
        train_ds_dict = {}
        for path, label in train_ds:
            train_ds_dict[path] = {'label' : label,
                                   'loss': 0,
                                   'average_loss' : 0,
                                   'occurances' : 0}
        counter = 0
        for batch in full_logs:
            loss = batch['loss']
            for path_class in batch['batch_samples']:
                train_ds_dict[path_class[0]]['loss'] += loss
                train_ds_dict[path_class[0]]['occurances'] += 1

        train_ds_list = []
        for sample in np.array(train_ds[:,0]):
            if train_ds_dict[sample]['occurances'] == 0:
                continue
            train_ds_dict[sample]['average_loss'] = train_ds_dict[sample]['loss'] / train_ds_dict[sample]['occurances']
            train_ds_list.append((sample, train_ds_dict[sample]['label'],train_ds_dict[sample]['average_loss']))

        sorted_train_ds_list = sorted(train_ds_list, key=lambda x: x[2], reverse = True)

        return sorted_train_ds_list[0:n]