import numpy as np
import h5py
import matplotlib.pyplot as plt
from obspy import Stream, Trace
from sklearn.metrics import classification_report, precision_recall_fscore_support, precision_recall_curve, average_precision_score
import seaborn as sns
import os

import pprint
import gc

import tensorflow as tf
from tensorflow.keras import utils

from Classes.DataProcessing.RamGen import RamGen
from Classes.DataProcessing.RamLessGen import RamLessGen
from tensorflow.keras import utils
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

import datetime
import re
from livelossplot import PlotLossesKeras
base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3/'
os.chdir(base_dir)
from GlobalUtils import GlobalUtils
from Classes.Modeling.CustomCallback import CustomCallback
glob_utils = GlobalUtils()

class HelperFunctions():
    
    def predict_model_no_generator(self, model, x_test, y_test, noiseAug, class_dict, scaler_name, num_channels):
        print("This function has not been updated since overhaul.")
        if noiseAug is not None:
            if scaler_name != "normalize":
                x_test = noiseAug.batch_augment_noise(x_test, 0, noiseAug.noise_std/15)
            else:
                x_test = noiseAug.batch_augment_noise(x_test, 0, noiseAug.noise_std/20)
        x_test[:][:,:num_channels]
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[2], x_test.shape[1]))
        model.evaluate(x = x_test, y = y_test)
        predictions = model.predict(x_test)
        predictions = model.predict(x_test)
        predictions = self.convert_to_class(predictions)
        return predictions
    
    def predict_RamGenerator(self, model, X, y, batch_size, norm_scale, noiseAug, label_dict, num_channels):
        steps = self.get_steps_per_epoch(X, batch_size)
        gen = RamGen(X, y, batch_size, noiseAug, num_channels, norm_scale = norm_scale, shuffle = False, label_dict = label_dict)
        predictions = model.predict(x = gen, steps = steps)
        del gen
        return predictions

    def predict_RamLessGenerator(self, model, ds, y, batch_size, norm_scale, ramLessLoader, timeAug, num_channels):
        steps = self.get_steps_per_epoch(ds, batch_size)
        gen = RamLessGen(ds, y, batch_size, ramLessLoader, timeAug, num_channels, norm_scale = norm_scale, shuffle = False)
        predictions = model.predict(x = gen, steps = steps)
        del gen
        return predictions
    
    def convert_to_class(self, predictions):
        if predictions.shape[1] == 1:
            predictions = np.rint(predictions)
            return predictions
        raise Exception("More than two classes has not been implemented")

    def evaluate_no_generator(self, model, x_test, y_test, label_dict, num_channels, noiseAug, scaler_name, num_classes, plot_confusion_matrix = False, plot_p_r_curve = False, beta = 1):
        predictions = self.predict_model_no_generator(model, x_test, y_test, noiseAug, label_dict, scaler_name)
        return self.evaluate_predictions(predictions, y_test, num_classes, label_dict, plot_confusion_matrix, plot_p_r_curve, beta)

    def evaluate_generator(self, model, x_test, y_test, batch_size, label_dict, num_channels, noiseAug, scaler_name, num_classes, plot_conf_matrix = False, plot_p_r_curve = False, beta = 1):
        norm_scale = False
        if scaler_name == "normalize":
            norm_scale = True
        pp = pprint.PrettyPrinter(indent=4)
        predictions = self.predict_RamGenerator(model, x_test, y_test, batch_size, norm_scale, noiseAug, label_dict, num_channels)
        return self.evaluate_predictions(predictions, y_test, num_classes, label_dict, plot_conf_matrix, plot_p_r_curve, beta)

    def evaluate_RamLessGenerator(self, model, ds, y, timeAug, batch_size, label_dict, num_channels, ramLessLoader, num_classes, plot_confusion_matrix = False, plot_p_r_curve = False, beta = 1):
        norm_scale = False 
        if ramLessLoader.scaler_name == "nornalize":
            norm_scale = True
        predictions = self.predict_RamLessGenerator(model, ds, y, batch_size, norm_scale, ramLessLoader, timeAug, num_channels)
        return self.evaluate_predictions(predictions, y, num_classes, label_dict, plot_confusion_matrix, plot_p_r_curve, beta)

    def evaluate_predictions(self, predictions, y_test, num_classes, label_dict, plot_conf_matrix, plot_p_r_curve, beta):
        predictions = np.abs(predictions)
        pp = pprint.PrettyPrinter(indent = 4)
        # The following line is used now that generators transform labels internally.
        #y_test = self.transform_labels(y_test, label_dict)
        
        #This code is used when RamLoader handles labels. Should be removed when the code works as expected.
        if predictions.shape[1] == 2:
            predictions = predictions[:,1]
            y_test = y_test[:,1]
        
        y_test = y_test[0:predictions.shape[0]]
        predictions = np.reshape(predictions,(predictions.shape[0]))
        rounded_predictions  = np.rint(predictions)
        assert len(rounded_predictions[rounded_predictions < 0]) == 0, f"Contains negative values: {rounded_predictions[rounded_predictions < 0]}"
        y_test = np.reshape(y_test, (y_test.shape[0]))
        assert predictions.shape == y_test.shape
        conf = tf.math.confusion_matrix(y_test, rounded_predictions, num_classes=num_classes)
        class_report = classification_report(y_test, rounded_predictions, target_names = self.handle_non_noise_dict(label_dict))
        precision, recall, fscore = np.round(precision_recall_fscore_support(y_test, rounded_predictions, beta = beta, pos_label = 1, average = 'binary', zero_division = 0.0)[0:3], 6)
        accuracy = self.calculate_accuracy(conf)
        if plot_p_r_curve:
            p,r,_ = precision_recall_curve(y_test, predictions, pos_label=1, sample_weight=None)
            average_precision = average_precision_score(y_test, predictions)
            baseline = len(y_test[y_test == 1])/len(y_test)
            self.plot_precision_recall_curve(p, r, average_precision, baseline)
        if plot_conf_matrix:
            self.plot_confusion_matrix(conf, self.handle_non_noise_dict(label_dict))
        pp.pprint(conf)
        print(class_report)
        gc.collect()
        return conf, class_report, accuracy, precision, recall, fscore

    def get_prediction_errors_indexes(self, rounded_predictions, y_test):
        incorrect_prediction_indexes = []
        for i in range(len(rounded_predictions)):
            if rounded_predictions[i] != y_test[i]:
                incorrect_prediction_indexes.append(i)
        return incorrect_prediction_indexes

    def get_false_negative_indexes(self, rounded_predictions, y_test): 
        false_negative_indexes = []
        for i in range(len(rounded_predictions)):
            if y_test[i] == 1 and rounded_predictions[i] == 0:
                false_negative_indexes.append(i)
        return false_negative_indexes

    def get_false_positive_indexes(self, rounded_predictions, y_test):
        false_positive_indexes = []
        for i in range(len(rounded_predictions)):
            if y_test[i] == 0 and rounded_predictions[i] == 1:
                false_positive_indexes.append(i)
        return false_positive_indexes

    def calculate_accuracy(self, conf):
        tn, fp, fn, tp = np.array(conf).ravel()
        print(tn, fp, fn, tp)
        accuracy = (tn + tp)/(tp + tn + fn + fp)
        return accuracy


    def plot_confusion_matrix(self, conf, label_dict):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        labels = list(self.handle_non_noise_dict(label_dict))
        for x in range(len(labels)):
            for y in range(len(labels)):
                ax.annotate(str(np.array(conf[x,y]).ravel()[0]), xy = (y, x), ha = 'center', va = 'center')
        print(labels)
        if labels == ["noise", "not_noise"]:
            classifier = "3N"
        else:
            classifier = "EE"
        print(classifier)
        offset = 0.5
        width = height = len(labels)
        ax.set_xlim(-offset, width - offset)
        ax.set_ylim(-offset, height - offset)
        plt.title(f'Confusion matrix of the {classifier} classifier')
        #ax.set_xticklabels([''] + labels)
        #ax.set_yticklabels([''] + labels)
        ax.hlines(y=np.arange(height+1)- offset, xmin=-offset, xmax=width-offset, color  = "black")
        ax.vlines(x=np.arange(width+1) - offset, ymin=-offset, ymax=height-offset, color = "black")
        plt.xticks(range(width), labels)
        plt.yticks(range(height), labels)
        ax.invert_yaxis()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()



    def plot_precision_recall_curve(self, precision, recall, average_precision, baseline):
        plt.clf()
        plt.plot(recall, precision, label = "Classifier")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.plot([0,1], [baseline, baseline], linestyle = '--', label = "Baseline")
        plt.title(f'Precision-Recall curve, AP: {np.round(average_precision, 2)}')
        plt.legend(loc="upper right")
        plt.show()

    def handle_non_noise_dict(self, label_dict):
        if len(list(set(label_dict.values()))) == 2 and len(list(label_dict.keys())) == 3:
            label_dict = {'noise' : 0, 'not_noise' : 1}
        return label_dict
            
    def get_steps_per_epoch(self, gen_set, batch_size):
        return int(np.floor(len(gen_set)/batch_size))
    
    def get_class_array(self, ds, class_dict):
        if len(list(class_dict.keys())) > len(set(class_dict.values())):
            class_array = np.zeros((len(ds), len(set(class_dict.values())))) 
            for idx, path_and_label in enumerate(ds):
                label = path_and_label[1]
                class_array[idx][class_dict.get(label)] = class_dict.get(label)
        else:
            class_array = np.zeros((len(ds), len(set(class_dict.values()))))       
            for idx, path_and_label in enumerate(ds):
                label = path_and_label[1]
                class_array[idx][class_dict.get(label)] = 1
        return class_array
    def f1_metric(self, y_true, y_pred):

        def recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = (true_positives + K.epsilon()) / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())
            return precision

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    def precision(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())
        return precision
    
    def recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = (true_positives + K.epsilon()) / (possible_positives + K.epsilon())
        return recall

    def fbeta(self, precision, recall, beta):
        return ((1+beta**2)*precision*recall)/((beta**2)*precision + recall)
    
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
    
    def generate_inceptionTime_build_args(self, input_shape, nr_classes, optimizer, use_residuals, use_bottleneck, 
                                          nr_modules, kernel_size, num_filters, bottleneck_size, shortcut_activation, 
                                          module_activation, module_output_activation, output_activation, reg_shortcut,
                                          reg_module, l1_r, l2_r):
        return {'input_shape' : input_shape,
                    'nr_classes' : nr_classes,
                    'optimizer' : optimizer,
                    'use_residuals' : use_residuals,
                    'use_bottleneck' : use_bottleneck,
                    'nr_modules' : nr_modules,
                    'kernel_size' : kernel_size,
                    'num_filters' : num_filters,
                    'bottleneck_size' : bottleneck_size,
                    'shortcut_activation'  : shortcut_activation,
                    'module_activation' : module_activation,
                    'module_output_activation' : module_output_activation,
                    'output_activation' : output_activation,
                    'reg_shortcut' : reg_shortcut,
                    'reg_module' : reg_module,
                    'l1_r' : l1_r,
                    'l2_r' : l2_r}
    
    def generate_build_model_args(self, model_nr_type, batch_size, dropout_rate, activation, output_layer_activation, l2_r, l1_r, 
                                  start_neurons, filters, kernel_size, padding, num_layers = 1,
                                  decay_sequence = [1], use_layerwise_dropout_batchnorm = True, is_lstm = False,
                                  num_classes = 3, channels = 3, timesteps = 6000):
        if type(model_nr_type) is str:
            input_shape = (channels, timesteps)
            if is_lstm:
                input_shape = (timesteps, channels)
            if type(decay_sequence) is not list:
                print(decay_sequence)
                #decay_sequence = json.loads(decay_sequence)
            return {"model_type" : model_nr_type,
                    "num_layers": num_layers,
                    "input_shape" : input_shape,
                    "num_classes" : num_classes,
                    "dropout_rate" : dropout_rate,
                    "activation" : activation,
                    "output_layer_activation" : output_layer_activation,
                    "l2_r" : l2_r,
                    "l1_r" : l1_r,
                    "full_regularizer" : True,
                    "start_neurons" : start_neurons,
                    "decay_sequence" : decay_sequence,
                    "filters" : filters,
                    "kernel_size" : kernel_size,
                    "padding" : padding,
                    "use_layerwise_dropout_batchnorm" : use_layerwise_dropout_batchnorm}
        return {"model_nr" : model_nr_type,
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
    
    def generate_model_compile_args(self, opt, nr_classes, balance_non_train_set, noise_not_noise):
        if nr_classes == 2:
            loss = "binary_crossentropy"
            acc = tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)
        else:
            raise Exception("Need to rewrite class_id!")
            loss = "categorical_crossentropy"
            acc = tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None)
        if noise_not_noise:
            return {"loss" : loss,
                    "optimizer" : opt,
                    "metrics" : [acc,
                                self.precision,
                                self.recall
                    ]}
        if balance_non_train_set:
            return {"loss" : loss,
                    "optimizer" : opt,
                    "metrics" : [acc,
                                self.precision,
                                self.recall]
                                }
        else:
            return {"loss" : loss,
                    "optimizer" : opt,
                    "metrics" : [self.precision,
                                 self.recall,
                                 acc]}
    
    def generate_gen_args(self, batch_size, detrend, use_scaler = False, scaler = None, 
                          use_time_augmentor = False, timeAug = None, use_noise_augmentor = False, 
                          noiseAug = None, use_highpass = False, highpass_freq = 0.3, num_channels = 3, is_lstm = False):
        return {    "num_channels" : num_channels,
                    "batch_size" : batch_size,
                    "detrend" : detrend,
                    "use_scaler" : use_scaler,
                    "scaler" : scaler,
                    "use_time_augmentor": use_time_augmentor,
                    "timeAug" : timeAug,
                    "use_noise_augmentor" : use_noise_augmentor,
                    "noiseAug" : noiseAug,
                    "use_highpass" : use_highpass,
                    "highpass_freq" : highpass_freq,
                    "is_lstm" : is_lstm}
    
    def generate_fit_args(self, train_ds, val_ds, loadData, batch_size, epoch, val_gen, workers, max_queue_size, use_tensorboard, use_liveplots, use_custom_callback, use_early_stopping, use_reduced_lr = False, y_val = None, beta = 1):
        callbacks = []
        if use_liveplots:
            #callbacks.append(PlotLossesKeras())
            print("")
        if use_tensorboard:
            log_dir = f"{glob_utils.base_dir}/Tensorboard_dir/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
            callbacks.append(tensorboard_callback)
        if use_custom_callback:
            custom_callback = CustomCallback(val_gen, self.get_steps_per_epoch(val_ds, batch_size), y_val, beta)
            callbacks.append(custom_callback)
        if use_early_stopping:
            if loadData.balance_non_train_set or loadData.noise_not_noise:
                earlystop = EarlyStopping(monitor = 'val_binary_accuracy',
                            min_delta = 0,
                            patience = 7,
                            verbose = 1,
                            mode = "max",
                            restore_best_weights = True)
                callbacks.append(earlystop)
                print("----------EarlyStop monitoring val_binary_accuracy------------") 
            else: 
                earlystop = EarlyStopping(monitor = f"val_f{beta}",
                            min_delta = 0,
                            patience = 7,
                            verbose = 1,
                            mode = "max",
                            restore_best_weights = True)
                callbacks.append(earlystop)
                print(f"----------EarlyStop monitoring val_f{beta}------------") 
        if use_reduced_lr:
            if loadData.balance_non_train_set or loadData.noise_not_noise:
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_binary_accuracy', 
                                                                    factor=0.5, patience=3, mode = "max",
                                                                    min_lr=0.00005, 
                                                                    verbose = 1))
                print("----------Reduce LR monitoring val_binary_accuracy------------") 
            else:
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor=f'val_f{beta}', 
                                                                      factor=0.5, patience=3, mode = "max",
                                                                      min_lr=0.00005, 
                                                                      verbose = 1))
                print(f"----------Reduce LR monitoring val_f{beta}------------")                                           
        
        return {"steps_per_epoch" : self.get_steps_per_epoch(train_ds, batch_size),
                "epochs" : epoch,
                "validation_data" : val_gen,
                "validation_steps" : self.get_steps_per_epoch(val_ds, batch_size),
                "verbose" : 1,
                "max_queue_size" : max_queue_size,
                "use_multiprocessing" : False, 
                "workers" : workers,
                "callbacks" : callbacks
                }
    
    def generate_meier_fit_args(self, train_ds, val_ds, loadData, batch_size, epoch, val_gen, workers, max_queue_size, use_tensorboard, use_liveplots, use_custom_callback, use_early_stopping, use_reduced_lr = False, y_val = None, beta = 1):
        callbacks = []
        print("------------ Meier ------------")
        if use_liveplots:
            #callbacks.append(PlotLossesKeras())
            print("")
        if use_tensorboard:
            log_dir = f"{glob_utils.base_dir}/Tensorboard_dir/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
            callbacks.append(tensorboard_callback)
        if use_custom_callback:
            custom_callback = CustomCallback(val_gen, self.get_steps_per_epoch(val_ds, batch_size), y_val, beta)
            callbacks.append(custom_callback)
        if use_early_stopping:
            if loadData.balance_non_train_set or loadData.noise_not_noise:
                earlystop = EarlyStopping(monitor = 'val_categorical_accuracy',
                            min_delta = 0,
                            patience = 7,
                            verbose = 1,
                            mode = "max",
                            restore_best_weights = True)
                callbacks.append(earlystop)
                print("----------Early stop monitoring val_categorical_accuracy----------")
            else: 
                earlystop = EarlyStopping(monitor = f'val_f{beta}',
                            min_delta = 0,
                            patience = 7,
                            verbose = 1,
                            mode = "max",
                            restore_best_weights = True)
                callbacks.append(earlystop)
                print(f"----------Early stop monitoring val_f{beta}----------")
        if use_reduced_lr:
            if loadData.balance_non_train_set or loadData.noise_not_noise:
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', mode = "max",
                                                                    factor=0.5, patience=3,
                                                                    min_lr=0.00005, 
                                                                    verbose = 1))
                print("----------Reduce LR monitoring val_categorical_accuracy----------")
            else:
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor=f'val_f{beta}', mode = "max",
                                                                        factor=0.5, patience=3,
                                                                        min_lr=0.00005, 
                                                                        verbose = 1))
                print(f"----------Reduce LR monitoring val_f{beta}------------")
        
        return {"steps_per_epoch" : self.get_steps_per_epoch(train_ds, batch_size),
                "epochs" : epoch,
                "validation_data" : val_gen,
                "validation_steps" : self.get_steps_per_epoch(val_ds, batch_size),
                "verbose" : 1,
                "max_queue_size" : max_queue_size,
                "use_multiprocessing" : False, 
                "workers" : workers,
                "callbacks" : callbacks
                }


    def get_optimizer(self, optimizer, learning_rate):
        if optimizer == "adam":
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if optimizer == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        if optimizer == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise Exception(f"{optimizer} not implemented into get_optimizer")    
    
    def plot_event(self, trace, info):
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
        stream.plot(number_of_ticks = 8)
    
    def get_max_decay_sequence(self, num_layers, units_or_num_filters, attempted_decay_sequence, num_classes):
        decay_sequence = attempted_decay_sequence
        num_out_neurons = num_classes
        if num_classes == 2:
            num_out_neurons = 1
        for idx, decay in enumerate(attempted_decay_sequence):
            if units_or_num_filters//decay < num_out_neurons:
                decay_sequence[idx] = decay_sequence[idx-1]
        return decay_sequence[0:num_layers]

    def handle_hyperparams(self, num_classes, **p):
        if "first_dense_units" in p.keys() and "second_dense_units" in p.keys():
            if p["first_dense_units"] < p["second_dense_units"]:
                p["second_dense_units"] = p["first_dense_units"]

        if "decay_sequence" in p.keys() or "growth_sequence" in p.keys():
            if "num_filters" in p.keys():
                units_or_num_filters = p["num_filters"]
            else:
                units_or_num_filters = p["units"]
            num_layers = p["num_layers"]
            if "decay_sequence" in p:
                p["decay_sequence"] = self.get_max_decay_sequence(num_layers,
                                                                  units_or_num_filters,
                                                                  p["decay_sequence"],
                                                                  num_classes)
            else:
                p["growth_sequence"] = p["growth_sequence"][:num_layers]
        return p

    def transform_labels(self, labels, label_dict):
        num_classes = len(set(label_dict.values()))
        lab = [label_dict.get(x) for x in np.reshape(labels,(labels.shape[0]))]
        lab = utils.to_categorical(lab, num_classes, dtype = np.int8)
        lab = lab[:,1]
        return np.reshape(lab, (lab.shape[0], 1))

    def load_model(self, path):
        acc = tf.keras.metrics.BinaryAccuracy(name="binary_accuacy", dtype = None, threshold = 0.5)
        precision = self.precision
        recall = self.recall
        return tf.keras.models.load_model(path, custom_objects = {"accuracy" : acc,
                                                                  "precision": precision,
                                                                  "recall" : recall})


    def progress_bar(self, current, total, text, barLength = 40):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('%s: [%s%s] %d %%' % (text, arrow, spaces, percent), end='\r')