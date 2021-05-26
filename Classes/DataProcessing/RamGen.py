from tensorflow.keras.utils import Sequence
import numpy as np
from tensorflow.keras import utils

class RamGen(Sequence):
    """
    Class used as the generator for models. This is used when the data is stored in RAM. Only performs reduction of channels, noiseAug and transforms labels when using true test set.
    This implementation allows the use of several workers. Allows for multiprocessing, but this worsens current memory leak.
    Code has been inspired by: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.

    PARAMETERS:
    -------------------------------------------------------------------------------------------
    traces: (np.array)          Loaded and transformed waveforms.
    labels: (np.array)          Labels corresponding to the order of traces.
    batch_size: (int)           Size of the batches to generate.
    noiseAug: (object)          Fitted noise augmentor. Can be None if it is not used.
    num_channels: (int)         Option to train and evaluate the models on a reduced number of channels. P-beam is the last channel to be removed.
    norm_scale: (bool)          Whether or not normalize scaler is used.
    shuffle: (bool)             Unused. Has no effect.
    label_dict: (dict)          Dictionary used to translate string labels to a format the model understands.
    final_eval: (bool)          Whether or not the true test set is being used.
    """



    def __init__(self, traces, labels, batch_size, noiseAug, num_channels, norm_scale = False, shuffle=False, label_dict = None, final_eval = False):
        self.traces = traces
        self.labels = labels
        self.batch_size = batch_size
        self.noiseAug = noiseAug
        self.num_channels = num_channels
        self.norm_scale = norm_scale
        self.shuffle = shuffle
        self.label_dict = label_dict
        if label_dict is not None:
            self.num_classes = len(set(self.label_dict.values()))
        else:
            self.num_classes = 2
        self.final_eval = final_eval
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        batch_traces = self.traces[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        batch_traces, batch_labels = self.__data_generation(batch_traces, batch_labels)
        return batch_traces, batch_labels

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.labels))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def on_train_end(self):
        import gc
        self.traces = None
        self.labels = None
        gc.collect()

    def __data_generation(self, batch_traces, batch_labels):
        if self.noiseAug is not None:
            if self.norm_scale:
                batch_traces = self.noiseAug.batch_augment_noise(batch_traces, 0, self.noiseAug.noise_std*(1/20))
            else:
                batch_traces = self.noiseAug.batch_augment_noise(batch_traces, 0, self.noiseAug.noise_std*(1/15))
        batch_traces = batch_traces[:][:,0:self.num_channels]
        if self.final_eval:
            batch_labels = self.__transform_labels(batch_labels)
        if self.batch_size == 1:
            batch_traces = np.reshape(batch_traces, (1, batch_traces.shape[2], batch_traces.shape[1]))
        else:
            batch_traces = np.reshape(batch_traces, (batch_traces.shape[0], batch_traces.shape[2], batch_traces.shape[1]))
        return batch_traces, batch_labels

    def __transform_labels(self, labels):
        if self.label_dict != None:
            lab = [self.label_dict.get(x) for x in labels]
            lab = utils.to_categorical(lab, self.num_classes, dtype = np.int8)
            lab = lab[:,1]
            return np.reshape(lab, (lab.shape[0], 1))
        else:
            return np.empty((len(labels), 1))

