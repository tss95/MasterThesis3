from tensorflow.keras.utils import Sequence

class RamLessGen(Sequence):
    'Generates data for Keras'
    def __init__(self, ds, labels, timeAug, batch_size, ramLessLoader, num_channels, norm_scale = False, shuffle=False):
        'Initialization'
        self.ds = ds
        self.labels = labels
        self.batch_size = batch_size
        self.ramLessLoader = ramLessLoader
        self.timeAug = timeAug
        self.noiseAug = self.ramLessLoader.noiseAug
        self.num_channels = num_channels
        self.norm_scale = norm_scale
        self.shuffle = shuffle
        self.timesteps = 9460
        if self.timeAug is not None:
            self.timesteps = 6000
        self.on_epoch_end()

    def __len__(self):
        import numpy as np
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_traces = self.__data_generation(self.ds[index*self.batch_size:(index+1)*self.batch_size])
        batch_labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        return batch_traces, batch_labels

    def on_epoch_end(self):
        import numpy as np
        'Updates indexes after each epoch'
        pass

    def on_train_end(self):
        import gc
        self.traces = None
        self.labels = None
        gc.collect()

    def __data_generation(self, batch_ds):
        import numpy as np
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_traces = self.ramLessLoader.load_batch(batch_ds, self.timeAug, np.empty((self.batch_size, self.num_channels, self.timesteps)))
        if self.noiseAug is not None:
            if self.norm_scale:
                batch_traces = self.noiseAug.batch_augment_noise(batch_traces, 0, self.noiseAug.noise_std*(1/20))
            else:
                batch_traces = self.noiseAug.batch_augment_noise(batch_traces, 0, self.noiseAug.noise_std*(1/15))
        batch_traces = batch_traces[:][:,0:self.num_channels]
        batch_traces = np.reshape(batch_traces, (batch_traces.shape[0], batch_traces.shape[2], batch_traces.shape[1]))
        return batch_traces
