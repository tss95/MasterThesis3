from tensorflow.keras.utils import Sequence

class RamGen(Sequence):
    'Generates data for Keras'
    def __init__(self, traces, labels, batch_size, noiseAug, num_channels, norm_scale = False, shuffle=True):
        'Initialization'
        self.traces = traces
        self.labels = labels
        self.batch_size = batch_size
        self.noiseAug = noiseAug
        self.num_channels = num_channels
        self.norm_scale = norm_scale
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        import numpy as np
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # Find list of IDs
        batch_traces = self.traces[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        batch_traces = self.__data_generation(batch_traces)
        return batch_traces, batch_labels

    def on_epoch_end(self):
        import numpy as np
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def on_train_end(self):
        import gc
        self.traces = None
        self.labels = None
        gc.collect()

    def __data_generation(self, batch_traces):
        import numpy as np
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.noiseAug is not None:
            if self.norm_scale:
                batch_traces = self.noiseAug.batch_augment_noise(batch_traces, 0, self.noiseAug.noise_std*(1/20))
            else:
                batch_traces = self.noiseAug.batch_augment_noise(batch_traces, 0, self.noiseAug.noise_std*(1/15))
        batch_traces = batch_traces[:][:,0:self.num_channels]
        batch_traces = np.reshape(batch_traces, (batch_traces.shape[0], batch_traces.shape[2], batch_traces.shape[1]))
        return batch_traces


"""
def get_batch(num_workers, max_queue_size, **kwargs):
    from tensorflow.keras.utils import GeneratorEnqueuer
    import time
    # Code similar to: https://github.com/watersink/Character-Segmentation/blob/master/data_generator.py
    try:
        enq = GeneratorEnqueuer(RamGen(**kwargs).__getitem__, use_multiprocessing = False)
        enq.start(workers = num_workers, max_queue_size = max_queue_size)
        gen_output = None
        while True:
            while enq.is_running():
                if not enq.queue.empty():
                    gen_output = enq.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield gen_output
            gen_output = None
    finally:
        if enq is not None:
            enq.stop()
"""