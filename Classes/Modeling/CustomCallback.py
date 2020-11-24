import tensorflow as tf
from tensorflow import keras

class CustomCallback(keras.callbacks.Callback):
    
    def __init__(self, gen):
        self.gen = gen
        self.full_training_logs = []
    
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())

    def on_train_end(self, logs=None):
        keys = list(logs.keys())

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        self.full_training_logs = []
        
    def on_test_begin(self, logs=None):
        keys = list(logs.keys())

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
    def on_predict_end(self, logs=None):
        keys = list(logs.keys())

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        batch_samples = self.gen.batch_samples
        logs['batch_samples'] = batch_samples
        self.full_training_logs.append(logs)
        

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())