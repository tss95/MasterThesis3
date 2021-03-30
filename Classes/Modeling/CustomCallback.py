import tensorflow as tf
import gc

class CustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self):
        pass
    
    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        
    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass
    def on_predict_begin(self, logs=None):
        pass
    def on_predict_end(self, logs=None):
        gc.collect()
        

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass
        

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass