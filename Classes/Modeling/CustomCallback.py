import tensorflow as tf
import gc
import os
from sklearn.metrics import fbeta_score
import numpy as np

class CustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, val_gen, steps, y_val, beta):
        self.val_gen = val_gen
        self.steps = steps
        self.y_val = y_val
        self.val_fs = 0
        self.beta = beta
    
    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        self.val_gen = None
        gc.collect()
        

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(x = self.val_gen, steps = self.steps, max_queue_size = 15, workers = 8, use_multiprocessing = False)
        val_predict = np.asarray(pred).round()
        val_targ = self.y_val[:len(val_predict)]
        if val_predict.shape[1] == 2:
            val_predict = val_predict[:,1]
            val_targ = val_targ[:,1]
        _val_f = np.round(fbeta_score(val_targ, val_predict, pos_label = 1, beta = self.beta, average = 'binary'), 4 )
        self.val_fs = _val_f
        logs[f"val_f{self.beta}"] = _val_f

        print(f"- val_f{self.beta}: {_val_f}")
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