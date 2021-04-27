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
        tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        print(f"------------------Train end RAM usage: {used_m}/{tot_m} (Free: {free_m})------------------")
        

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        val_predict = (np.asarray(self.model.predict(x = self.val_gen, steps = self.steps, max_queue_size = 15, workers = 8, use_multiprocessing = False))).round()
        val_targ = self.y_val[:len(val_predict)]
        _val_f = np.round(fbeta_score(val_targ, val_predict, beta = self.beta), 4 )
        self.val_fs = _val_f
        logs[f"val_f{self.beta}"] = _val_f

        print(f"- val_f{self.beta}: {_val_f}")
        gc.collect()
        tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        print(f"------------------Epoch end RAM usage: {used_m}/{tot_m} (Free: {free_m})------------------")
        
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