import tensorflow as tf
import gc
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np

class CustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, val_gen, steps, y_val):
        self.val_gen = val_gen
        self.steps = steps
        self.y_val = y_val
        self.val_f1s = 0
    
    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        val_predict = (np.asarray(self.model.predict(x = self.val_gen, steps = self.steps))).round()
        val_targ = self.y_val[:len(val_predict)]
        _val_f1 = f1_score(val_targ, val_predict)
        self.val_f1s = _val_f1
        logs["val_f1"] = _val_f1

        print("- val_f1: %f" %(_val_f1))
        gc.collect()
        return
        
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