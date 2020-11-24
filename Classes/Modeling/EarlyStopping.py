from keras.callbacks import EarlyStopping

# This is taken directly from: https://medium.com/towards-artificial-intelligence/keras-callbacks-explained-in-three-minutes-846a43b44a16

class EarlyStopping:
    
    def _init_(self, monitor = 'val_loss', patience = 3, verbose = 1, restore_best_weights = True):
           self.early_stopping_callback = EarlyStopping(monitor = monitor,
                                                  min_delta = min_delta,
                                                  patience = patience,
                                                  verbose = verbose,
                                                  restore_best_weights = restore_best_weights)
            return self.early_stopping_callback