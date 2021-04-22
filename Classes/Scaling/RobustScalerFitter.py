from sklearn.preprocessing import RobustScaler
from .ScalerFitter import ScalerFitter
import time
import numpy as np

class RobustScalerFitter(ScalerFitter):

    def __init__(self):
        self.scaler = RobustScaler()


    def fit_scaler_ram(self, traces):
        num_samples = traces.shape[0]
        start = time.time()
        for i in range(num_samples):
            self.scaler.fit(np.transpose(traces[i]))
            self.progress_bar(i + 1, num_samples)
        end = time.time()
        print(f"Completed fitting robust scaler. Total time taken: {int(end-start)} seconds")
        print(f"Seconds per event: {(end-start)/num_samples}")
        return self.scaler
        
    def partial_fit_ramless(self, trace):
        self.scaler.fit(np.transpose(trace))