from sklearn.preprocessing import StandardScaler
from .ScalerFitter import ScalerFitter
import numpy as np

class StandardScalerFitter(ScalerFitter):

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_scaler_ram(self, traces):
        num_samples = traces.shape[0]
        for i in range(num_samples):
            self.scaler.partial_fit(np.transpose(traces[i]))
            self.progress_bar(i + 1, num_samples)
        return self.scaler
        
    def partial_fit_ramless(self, trace):
        self.scaler.partial_fit(np.transpose(trace))