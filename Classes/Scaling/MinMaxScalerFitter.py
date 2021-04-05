from .ScalerFitter import ScalerFitter
from sklearn.preprocessing import MinMaxScaler

class MinMaxScalerFitter(ScalerFitter):

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_scaler_ram(self, traces):
        num_samples = traces.shape[0]
        for i in range(num_samples):
            self.scaler.partial_fit(traces[i])
            self.progress_bar(i + 1, num_samples)
        return self.scaler