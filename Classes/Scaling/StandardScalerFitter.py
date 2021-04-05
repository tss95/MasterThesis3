from sklearn.preprocessing import StandardScaler
from .ScalerFitter import ScalerFitter

class StandardScalerFitter(ScalerFitter):

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_scaler_ram(self, traces):
        num_samples = traces.shape[0]
        for i in range(num_samples):
            self.scaler.partial_fit(traces[i])
            self.progress_bar(i + 1, num_samples)
        return self.scaler
        