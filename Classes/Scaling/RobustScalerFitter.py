from sklearn.preprocessing import RobustScaler
from .ScalerFitter import ScalerFitter
import time

class RobustScalerFitter(ScalerFitter):

    def __init__(self):
        self.scaler = RobustScaler()


    def fit_scaler_ram(self, traces):
        print("I SHOULD NOT BE USED")
        num_samples = traces.shape[0]
        start = time.time()
        for i in range(num_samples):
            self.scaler.fit(traces[i])
            self.progress_bar(i + 1, num_samples)
        end = time.time()
        print(f"Completed fitting robust scaler. Total time taken: {int(end-start)} seconds")
        print(f"Seconds per event: {(end-start)/num_samples}")
        return self.scaler
        
    def fit_transform_trace(self, trace):
        return self.scaler.fit_transform(trace)