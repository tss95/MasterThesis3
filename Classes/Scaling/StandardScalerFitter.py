from sklearn.preprocessing import StandardScaler
from .ScalerFitter import ScalerFitter

class StandardScalerFitter(ScalerFitter):
    
        def __init__(self, train_ds):
            self.train_ds = train_ds
            self.scaler = StandardScaler()
            
            
        def fit_scaler(self, test = False, detrend = False):
            ds = self.train_ds
            num_samples, channels, timesteps = self.get_trace_shape_no_cast(ds)

            if test:
                num_samples = int(num_samples * 0.1)

            for i in range(num_samples):
                X = self.path_to_trace(ds[i][0])[0]
                if detrend:
                    
                    X = self.detrend_trace(X)
                self.progress_bar(i, num_samples)
                self.scaler.partial_fit(X)
            return self.scaler