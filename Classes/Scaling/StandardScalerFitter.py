from sklearn.preprocessing import StandardScaler
from .ScalerFitter import ScalerFitter

class StandardScalerFitter(ScalerFitter):

    def __init__(self, train_ds, timeAug):
        self.timeAug = timeAug
        self.train_ds = train_ds
        self.scaler = StandardScaler()
        self.use_time_augmentor = False
        if timeAug != None:
            self.use_time_augmentor = True


    def fit_scaler(self, detrend = False):
        ds = self.train_ds
        num_samples, channels, timesteps = self.get_trace_shape_no_cast(ds, self.use_time_augmentor)
        for i in range(num_samples):
            X = self.path_to_trace(ds[i][0], ds[i][2])[0]
            if detrend:
                X = self.detrend_trace(X)
            self.progress_bar(i, num_samples)
            self.scaler.partial_fit(X)
        return self.scaler

    def fit_scaler_ram(self, traces):
        num_samples = traces.shape[0]
        for i in range(num_samples):
            self.scaler.partial_fit(traces[i])
            self.progress_bar(i + 1, num_samples)
        return self.scaler
        