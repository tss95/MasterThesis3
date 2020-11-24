from sklearn.preprocessing import StandardScaler
from .BigScalerFitter import BigScalerFitter
from .BigDataHandler import BigDataHandler

class BigStandardScalerFitter(BigScalerFitter):
    
    def __init__(self, train_ds, handler):
        self.train_ds = train_ds
        self.handler = handler
        self.scaler_name = "BigStandardScaler"
        self.scaler_folder = self.handler.source_path + "\MasterThesis\Scalers"
        self.scaler = StandardScaler()

    def fit_scaler(self):
        if self.load_fit(self.scaler_name) != None:
            self.scaler = self.load_fit(self.scaler_name)
            return self.scaler
        else:
            ds = self.train_ds
            channels, timesteps = self.handler.get_trace_shape(ds)
            num_samples = len(ds)
            ds = np.array(ds)

            for i in range(num_samples):
                self.progress_bar(i, num_samples, "Fitting scaler")
                X = self.handler.name_to_trace(ds[i][0])

                self.scaler.partial_fit(X)
            self.save_fit(self.scaler)
            return self.scaler