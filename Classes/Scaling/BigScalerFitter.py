from joblib import dump, load
from os import path
import sys

from .BigDataLoader import BigDataLoader

class BigScalerFitter():
    
    def __init__(self, train_ds, scaler, data_loader):
        self.train_ds = train_ds
        self.scaler = scaler
        self.loader = data_loader

    def subsample(self, ds, shuffle = False, subsample_rate = 0.2):
        channels, timesteps = self.data_loader.handler.get_trace_shape(ds)
        num_samples = len(ds)
        num_samples = int(num_samples*subsample_rate)
        if shuffle:
            ds = ds.sample(frac = 1, random_state = self.loader.seed)
        subsample_X = np.empty((num_samples, channels, timesteps))
        subsample_y = np.empty((num_samples,1), dtype=np.dtype('<U10'))
        for idx, name, label in enumerate(ds):
            subsample_X[idx] = self.handler.name_to_trace(name)
            subsample_y[idx] = label
        return subsample_X, subsample_y

    def transform_subsample(self, ds, subsample_rate = 0.2, shuffle = False):
        subsamples_X, subsamples_y = self.subsample(ds, shuffle, subsample_rate)
        for i in range(len(subsamples_X)):
            subsamples_X[i] = self.scaler.transform(subsamples_X[i])
        return subsamples_X, subsamples_y


    def transform_sample(self, sample_X):
        return self.scaler.transform(sample_X)
    
    def save_fit(self, scaler):
        dump(scaler, f'{self.scaler_folder}\{self.scaler_name}_{self.handler.seed}')
    
    def load_fit(self, scaler_type):
        if path.exists(f'{self.scaler_folder}\{scaler_type}_{self.handler.seed}'):
            return load(f'{self.scaler_folder}\{self.scaler_name}_{self.handler.seed}')
        else:
            return None
    
    def progress_bar(self, current, total, barLength = 20):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('Fitting scaler progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')