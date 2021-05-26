
class ScalerFitter():

    """
    Parent class for the scalers. Holds functions used by all. Should not be initialized.
    """
    
    def __init__(self, scaler):
        self.scaler = scaler

    def transform_sample(self, sample_X):
        return self.scaler.transform(sample_X)
    
    def progress_bar(self, current, total, barLength = 20):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('Fitting scaler progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')