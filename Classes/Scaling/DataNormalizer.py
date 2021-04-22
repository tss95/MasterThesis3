from sklearn.preprocessing import Normalizer
from Classes.Scaling.ScalerFitter import ScalerFitter
import numpy as np

class DataNormalizer(ScalerFitter):

    def __init__(self):
        self.scaler = Normalizer()


    def fit_transform_trace(self, trace):
        return self.scaler.fit_transform(trace)

        