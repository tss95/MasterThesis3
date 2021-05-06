import numpy as np
import pandas as pd
import json
import h5py
import matplotlib.pyplot as plt
from obspy import Stream, Trace, UTCDateTime
import os
import csv
from tensorflow.keras import utils
import seaborn as sns
import time
import tables
import random

import tensorflow as tf


class ScalerFitter():
    
    def __init__(self, scaler):
        self.scaler = scaler

    def transform_sample(self, sample_X):
        return self.scaler.transform(sample_X)
    
    def progress_bar(self, current, total, barLength = 20):
        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * barLength - 1) + '>'
        spaces  = ' ' * (barLength - len(arrow))
        print('Fitting scaler progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')