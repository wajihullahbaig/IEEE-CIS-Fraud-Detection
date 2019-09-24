# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:00:33 2019

@author: TNDUser
"""
import numpy as np
class FeatureTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
        self.count = 0;
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        Xt = X[X.days == self.count]
        n_samples = len(Xt)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            self.count +=1
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]