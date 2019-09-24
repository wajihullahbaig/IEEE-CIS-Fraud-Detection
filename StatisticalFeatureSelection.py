#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 12:22:17 2019

@author: wajih
"""
# https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again?scriptVersionId=18889353 
import pandas as pd
from scipy.stats  import ks_2samp
from tqdm import tqdm

class StatisticalFeatureSelection():
    def __init__(self):
        pass
    
    def eliminate_features_on_ks2(train,test,features):
        print("Eliminating features on ks_2samp")
        features_check = []
        for i in tqdm(features):
            features_check.append(ks_2samp(test[i], train[i])[1])
        
        features_check = pd.Series(features_check, index=features).sort_values() 
        features_discard = list(features_check[features_check==0].index)
        print(len(features_discard))
        print("Eliminating features:")
        print(features_discard)
        return features_discard