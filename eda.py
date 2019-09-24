# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:25:25 2019

@author: TNDUser
"""

import pandas as pd
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from BlockingTimeSeriesSplit import BlockingTimeSeriesSplit
from TimeSeriesSplitter import TimeSeriesSplit
from FeatureTimeSeriesSplit import FeatureTimeSeriesSplit as FT
from FeatureEngineering import FeatureEngineering as FE
from FeatureSelection import FeatureSelection as FS
from RecursiveFeatureSelection import RecursiveFeatureSelection as RFS
from StatisticalFeatureSelection import StatisticalFeatureSelection as SFS
from DataSampling import DataSampling as DS
import warnings
warnings.filterwarnings("ignore")

print("Reading csv's...")    

local_test = False
use_sampling = True
if not local_test:
    train = pd.read_csv('train_transaction.csv')
    test = pd.read_csv('test_transaction.csv')
    traini = pd.read_csv('train_identity.csv') 
    testi = pd.read_csv('test_identity.csv')
    train = pd.merge(train, traini, on='TransactionID', how='left')
    test = pd.merge(test, testi, on='TransactionID', how='left')
    del traini
    del testi
    
else:

    train = pd.read_csv('train_transaction.csv')
    traini = pd.read_csv('train_identity.csv') 
    train = pd.merge(train, traini, on='TransactionID', how='left')
    test = train.sample(frac = 0.7,random_state = 99)
    train = train[~train.index.isin(test.index)]
    del traini


print("Done!")



print("Feature engineering...")
train = FE.reduce_mem_usage(train)
test = FE.reduce_mem_usage(test)


        
FE.make_ymdhd_feature(train)
FE.make_ymdhd_feature(test)


train = train.sort_values('day')
test = test.sort_values('day')
  

    
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y

feature_group = ['V']
all_cols = FS.get_feature_names(feature_group,train.columns)  
all_cols= SFS.eliminate_features_on_ks2(train,test,all_cols)
for col in all_cols:
# Compute ECDF for versicolor data: x_vers, y_vers
    x_set0, y_set0 = ecdf(train[train.isFraud==0][col])
    x_set1, y_set1 = ecdf(train[train.isFraud==1][col])
    x0_label = 'positives:'+str(len(y_set1)/len(train))
    x1_label = 'negatives:'+str(len(y_set0)/len(train))
    plt.figure()
    f, ax = plt.subplots(figsize=(10, 8))
    # Generate plot
    plt.plot(x_set0, y_set0, marker='_', linestyle='none', alpha=0.5,color='r',label=x0_label)    
    plt.plot(x_set1, y_set1, marker='+', linestyle='none', alpha=0.5,color='g',label=x1_label)    
    plt.legend()
    plt.title(col)


    plt.savefig("eda-ecdf-images/"+col)   
    plt.show()