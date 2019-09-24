#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:36:13 2019

@author: wajih
"""
# https://www.kaggle.com/nroman/recursive-feature-elimination
import matplotlib.pyplot as plt
from TimeSeriesSplitter import TimeSeriesSplit
from sklearn.feature_selection import RFECV
import lightgbm as lgb

class RecursiveFeatureSelection():
    def __init__(self):
        pass
    
    def get_features_from_recursive_feature_elimination(X,y):
        params = {'num_leaves': 64,
              'min_child_weight': 0.03,
              'colsample_bytree': 1.0, #feature_fraction
              'subsample': 1.0, #bagging_fraction
              'min_child_samples': 10,
              'objective': 'binary',
              'max_depth': 9,
              'learning_rate': 0.02,
              "boosting_type": "gbdt",
              "bagging_seed": 11,
              "metric": 'auc',
              "verbosity": -1,
              'random_state': 47
             }
        print("Performing Recursive Feature Selection...")
        clf = lgb.LGBMClassifier(**params)
        rfe = RFECV(estimator=clf, step=16, cv=TimeSeriesSplit(n_splits=7), scoring='roc_auc', verbose=2)
        rfe.fit(X, y)
        print('Optimal number of features:', rfe.n_features_)
#        plt.figure(figsize=(14, 8))
#        plt.xlabel("Number of features selected")
#        plt.ylabel("Cross validation score")
#        plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
#        plt.show()
#        plt.close("all")
        best_features = []
        for col in X.columns[rfe.ranking_ == 1]:
            best_features.append(col)
            #print(col)
        
        print("Done!")
        return best_features