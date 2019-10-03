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
    
    def get_features_from_recursive_feature_elimination(X,y,elimination_step):
        params = {    'nthread':-1,
    'objective': "binary",
    'metric':"auc",
    'boosting':'gbdt',
    'max_depth' : 9,
    'num_leaves': 56,
    'learning_rate': 0.03,
    'bagging_freq':5,
    'bagging_fraction': 1.0,
    'feature_fraction' : 0.8,
    'min_child_samples': 10,
    'tree_learner': "serial",
    'n_estimators':500,
    'seed': 42,
    'feature_fraction_seed' : 42,
    'bagging_seed' : 42,
    'drop_seed':42,
    'data_random_seed' : 42,
    'boost_from_average' : True
             }
        print("Performing Recursive Feature Selection...")
        clf = lgb.LGBMClassifier(**params)
        rfe = RFECV(estimator=clf, step=elimination_step, cv=2, scoring='roc_auc', verbose=2)
        rfe.fit(X, y)
        print('Optimal number of features:', rfe.n_features_)
        plt.figure(figsize=(14, 8))
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score")
        plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
        plt.show()
        plt.close("all")
        best_features = []
        for col in X.columns[rfe.ranking_ == 1]:
            best_features.append(col)
            #print(col)
        
        print("Done!")
        return best_features