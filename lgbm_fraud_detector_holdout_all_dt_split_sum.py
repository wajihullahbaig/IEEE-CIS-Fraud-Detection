#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 06:26:58 2019

@author: wajih
"""

import pandas as pd
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
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
use_sampling = 0
use_cyclical_features = True
use_V_log_features = True
use_monthly_splits = True
visualize_corr_clusters = False 

granularity_to_use  ={}
granularity_to_use["month"] = DS.per_month_down_sampling
granularity_to_use["dow"] = DS.per_week_down_sampling
granularity_to_use["day"] = DS.per_day_down_sampling
granularity_to_use["hour"] = DS.per_hour_down_sampling

granularity_key = "day"


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
    
if use_sampling == 0:
    train,_ = granularity_to_use[granularity_key](train)   
elif use_sampling == 1:
    train_copy = train.copy()
    train_month,_ = DS.per_month_down_sampling(train)
    train = train_copy.copy()
    train_dow,_ = DS.per_week_down_sampling(train)
    train = train_copy.copy()
    train_day,_ = DS.per_day_down_sampling(train)
    train = train_copy.copy()
    train_hour,_ = DS.per_hour_down_sampling(train)
    del train
    del train_copy
    train = pd.concat([train_month,train_dow,train_day,train_hour],ignore_index=True)
    del train_month,train_dow,train_day,train_hour
    train.drop_duplicates(keep='first', inplace=True)
    train = train.reset_index(drop=True)
elif use_sampling == 2:
    train,_ = DS.per_day_up_sampling(train,50)
    train,_ = DS.per_day_down_sampling(train)
elif use_sampling == 3:
    train = DS.one_go_down_sampling(train,0.5)
elif use_sampling == 4:
    positive_class = train[train.isFraud==1].copy()
    extension = DS.reverse_column_values(positive_class,['TransactionAmt'])
    train = pd.concat([extension,train],ignore_index=True)
    del extension;positive_class
    train.drop_duplicates(keep='first', inplace=True)
    train = train.reset_index(drop=True)
    train,_ = granularity_to_use[granularity_key](train)   
    
print('Training size:',len(train))
print("isFraud = 0:",len(train[train.isFraud==0]))
print("isFraud = 1:",len(train[train.isFraud==1]))

if use_cyclical_features:
    cyclical_features = {'day':31,'hour':23,'dow':6}
    cyclical_features = FE.encode_cyclical_features(train,test,cyclical_features)    
    
FE.expand_id31_and_DeviceInfo(train,test)    
cat_cols = FS.get_categorical_columns(train)     
sorted_cat_cols = FS.sort_cat_cols_with_uniqueness(train,cat_cols)
agg_cols = sorted_cat_cols[sorted_cat_cols["nunique"] >= 40]["feature"].values.tolist()
FE.aggregation_on_train_map_on_test(train,test,agg_cols,'var','isFraud')   
FE.noise_reset(train,test,['card1'],10)  
FE.encode_frequency(train,test,cat_cols)
FE.encode_label(train,test,cat_cols)   

FE.split_decimal_and_fractional_part(train,test,["TransactionAmt"]) 
train["ProductCD_fractioned"] = train["ProductCD"].astype(str) + train["TransactionAmt_fractional"].astype(str) 
test["ProductCD_fractioned"] = test["ProductCD"].astype(str) + test["TransactionAmt_fractional"].astype(str)
FE.aggregation_on_train_map_on_test(train,test,['ProductCD_fractioned'],'var','isFraud')  
FE.encode_frequency(train,test,['ProductCD_fractioned'])
FE.encode_label(train,test,['ProductCD_fractioned'])  

FE.make_transactionamt_features(train,test,'TransactionAmt','uid1') 
FE.encode_frequency(train,test,['uid1'])
FE.encode_label(train,test,['uid1']) 
FE.make_transactionamt_features(train,test,'TransactionAmt_fractional','uid2')  
FE.encode_frequency(train,test,['uid2'])
FE.encode_label(train,test,['uid2'])
    
if use_cyclical_features:
    feature_group = [granularity_key+'_']
else:
    feature_group = [granularity_key]
time_features = FS.get_feature_names(feature_group,train.columns)   
t_cols =  []
for col in tqdm(time_features):
    t_cols.append(col+'_'+"ProductCD")
    train[col+'_'+"ProductCD"] = train[col].astype(str) + train["ProductCD"].astype(str) 
    test[col+'_'+"ProductCD"] = test[col].astype(str) + test["ProductCD"].astype(str)
FE.aggregation_on_train_map_on_test(train,test,t_cols,'var','isFraud')
FE.encode_frequency(train,test,t_cols)      
FE.encode_label(train,test,t_cols)    

t_cols =  []
for col in tqdm(time_features):
    t_cols.append(col+'_'+"TransactionAmt_fractional")
    train[col+'_'+"TransactionAmt_fractional"] = train[col].astype(str) + train["TransactionAmt_fractional"].astype(str) 
    test[col+'_'+"TransactionAmt_fractional"] = test[col].astype(str) + test["TransactionAmt_fractional"].astype(str)
FE.aggregation_on_train_map_on_test(train,test,t_cols,'var','isFraud')
FE.encode_frequency(train,test,t_cols)      
FE.encode_label(train,test,t_cols)  

feature_group = ['uid','addr','dist','card','ProductCD','C','M','V','id_','D','P_','R_','Device']
all_cols = FS.get_feature_names(feature_group,train.columns)   
train.replace([np.inf, -np.inf], np.nan,inplace=True)
test.replace([np.inf, -np.inf], np.nan,inplace=True) 
droppable_cols = FS.get_drop_columns_manynans_onlyones(train,test,all_cols,threshold = 0.90)
train.drop(droppable_cols,inplace=True,axis=1)
test.drop(droppable_cols,inplace=True,axis=1)
train.fillna(-999, inplace=True)
test.fillna(-999, inplace=True)    
if use_V_log_features:
    V_cols = FS.get_high_difference_V_columns(train)
    FE.create_log_feature(train,V_cols,1000)     
    FE.create_log_feature(test,V_cols,1000)
    col_list = []
    for col in tqdm(V_cols):
        col_list.append(col+'_ProductCD')
        train[col+'_ProductCD'] = train[col].astype(str) + train["ProductCD"].astype(str) 
        test[col+'_ProductCD'] = test[col].astype(str) + test["ProductCD"].astype(str)
#    FE.aggregation_on_train_map_on_test(train,test,col_list,'var','isFraud')  
    FE.encode_frequency(train,test,col_list)
    FE.encode_label(train,test,col_list)  
    
feature_group = ['C','D']
all_cols = FS.get_feature_names(feature_group,train.columns)      
col_list = []
for col in tqdm(all_cols):
    col_list.append(col+'_ProductCD')
    train[col+'_ProductCD'] = train[col].astype(str) + train["ProductCD"].astype(str) 
    test[col+'_ProductCD'] = test[col].astype(str) + test["ProductCD"].astype(str)
#FE.aggregation_on_train_map_on_test(train,test,col_list,'var','isFraud')  
FE.encode_frequency(train,test,col_list)
FE.encode_label(train,test,col_list)        
#  
#    df = train[train.DeviceInfo<=1000] 
#    plt.scatter(df[df.isFraud==0].TransactionDT,df[df.isFraud==0]['TransactionAmt_fractional'])
#    plt.scatter(df[df.isFraud==1].TransactionDT,df[df.isFraud==1]['TransactionAmt_fractional'])

#feature_group = ['uid','addr','dist','card','ProductCD','C','M','V','id_','D','P_','R_','Device']
#all_cols = FS.get_feature_names(feature_group,train.columns)  
droppable_cols = SFS.eliminate_features_on_ks2(train,test,col_list)
train.drop(droppable_cols,inplace=True,axis=1)
test.drop(droppable_cols,inplace=True,axis=1)


if use_cyclical_features:
    feature_group = ['C','V','D','uid','card','M','id_','TransactionAmt','addr','R_', 'ProductCD','dist','P_','hour_','dow_','Device','day_']
else:
    feature_group = ['C','V','D','uid','card','M','id_','TransactionAmt','addr','R_', 'ProductCD','dist','P_','hour','dow','Device','day']
feature_group.reverse()

all_cols = FS.get_feature_names(feature_group,train.columns)  
if 'TransactionAmt_fractional' in all_cols:
    all_cols.remove('TransactionAmt_fractional')

print("Feature engineering all done!")   

if visualize_corr_clusters:
    #https://www.kaggle.com/tharug/ieee-fraud-detection
    kw_cbar = {'vmax':1, 'vmin':-1, 'cmap': 'RdYlGn'}
    for group in feature_group:
        visualize_cols = FS.get_feature_names([group],train.columns)
        visualize_cols.insert(0,'isFraud')
        corr = train[visualize_cols].corr('spearman')
        sns.clustermap(corr, **kw_cbar)
        plt.show()
    
train[all_cols]= train[all_cols].apply(pd.to_numeric) 
test[all_cols] = test[all_cols].apply(pd.to_numeric)
train = train.sort_values(granularity_key)
train.reset_index(drop=True,inplace=True) 
    
print("Starting gradient boosting...")
w = FE.get_positive_class_weight(train) 
print("Positive class weight:",w)

preds_proba = np.zeros(len(test),dtype = np.float64)
preds = np.zeros(len(test),dtype = np.float64)


for key in granularity_to_use.keys():
    s = sorted(train[key].unique())
    limit = np.ceil(len(s)*0.7)
    train_size = s[int(limit)]   
    print("train size limit:",train_size)
    train_idx = train[train[key] <= train_size].index.values
    valid_idx = train[train[key] > train_size].index.values
        
    train_x, train_y = train[all_cols].loc[train_idx], train['isFraud'].loc[train_idx]
    valid_x, valid_y = train[all_cols].loc[valid_idx], train['isFraud'].loc[valid_idx]
             
        
    clf = LGBMClassifier(
             nthread=-1,
        objective= "binary",
        metric= "auc",
        boosting= 'gbdt',
        max_depth = 9,
        num_leaves= 56,
        learning_rate= 0.03,
        bagging_freq= 5,
        bagging_fraction= 1.0,
        feature_fraction = 0.8,
        min_child_samples= 10,
        tree_learner= "serial",
        n_estimators=7000,
        seed= 42,
        feature_fraction_seed = 42,
        bagging_seed = 42,
        drop_seed = 42,
        data_random_seed = 42,
        boost_from_average = True,
        scale_pos_weight = w
                    )
        
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
    eval_metric= 'auc', early_stopping_rounds= 200,verbose = 10)
        
    ho_pred = clf.predict(valid_x)
    ho_proba = clf.predict_proba(valid_x)[:,1]
    v_pred = roc_auc_score(valid_y,ho_pred)
    v_proba = roc_auc_score(valid_y,ho_proba)
    print('##################')
    print('Training : Single Model Hold Out Pred AUC=',v_pred)
    print('##################')
    print('Training : Single Model Hold out ProabA AUC=',v_proba)
    ct = pd.crosstab(valid_y,ho_pred,rownames=['Actual'],colnames=['Predicted'],margins=True)
    print(ct)
    print(classification_report(valid_y,ho_pred))
    
    plt.figure(0)
    fpr, tpr, thresh = roc_curve(valid_y, ho_pred)
    plt.plot(fpr,tpr,label="auc="+str(v_pred))
    plt.legend(loc=4)
    
    plt.figure(1)
    fpr, tpr, thresh = roc_curve(valid_y, ho_proba)
    plt.plot(fpr,tpr,label="auc="+str(v_proba))
    plt.legend(loc=4)
    
    preds_proba += clf.predict_proba(test[all_cols], num_iteration=clf.best_iteration_)[:, 1]/4.0
    preds += clf.predict(test[all_cols], num_iteration=clf.best_iteration_)/4.0
del clf
plt.figure(2)
num_vars=50
sub = pd.DataFrame()
sub['TransactionID'] = test["TransactionID"]
sub['isFraud'] = preds
sub.to_csv('outputs/single_model_lgbm_submission_holdout_pred.csv',index=False)
# DISPLAY HISTOGRAM OF PREDICTIONS
b = plt.hist(sub['isFraud'], bins=num_vars)
 
sub = pd.DataFrame()
sub['TransactionID'] = test["TransactionID"]
sub['isFraud'] = preds_proba
sub.to_csv('outputs/single_model_lgbm_submission_holdout_proba.csv',index=False)
# DISPLAY HISTOGRAM OF PREDICTIONS
b = plt.hist(sub['isFraud'], bins=num_vars)

