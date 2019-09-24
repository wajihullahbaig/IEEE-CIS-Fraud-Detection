#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:27:38 2019

@author: wajih
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt

submission = pd.read_csv('test_transaction.csv')
label_file = 'single_model_lgbm_submission_pl_pred_s0_d_ms.csv'
training_files = [
                  'single_model_lgbm_submission_pl_proba_s0_d_ms.csv',
                  'single_model_lgbm_submission_pl_proba_s0_m_ms.csv',
                  'single_model_lgbm_submission_pl_proba_s0_h_ms.csv',
                  'single_model_lgbm_submission_pl_proba_s0_dow_ms.csv',

                  ]

training_data = pd.DataFrame()
file_count = 0
cols = []
for file in training_files:
    col =  "td"+str(file_count)   
    cols.append(col)
    d = pd.read_csv('outputs/'+file,usecols=['isFraud'])
    training_data[col] = d.isFraud
    file_count +=1

training_data['isFraud'] = pd.read_csv('outputs/'+label_file,usecols=['isFraud'])
    
lr= LogisticRegression(random_state=99,solver='saga')
lr.fit(training_data[cols],training_data['isFraud'])
pred= lr.predict(training_data[cols])
pred_proba = lr.predict_proba(training_data[cols])[:,1]


ct = pd.crosstab(training_data['isFraud'],pred,rownames=['Actual'],colnames=['Predicted'],margins=True)
print(ct)
print(classification_report(training_data['isFraud'],pred))
print("coefficeints:",lr.coef_)

print('Ensemble Pred AUC = ',roc_auc_score(training_data['isFraud'],pred))
plt.figure(0)
fpr, tpr, thresh = roc_curve(training_data['isFraud'],pred)
plt.plot(fpr,tpr,label="auc="+str(roc_auc_score(training_data['isFraud'],pred)))
plt.legend(loc=4)
    
print('Ensemble Proba AUC = ',roc_auc_score(training_data['isFraud'],pred_proba))
plt.figure(1)
fpr, tpr, thresh = roc_curve(training_data['isFraud'],pred_proba)
plt.plot(fpr,tpr,label="auc="+str(roc_auc_score(training_data['isFraud'],pred_proba)))
plt.legend(loc=4)

submission['isFraud'] = pred_proba


plt.figure(2)
num_vars=50
sub = pd.DataFrame()
sub['TransactionID'] = submission["TransactionID"]
sub['isFraud'] = pred
sub.to_csv('outputs/single_model_lgbm_submission_holdout_pred_ensemble.csv',index=False)
# DISPLAY HISTOGRAM OF PREDICTIONS
b = plt.hist(sub['isFraud'], bins=num_vars)
 
sub = pd.DataFrame()
sub['TransactionID'] = submission["TransactionID"]
sub['isFraud'] = pred_proba
sub.to_csv('outputs/single_model_lgbm_submission_holdout_proba_ensemble.csv',index=False)
# DISPLAY HISTOGRAM OF PREDICTIONS
b = plt.hist(sub['isFraud'], bins=num_vars)