#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 12:05:16 2019

@author: wajih
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.utils import resample
from tqdm import tqdm
class DataSampling():
    def __init__(self):
        pass
    
    def reverse_column_values(train,cols):
        print('Reversing columns...')
        train_augumented = train.copy()
        for col in tqdm(cols):
            train_augumented[col] = train_augumented[col].values[::-1]
        return train_augumented
    def unique_val_diff(train,test):
        for col in test.columns:
            trv = len(train[col].unique())
            tsv = len(test[col].unique())
            print("col:",col,"train:",trv,"test:",tsv,'diff:',abs(trv-tsv))
    def sample_on_column_unique_value(train,cols):
        for col in cols:
            print('sampling on ',col,' groups...')
            col_groups = pd.DataFrame([[k,v.values]
                            for k,v in train.groupby(col).groups.items()], 
                          columns=[col,'indices'])
            print('Column:',col,'Groups:',len(col_groups))
            selected_indices= []
            for row in col_groups['indices']:
                for idx in range(0,min(len(row),3)):
                    selected_indices.append(row[idx])
            sub_sample = train.loc[selected_indices].copy()
        return sub_sample
        
    def one_go_up_sample_and_select(df,select_perecentage):
        df = DataSampling.one_go_up_sampling(df)
        df = df.sample(frac=select_perecentage,random_state = 99)
        return df
    def per_week_down_sampling(df):
        print("Per week down sampling...")
        l1 = len(df)
        print("Observations before downsampling:",l1)
        down_sampled_df = df.copy()
        down_sampled_df.drop(down_sampled_df.index, inplace=True)
        fraud_ratio = []
        for dow in range(df.dow.min(),df.dow.max()+1):
            fc = len(df[(df.isFraud==1) &(df.dow==dow)])
            nfc = len(df[(df.isFraud==0) &(df.dow==dow)])
            print("Day of week ",dow,"fraud counts:",fc)
            print("Day of week ",dow,"non-fraud counts:",nfc)
            if nfc == 0:
                fr = 0.0
            else:
                fr = 100*fc/nfc
            print("Fraud Percentage:",fr )
            fraud_ratio.append(fr)
            if fc < nfc and fc > 0: # on small sample we need this check
                chunk = df[df.dow==dow]
                df_majority = chunk[chunk.isFraud==0]
                df_minority = chunk[chunk.isFraud==1]
                v = fc/nfc
                df_majority = df_majority.sample(frac = v,random_state = 100)
                down_sampled_df = pd.concat([down_sampled_df,df_majority, df_minority])
        df = down_sampled_df.copy()
        df = df.reset_index(drop=True)
        del down_sampled_df
        l2 = len(df)
        print("Observations after downsampling:",l2)
        print("Reduction precentage:",100.0*(l1-l2)/l1)
        print("Done!") 
        return df,fraud_ratio
    
    def per_hour_down_sampling(df):
        print("Per hour down sampling...")
        l1 = len(df)
        print("Observations before downsampling:",l1)
        down_sampled_df = df.copy()
        down_sampled_df.drop(down_sampled_df.index, inplace=True)
        fraud_ratio = []
        for h in range(df.hour.min(),df.hour.max()+1):
            fc = len(df[(df.isFraud==1) &(df.hour==h)])
            nfc = len(df[(df.isFraud==0) &(df.hour==h)])
            print("hour ",h,"fraud counts:",fc)
            print("hour ",h,"non-fraud counts:",nfc)
            if nfc == 0:
                fr = 0.0
            else:
                fr = 100*fc/nfc
            print("Fraud Percentage:",fr )
            fraud_ratio.append(fr)
            if fc < nfc and fc > 0: # on small sample we need this check
                chunk = df[df.hour==h]
                df_majority = chunk[chunk.isFraud==0]
                df_minority = chunk[chunk.isFraud==1]
                v = fc/nfc
                df_majority = df_majority.sample(frac = v,random_state = 100)
                down_sampled_df = pd.concat([down_sampled_df,df_majority, df_minority])
        df = down_sampled_df.copy()
        df = df.reset_index(drop=True)
        del down_sampled_df
        l2 = len(df)
        print("Observations after downsampling:",l2)
        print("Reduction precentage:",100.0*(l1-l2)/l1)
        print("Done!") 
        return df,fraud_ratio
    
    def per_day_down_sampling(df):
        print("Per day down sampling...")
        l1 = len(df)
        print("Observations before downsampling:",l1)
        down_sampled_df = df.copy()
        down_sampled_df.drop(down_sampled_df.index, inplace=True)
        fraud_ratio = []
        for d in range(int(df.day.min()),int(df.day.max()+1)):
            fc = len(df[(df.isFraud==1) &(df.day==d)])
            nfc = len(df[(df.isFraud==0) &(df.day==d)])
            print("day ",d,"fraud counts:",fc)
            print("day ",d,"non-fraud counts:",nfc)
            if nfc == 0:
                fr = 0.0
            else:
                fr = 100*fc/nfc
            print("Fraud Percentage:",fr )
            fraud_ratio.append(fr)
            if fc < nfc and fc > 0: # on small sample we need this check
                chunk = df[df.day==d]
                df_majority = chunk[chunk.isFraud==0]
                df_minority = chunk[chunk.isFraud==1]
                v = fc/nfc
                df_majority = df_majority.sample(frac = v,random_state = 100)
                down_sampled_df = pd.concat([down_sampled_df,df_majority, df_minority])
        df = down_sampled_df.copy()
        df = df.reset_index(drop=True)
        del down_sampled_df
        l2 = len(df)
        print("Observations after downsampling:",l2)
        print("Reduction precentage:",100.0*(l1-l2)/l1)
        print("Done!") 
        return df,fraud_ratio  
    
    def per_month_down_sampling(df):
        print("Per month down sampling...")
        l1 = len(df)
        print("Observations before downsampling:",l1)
        down_sampled_df = df.copy()
        down_sampled_df.drop(down_sampled_df.index, inplace=True)
        fraud_ratio = []
        for m in range(df.month.min(),df.month.max()+1):
            fc = len(df[(df.isFraud==1) &(df.month==m)])
            nfc = len(df[(df.isFraud==0) &(df.month==m)])
            print("month ",m,"fraud counts:",fc)
            print("month ",m,"non-fraud counts:",nfc)
            if nfc == 0:
                fr = 0.0
            else:
                fr = 100*fc/nfc
            print("Fraud Percentage:",fr )
            fraud_ratio.append(fr)
            if fc < nfc and fc > 0: # on small sample we need this check
                chunk = df[df.month==m]
                df_majority = chunk[chunk.isFraud==0]
                df_minority = chunk[chunk.isFraud==1]
                v = fc/nfc
                df_majority = df_majority.sample(frac = v,random_state = 100)
                down_sampled_df = pd.concat([down_sampled_df,df_majority, df_minority])
        df = down_sampled_df.copy()
        df = df.reset_index(drop=True)
        del down_sampled_df
        l2 = len(df)
        print("Observations after downsampling:",l2)
        print("Reduction precentage:",100.0*(l1-l2)/l1)
        print("Done!") 
        return df,fraud_ratio  
    
    def one_go_down_sampling(df,percentage = None):
        print("One go down sampling...")
        l1 = len(df)
        print("Observations before downsampling:",l1)
        z = len(df[df.isFraud == 0])
        o = len(df[df.isFraud == 1])
        if not percentage:
            v = (o/z)
        else :
            v = percentage
        nf = df[df.isFraud==0].sample(frac = v,random_state = 100,replace=True)
        f = df[df.isFraud == 1]
        df = pd.concat([f, nf])
        df = df.reset_index(drop=True)
        del f
        del nf
        l2 = len(df)
        print("Observations after downsampling:",l2)
        print("Reduction precentage:",100.0*(l1-l2)/l1)
        print("Done!") 
        return df
    
    def per_hour_up_sampling(df):
        print("Per hour up sampling...")
        print("Fraud counts",len(df[df.isFraud == 1]))
        print("Non Fraud counts",len(df[df.isFraud == 0]))
        for h in range(df.hour.min(),df.hour.max()+1):
            fc = len(df[(df.isFraud==1) &(df.hour==h)])
            nfc = len(df[(df.isFraud==0) &(df.hour==h)])
            print("hour ",h,"fraud counts:",fc)
            print("hour ",h,"non-fraud counts:",nfc)
            if fc < nfc and fc > 0: # on small sample we need this check    
                chunk = df[df.hour==h]
                df_majority = chunk[chunk.isFraud==0]
                df_minority = chunk[chunk.isFraud==1]
                maj_len = len(df_majority) 
                min_len = len(df_minority) 
                # Upsample minority class
                df_minority_upsampled = resample(df_minority, 
                                             replace=True,     # sample with replacement
                                             n_samples=maj_len,    # to match majority class
                                             random_state=123) # reproducible results
             
            # Combine majority class with upsampled minority class
                df = pd.concat([df, df_minority_upsampled])
                del df_majority
                del df_minority
                del df_minority_upsampled
        print("Done!")        
        return df
    
    def per_day_up_sampling(df,precentage=None):
        print("Per day up sampling...")
        l1 = len(df)
        print("Observations before upsampling:",l1)
        print("Fraud counts",len(df[df.isFraud == 1]))
        print("Non Fraud counts",len(df[df.isFraud == 0]))
        fraud_ratio = []
        for d in range(df.day.min(),df.day.max()+1):
            fc = len(df[(df.isFraud==1) &(df.day==d)])
            nfc = len(df[(df.isFraud==0) &(df.day==d)])
            print("day ",d,"fraud counts:",fc)
            print("day ",d,"non-fraud counts:",nfc)
            if nfc == 0:
                fr = 0.0
            else:
                fr = 100*fc/nfc
            print("Fraud Percentage:",fr )
            fraud_ratio.append(fr)
            if fc < nfc and fc > 0: # on small sample we need this check
                chunk = df[df.day==d]
                df_majority = chunk[chunk.isFraud==0]
                df_minority = chunk[chunk.isFraud==1]
                maj_len = len(df_majority) 
                min_len = len(df_minority) 
                # Upsample minority class
                if precentage is None:
                    df_minority_upsampled = resample(df_minority, 
                                             replace=True,     # sample with replacement
                                             n_samples=maj_len,    # to match majority class
                                             random_state=123) # reproducible results
                else:
                     df_minority_upsampled = resample(df_minority, 
                                             replace=True,     # sample with replacement
                                             n_samples=int(np.round(min_len+min_len*precentage/100.0)),    # to match percentage
                                             random_state=123) # reproducible results
            # Combine majority class with upsampled minority class
                df = pd.concat([df, df_minority_upsampled])
                del df_majority
                del df_minority
                del df_minority_upsampled
        l2 = len(df)
        print("Observations after upsampling:",l2)
        print("Addition precentage:",100.0*(l2-l1)/l1)
        print("Done!") 
        return df,fraud_ratio

    
    def one_go_up_sampling(df):
        print("One go up sampling...")
        l1 = len(df)
        print("Observations before upsampling:",l1)
        # Separate majority and minority classes
        df_majority = df[df.isFraud==0]
        df_minority = df[df.isFraud==1]
        maj_len = len(df_majority) 
        min_len = len(df_minority) 
        # Upsample minority class
        df_minority_upsampled = resample(df_minority, 
                                         replace=True,     # sample with replacement
                                         n_samples=maj_len,    # to match majority class
                                         random_state=123) # reproducible results
         
        # Combine majority class with upsampled minority class
        df = pd.concat([df_majority, df_minority_upsampled])
        del df_majority
        del df_minority
        del df_minority_upsampled
        df = df.sample(frac=1,random_state=100)
        l2 = len(df)
        print("Observations after upsampling:",l2)
        print("Addition precentage:",100.0*(l2-l1)/l2)
        print("Done!") 
        return df
    