# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:15:00 2019

@author: TNDUser
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pandas as pd
import datetime
from sklearn.utils import resample
from collections import Counter
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.decomposition import PCA
import gc

# References
# https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
# https://www.kaggle.com/bogorodvo/lightgbm-baseline-model-using-sparse-matrix
# https://www.kaggle.com/kevinbonnes/transactiondt-starting-at-2017-12-01
# https://github.com/imor-de/microsoft_malware_prediction_kaggle_2nd/blob/master/code/1_Data_Cleaning_train_set.ipynb
# https://github.com/Johnnyd113/Microsoft-Malware-Prediction/blob/master/prepare_data.py
# https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again?scriptVersionId=18889353 
# https://www.kaggle.com/dustinthewind/making-sense-of-mean-encoding
# https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/53696
# https://www.kaggle.com/yasagure/places-after-the-decimal-point-tell-us-a-lot
# http://wkirgsn.github.io/2018/02/25/pandas-aggregate
# https://kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
# http://ianlondon.github.io/blog/encoding-cylcical-features-24-hour-time/
# https://kaggle.com/zikazika/could-we-predict-fraud-solely-based-on-nan
# https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575#latest-625440
class FeatureEngineering():
    def __init__(self):
        pass    
        
    def PCA_reduction(df,cols,feature_group,components = 7):
        # use only one feature group
        print('PCA reduction...')
        pca_reducer = PCA(n_components=components,random_state=99)
        reduced = pca_reducer.fit_transform(df[cols])
        count_cols = list(range(0,components))
        reduced_cols = [feature_group[0]+'_'+str(col) for col in count_cols]
        pca_df = pd.DataFrame(data = reduced , columns = reduced_cols)
        print('Done!')
        return pca_df  
        
    def scale_data(X, scaler=None):
        print('Performing data scaling...')
        if not scaler:
            scaler = MaxAbsScaler()
            scaler.fit(X)
        X = scaler.transform(X)
        return X, scaler
        
    def expand_id31_and_DeviceInfo(train,test):
        # We splits the categorical data 
        print('Expanding categorical columns...')
        expansion = train['id_31'].str.split(' ',expand=True)
        train['id_31_browser_name'] = expansion[0].copy()
        train['id_31_browser_name_or_version'] = expansion[1].copy()
        train['id_31_browser_version'] = expansion[2].copy()
        expansion = train['DeviceInfo'].str.split(' ',expand=True)
        train['DeviceInfo_name'] = expansion[0].copy()
        train['DeviceInfo_version'] = expansion[1].copy()
        train['DeviceInfo_Build'] = expansion[2].copy()
        
        expansion = test['id_31'].str.split(' ',expand=True)
        test['id_31_browser_name'] = expansion[0].copy()
        test['id_31_browser_name_or_version'] = expansion[1].copy()
        test['id_31_browser_version'] = expansion[2].copy()
        expansion = test['DeviceInfo'].str.split(' ',expand=True)
        test['DeviceInfo_name'] = expansion[0].copy()
        test['DeviceInfo_version'] = expansion[1].copy()
        test['DeviceInfo_Build'] = expansion[2].copy()
        
        del expansion

    def normalize_central_mean_std(train,test,cols):
        print('Normalize central mean...')
        for col in tqdm(cols):
            train[col] = ( train[col]-train[col].mean() ) / train[col].std() 
            test[col] = ( test[col]-test[col].mean() ) / test[col].std()            
    def normalize_in_time(train,test,cols,time_feature):
        print('Normalization in time...')
        for col in tqdm(cols):         
            temp_dict = train.groupby([col])[time_feature].agg(['mean']).reset_index().rename(
                                                                columns={'mean': col+'_target_'+'mean'})
            temp_dict.index = temp_dict[col].values
            temp_dict = temp_dict[col+'_target_'+'mean'].to_dict()    
            train_mean= train[col].map(temp_dict)    
            train[col] = train[col] - train_mean
    
            temp_dict = test.groupby([col])[time_feature].agg(['mean']).reset_index().rename(
                                                                columns={'mean': col+'_target_'+'mean'})
            temp_dict.index = temp_dict[col].values
            temp_dict = temp_dict[col+'_target_'+'mean'].to_dict()    
            test_mean= test[col].map(temp_dict)    
            test[col] = test[col] - test_mean
    
        
    def create_nan_indicator(train,test,cols):
        print('Creating nan indicator features...')
        for c in tqdm(cols):
            train[c+"_nan_indicator"] = np.where(train[c].isna(),0,1)
            test[c+"_nan_indicator"] = np.where(test[c].isna(),0,1)
        
        
    def encode_cyclical_features(train,test,cols_maxvals):
        print('Encoding cyclical features...')
        cyclical_features = []
        for c in tqdm(list(cols_maxvals.keys())):
            train[c+'_sin'] = np.sin(2*np.pi*train[c]/cols_maxvals[c])
            test[c+'_sin'] = np.sin(2*np.pi*test[c]/cols_maxvals[c])
            train[c+'_cos'] = np.cos(2*np.pi*train[c]/cols_maxvals[c])
            test[c+'_cos'] = np.cos(2*np.pi*test[c]/cols_maxvals[c])
            cyclical_features.append(c+'_sin')
            cyclical_features.append(c+'_cos')
        return cyclical_features    
    def row_wise_normalize(train,test,cols):
        print('Row-wise magnitude normalization...')
        train[cols]  = train[cols].div(1000.0+np.linalg.norm(train[cols].values,axis=1),axis=0)
        test[cols]  = test[cols].div(1000.0+np.linalg.norm(test[cols].values,axis=1),axis=0)
        print('Done!')
        
    def make_polynomial_features(train,test,cols,drop_originals):
        print('Creating polynomial features...')
        p = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False).fit(train[cols])
        features = pd.DataFrame(p.transform(train[cols]),columns=p.get_feature_names(cols))
        if drop_originals:
            features.drop(cols,inplace=True,axis=1)
        train = pd.concat([train,features],axis=1)
        
        p = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False).fit(test[cols])
        features = pd.DataFrame(p.transform(test[cols]),columns=p.get_feature_names(cols))
        if drop_originals:
            features.drop(cols,inplace=True,axis=1)
        test = pd.concat([test,features],axis=1)
        
        return train,test
        
    def split_decimal_and_fractional_part(train,test,cols):
        print("Getting fractional parts...")
        for col in tqdm(cols):
            if train is not None:
                train[col+"_fractional"] =np.modf(train[col])[0]
            if test is not None:
                test[col+"_fractional"] =np.modf(test[col])[0]
        
    def create_value_sum_percentage(train,test,cols,val):
        print("Creating value sum percentage feature...")
        for c in tqdm(cols):
            train[c+"_sum"] = train[train[c] == val][c].astype(np.int32).sum()/len(train)
            test[c+"_sum"] = test[test[c] == val][c].astype(np.int32).sum()/len(test)
        print('Done!')    
    def get_positive_class_weight (train):
        P = len(train[train.isFraud == 1])
        T = len(train)
        return T/P - 1.0
    def add_noise(series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    def encode_target_exponential_smooth(trn_series=None, 
                      tst_series=None, 
                      target=None, 
                      min_samples_leaf=1, 
                      smoothing=1,
                      noise_level=0):
        """
        Smoothing is computed like in the following paper by Daniele Micci-Barreca
        https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
        trn_series : training categorical feature as a pd.Series
        tst_series : test categorical feature as a pd.Series
        target : target data as a pd.Series
        min_samples_leaf (int) : minimum samples to take category average into account
        smoothing (int) : smoothing effect to balance categorical average vs prior  
        """ 
        assert len(trn_series) == len(target)
        assert trn_series.name == tst_series.name
        print("Target Exponential Smooth Encoding for:",trn_series.name)
        temp = pd.concat([trn_series, target], axis=1)
        # Compute target mean 
        averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
        # Apply average function to all target data
        prior = target.mean()
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(["mean", "count"], axis=1, inplace=True)
        # Apply averages to trn and tst series
        ft_trn_series = pd.merge(
            trn_series.to_frame(trn_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=trn_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index 
        ft_tst_series = pd.merge(
            tst_series.to_frame(tst_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=tst_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_tst_series.index = tst_series.index
        print("Done!")
        return FeatureEngineering.add_noise(ft_trn_series, noise_level), FeatureEngineering.add_noise(ft_tst_series, noise_level)

    def encode_target_smooth(train, target, categ_variables, smooth):
        """    
        Apply target encoding with smoothing.
        
        Parameters
        ----------
        data: pd.DataFrame
        target: str, dependent variable
        categ_variables: list of str, variables to encode
        smooth: int, number of observations to weigh global average with
        
        Returns
        --------
        encoded_dataset: pd.DataFrame
        code_map: dict, mapping to be used on validation/test datasets 
        defaul_map: dict, mapping to replace previously unseen values with
        """
        print("Target Smooth Encoding...")
        code_map = dict()    # stores mapping between original and encoded values
        default_map = dict() # stores global average of each variable
        
        for v in tqdm(categ_variables):
            prior = train[target].mean()
            n = train.groupby(v).size()
            mu = train.groupby(v)[target].mean()
            mu_smoothed = (n * mu + smooth * prior) / (n + smooth)
            
            train.loc[:, v] = train[v].map(mu_smoothed)        
            code_map[v] = mu_smoothed
            default_map[v] = prior   
            
        return train, code_map, default_map

    def make_transactionamt_features(test,train,amount_col,unique_id_col):
    ########################## TransactionAmt

        # Let's add some kind of client uID based on cardID ad addr columns
        # The value will be very specific for each client so we need to remove it
        # from final feature. But we can use it for aggregations.
        print(amount_col," feature engineering...")
        train[unique_id_col] = train['card1'].astype(str)+train['card2'].astype(str)+train['card3'].astype(str)+train['card4'].astype(str)
        test[unique_id_col] = test['card1'].astype(str)+test['card2'].astype(str)+test['card3'].astype(str)+test['card4'].astype(str)
            
        train[amount_col+'_check'] = np.where(train[amount_col].isin(test[amount_col]), 1, 0)
        test[amount_col+'_check']  = np.where(test[amount_col].isin(train[amount_col]), 1, 0)
        train[amount_col+'_check'] = pd.to_numeric(train[amount_col+'_check'])
        test[amount_col+'_check'] = pd.to_numeric(test[amount_col+'_check'])
        # For our model current TransactionAmt is a noise (even when features importances are telling contrariwise)
        # There are many unique values and model doesn't generalize well
        # Lets do some aggregations

        i_cols = ['card1','card2','card3','card5',unique_id_col]

        for col in tqdm(i_cols):
            for agg_type in tqdm(['var']):
                new_col_name = col+'_'+amount_col+'_'+agg_type
                temp_df = pd.concat([train[[col, amount_col]], test[[col,amount_col]]])
                temp_df = temp_df.groupby([col])[amount_col].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})
                
                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   
            
                train[new_col_name] = train[col].map(temp_df)
                train[new_col_name] = pd.to_numeric(train[new_col_name])
                test[new_col_name]  = test[col].map(temp_df)
                test[new_col_name] = pd.to_numeric(test[new_col_name])
        
        # Small "hack" to transform distribution 
        # (doesn't affect auc much, but I like it more)
        # please see how distribution transformation can boost your score 
        # (not our case but related)
        # https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
        train[amount_col] = np.log1p(train[amount_col])
        test[amount_col] = np.log1p(test[amount_col])
        
#    def make_transactionamt_features(test,train):
#    ########################## TransactionAmt
#
#        # Let's add some kind of client uID based on cardID ad addr columns
#        # The value will be very specific for each client so we need to remove it
#        # from final feature. But we can use it for aggregations.
#        print("TransactionAmt feature engineering...")
#        train['uid'] = train['card1'].astype(str)+train['card2'].astype(str)+train['card3'].astype(str)+train['card4'].astype(str)
#        test['uid'] = test['card1'].astype(str)+test['card2'].astype(str)+test['card3'].astype(str)+test['card4'].astype(str)
#        
##        train['uid2'] = train['uid']+train['addr1'].astype(str)+train['addr2'].astype(str)
##        test['uid2'] = test['uid']+test['addr1'].astype(str)+test['addr2'].astype(str)
#        
#        # Check if Transaction Amount is common or not (we can use freq encoding here)
#        # In our dialog with model we are telling to trust or not to these values  
#        valid_card = train['TransactionAmt'].value_counts()
#        valid_card = valid_card[valid_card>10]
#        valid_card = list(valid_card.index)
#            
#        train['TransactionAmt_check'] = np.where(train['TransactionAmt'].isin(test['TransactionAmt']), 1, 0)
#        test['TransactionAmt_check']  = np.where(test['TransactionAmt'].isin(train['TransactionAmt']), 1, 0)
#        train['TransactionAmt_check'] = pd.to_numeric(train['TransactionAmt_check'])
#        test['TransactionAmt_check'] = pd.to_numeric(test['TransactionAmt_check'])
#        # For our model current TransactionAmt is a noise (even when features importances are telling contrariwise)
#        # There are many unique values and model doesn't generalize well
#        # Lets do some aggregations
##        i_cols = ['card1','card2','card3','card5','uid','uid2']
#                
#        i_cols = ['card1','card2','card3','card5','uid']
#
#        for col in tqdm(i_cols):
#            for agg_type in tqdm(['var']):
#                new_col_name = col+'_TransactionAmt_'+agg_type
#                temp_df = pd.concat([train[[col, 'TransactionAmt']], test[[col,'TransactionAmt']]])
#                temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
#                                                        columns={agg_type: new_col_name})
#                
#                temp_df.index = list(temp_df[col])
#                temp_df = temp_df[new_col_name].to_dict()   
#            
#                train[new_col_name] = train[col].map(temp_df)
#                train[new_col_name] = pd.to_numeric(train[new_col_name])
#                test[new_col_name]  = test[col].map(temp_df)
#                test[new_col_name] = pd.to_numeric(test[new_col_name])
#        
#        # Small "hack" to transform distribution 
#        # (doesn't affect auc much, but I like it more)
#        # please see how distribution transformation can boost your score 
#        # (not our case but related)
#        # https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
#        train['TransactionAmt'] = np.log1p(train['TransactionAmt'])
#        test['TransactionAmt'] = np.log1p(test['TransactionAmt'])
        
    def geo_anomaly_search(train,test):
        # Let's look on bank addres and client addres matching
        # card3/card5 bank country and name?
        # Addr2 -> Clients geo position (country)
        # Most common entries -> normal transactions
        # Less common etries -> some anonaly
        print('Geo anomaly search...')
        train['bank_type'] = train['card3'].astype(str)+'_'+train['card5'].astype(str)
        test['bank_type']  = test['card3'].astype(str)+'_'+test['card5'].astype(str)
        
        train['address_match'] = train['bank_type'].astype(str)+'_'+train['addr2'].astype(str)
        test['address_match']  = test['bank_type'].astype(str)+'_'+test['addr2'].astype(str)
        
        for col in tqdm(['address_match','bank_type']):
            temp_df = pd.concat([train[[col]], test[[col]]])
            temp_df[col] = np.where(temp_df[col].str.contains('nan'), np.nan, temp_df[col])
            temp_df = temp_df.dropna()
            fq_encode = temp_df[col].value_counts().to_dict()   
            train[col] = train[col].map(fq_encode)
            test[col]  = test[col].map(fq_encode)
        
        train['address_match'] = train['address_match']/train['bank_type'] 
        test['address_match']  = test['address_match']/test['bank_type']
        print('Done!')  
        
    def noise_reset(train,test,features,min_entries):
        print("Reducing noise...")
        for col in tqdm(features):
            valid_entry = train[col].value_counts()
            valid_entry = valid_entry[valid_entry>min_entries]
            valid_entry = list(valid_entry.index)
                
            train[col] = np.where(train[col].isin(valid_entry), train[col], np.nan)
            test[col]  = np.where(test[col].isin(valid_entry), test[col], np.nan)      
        
    def aggregation_on_train_map_on_test(train,test,features, operation,target):
        print("Aggregating...")
        for col in tqdm(features):
            temp_dict = train.groupby([col])[target].agg([operation]).reset_index().rename(
                                                                columns={operation: col+'_target_'+operation})
            temp_dict.index = temp_dict[col].values
            temp_dict = temp_dict[col+'_target_'+operation].to_dict()
        
            train[col+'_target_'+operation] = train[col].map(temp_dict)
            test[col+'_target_'+operation]  = test[col].map(temp_dict)
            
    def aggregation(train,test,features, operation,target):
        print("Aggregating...")
        for col in tqdm(features):
            temp_dict = train.groupby([col])[target].agg([operation]).reset_index().rename(
                                                                columns={operation: col+'_target_'+operation})
            temp_dict.index = temp_dict[col].values
            temp_dict = temp_dict[col+'_target_'+operation].to_dict()
            train[col+'_target_'+operation] = train[col].map(temp_dict)
            
        for col in tqdm(features):
            temp_dict = test.groupby([col])[target].agg([operation]).reset_index().rename(
                                                                columns={operation: col+'_target_'+operation})
            temp_dict.index = temp_dict[col].values
            temp_dict = temp_dict[col+'_target_'+operation].to_dict()         
            test[col+'_target_'+operation]  = test[col].map(temp_dict)            
    
    def create_null_sum(train,test,features):
        print('Creating null sum features...')
        for f in tqdm(features):
            train[f+"_null_sum"] = train[f].isna().sum() 
            test[f+"_null_sum"] = test[f].isna().sum() 
    
    def replace_nan_mode(df,binary_features):
        print("Replacing NaN with mode on binary features...")
        for feature in tqdm(binary_features):
        # replace NaN-values with the most fequent value and add features to corespondig list for redoing same on testdata
            df[feature] = df[feature].fillna(df[feature].mode()[0])

    
    def drop_rows_on_nan(df,threshold):
        print('Dropping rows on nan',threshold)
        print('Row counts before drop',len(df))
        df = df.dropna(thresh = threshold)
        print('Row counts after drop',len(df))
        print('Done')
        return df
    def reduce_cardinality(train,test):
        print('Reducing cardinality...')
        col_count = 0
        tc = len(train.columns)-2
        for usecol in tqdm(train.columns.tolist()[1:]):
            print('Cardinality reduction for column:',usecol, 'column#',col_count,'out of :',tc)
            col_count +=1
            if 'isFraud' in usecol or 'TransactionDT' in usecol or 'TransactionAmt' in usecol:
                continue
            train[usecol] = train[usecol].astype('str')
            test[usecol] = test[usecol].astype('str')
            
            #Fit LabelEncoder
            le = LabelEncoder().fit(
                    np.unique(train[usecol].unique().tolist()+
                              test[usecol].unique().tolist()))
        
            #At the end 0 will be used for dropped values
            train[usecol] = le.transform(train[usecol])+1
            test[usecol]  = le.transform(test[usecol])+1
        
            agg_tr = (train
                      .groupby([usecol])
                      .aggregate({'TransactionID':'count'})
                      .reset_index()
                      .rename({'TransactionID':'Train'}, axis=1))
            agg_te = (test
                      .groupby([usecol])
                      .aggregate({'TransactionID':'count'})
                      .reset_index()
                      .rename({'TransactionID':'Test'}, axis=1))
        
            agg = pd.merge(agg_tr, agg_te, on=usecol, how='outer').replace(np.nan, 0)
            #Select values with more than 1000 observations
            agg = agg[(agg['Train'] > 100)].reset_index(drop=True)
            agg['Total'] = agg['Train'] + agg['Test']
            #Drop unbalanced values
            agg = agg[(agg['Train'] / agg['Total'] > 0.1) & (agg['Train'] / agg['Total'] < 0.4)]
            agg[usecol+'Copy'] = agg[usecol]
        
            train[usecol] = (pd.merge(train[[usecol]], 
                                      agg[[usecol, usecol+'Copy']], 
                                      on=usecol, how='left')[usecol+'Copy']
                             .replace(np.nan, 0).astype('int').astype('category'))
        
            test[usecol]  = (pd.merge(test[[usecol]], 
                                      agg[[usecol, usecol+'Copy']], 
                                      on=usecol, how='left')[usecol+'Copy']
                             .replace(np.nan, 0).astype('int').astype('category'))
        
            del le, agg_tr, agg_te, agg, usecol
            gc.collect()              
    
    def reduce_mem_usage(df):
        """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.        
        """
        print("Reducing Memory usage...")
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        
        for col in tqdm(df.columns):
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')
    
        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        return df
    
    def create_log_feature(df,cols,displacement):
        print("Creating log features...")
        for col in tqdm(cols):
            df[col] = np.log(displacement+df[col])
        
    
    def encode_label(train,test,cat_cols):
        for col in tqdm(cat_cols):
            if col in train.columns:            
                le = LabelEncoder()
                le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
                train[col] = le.transform(list(train[col].astype(str).values))
                test[col] = le.transform(list(test[col].astype(str).values))       
    # FREQUENCY ENCODE
    def encode_frequency(train,test,cols):
        print('Frequency Encoding...')   
        encoded_cols = []
        for col in tqdm(cols):
            temp_df = pd.concat([train[[col]], test[[col]]])
            fq_encode = temp_df[col].value_counts().to_dict()   
            train[col+'_fq_enc'] = train[col].map(fq_encode)
            test[col+'_fq_enc']  = test[col].map(fq_encode)
            encoded_cols.append(col+'_fq_enc')
        return encoded_cols   
    
    def encode_statistics(df,feature_group):
        print("Statistical Encoding...")
        num_vars = len(feature_group)
        for j in tqdm(range(num_vars)):
            cols = [col for col in df.columns if col.startswith(feature_group[j])]
            if cols is None:
                pass
            if "D1" in cols and "DeviceInfo"  in cols:
                cols.remove("DeviceInfo")
            if "D1" in cols and "DeviceType"  in cols:
                cols.remove("DeviceType")
            
            df[feature_group[j]+"_sum"] =df[cols].sum(axis=1)
                       
    
    def make_datetime_feature(df,start_date):
        df["TransactionDT"] = df["TransactionDT"].apply(lambda x:(start_date+datetime.timedelta(seconds =x)))
    
    def make_ymdhd_feature(df):
        print("Creating Date and Time features")
        # create date column
        START_DATE = '2017-12-01'
        startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
        df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
        
        df['year'] = df['TransactionDT'].dt.year
        df['month'] = df['TransactionDT'].dt.month
        df['dow'] = df['TransactionDT'].dt.dayofweek
        df['hour'] = df['TransactionDT'].dt.hour
        df['day'] = df['TransactionDT'].dt.day    
        print("Done!")
    
    def make_day_feature(df, offset=0, tname='TransactionDT'):
        """
        Creates a day of the week feature, encoded as 0-6. 
        
        Parameters:
        -----------
        df : pd.DataFrame
            df to manipulate.
        offset : float (default=0)
            offset (in days) to shift the start/end of a day.
        tname : str
            Name of the time column in df.
        """
        # found a good offset is 0.58
        days = df[tname] / (3600*24)        
        encoded_days = np.floor(days-1+offset) % 7
        return encoded_days    
        
    def make_hour_feature(df, tname='TransactionDT'):
        """
        Creates an hour of the day feature, encoded as 0-23. 
        
        Parameters:
        -----------
        df : pd.DataFrame
            df to manipulate.
        tname : str
            Name of the time column in df.
        """
        hours = df[tname] / (3600)        
        encoded_hours = np.floor(hours) % 24
        return encoded_hours
 
    
    def nan2mean(df,cols):
        for c in tqdm(cols):
            df[c] = df[c].fillna(df[c].mean())
            
        return df