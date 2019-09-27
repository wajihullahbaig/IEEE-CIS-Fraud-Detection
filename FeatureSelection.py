#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:53:39 2019

@author: wajih
"""
from sklearn.feature_selection import SelectPercentile, f_classif
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm


# https://www.kaggle.com/artgor/eda-and-models

class FeatureSelection():
    def __init__(self):
        pass
    
    def get_card_names(train):
        cards = ['card1','card2','card3','card4','card5','card6']
        found = []
        for card in cards:
            if card in train.columns:
                found.append(card)
        return found
        
    def sort_cat_cols_with_uniqueness(train,cat_cols):
        cat_sorted = pd.DataFrame(columns=['feature','nunique'])
        for col in cat_cols:
            cat_sorted = cat_sorted.append({"feature":col,"nunique":train[col].nunique()},ignore_index=True)
        
        cat_sorted.sort_values(["nunique"],inplace=True,ascending=False)
        return cat_sorted
    
    def quartile_based_cleaning(df,cols,show_plots=False):
        print("performing quartile based cleaning")
        for col in tqdm(cols):
            sorted_features = df.sort_values(col,ascending = True)
            sorted_features = sorted_features[col]
            if show_plots:
                fig, ax = plt.subplots(figsize=(15,10))
                sns.distplot(sorted_features,ax=ax );        
            q1, q3= np.nanpercentile(sorted_features.values,[25,75])
            iqr = q3 - q1
            lower_bound = q1 -(1.5 * iqr) 
            upper_bound = q3 +(1.5 * iqr) 
            if show_plots:
                fig, ax = plt.subplots(figsize=(15,10))    
                sns.distplot(sorted_features[sorted_features <= lower_bound],ax=ax);
                    
                fig, ax = plt.subplots(figsize=(15,10))    
                sns.distplot(sorted_features[sorted_features >= upper_bound],ax=ax);
                   
                fig, ax = plt.subplots(figsize=(15,10))        
                sns.distplot(sorted_features[(sorted_features > lower_bound)&(sorted_features < upper_bound)],ax=ax);        
              
                plt.show()
                print("Limits for " + col)         
                print(lower_bound)
                print(upper_bound)
            
            idx = sorted_features[(sorted_features < lower_bound)&(sorted_features > upper_bound)].index.values
            df[col].loc[idx] = np.nan
            
    def get_high_difference_V_columns(df):
        V_cols =[]
        for id in range(96,138):
            col = 'V'+str(id)
            if col in df.columns:
                V_cols.append('V'+str(id))
                
        for id in range(279,322):
            col = 'V'+str(id)
            if col in df.columns:
                V_cols.append('V'+str(id))
                
        return V_cols
    def get_numerical_columns(df):
        feature_group = ['D','V','C','TransactionAmt','dist']
        num_cols = FeatureSelection.get_feature_names(feature_group,df.columns)   
        for id in range(0,10):
            col = 'id_0'+str(id)
            if col in df.columns:
                num_cols.append('id_0'+str(id))
        if 'id_10' in df.columns:
            num_cols.append('id_10')
        if 'id_11' in df.columns:
            num_cols.append('id_11') 
        return num_cols
    
    def get_numerical_columns_under_id(df):
        num_cols = []
        for id in range(0,10):
            col = 'id_0'+str(id)
            if col in df.columns:
                num_cols.append('id_0'+str(id))
        if 'id_10' in df.columns:
            num_cols.append('id_10')
        if 'id_11' in df.columns:
            num_cols.append('id_11') 
        return num_cols   
    def get_categorical_columns(df):
        # Get categorical columns for label encoding and frequency encoding 
        feature_group = ['ProductCD','card','addr','P_','R_', 'M','Device']
        cat_cols = FeatureSelection.get_feature_names(feature_group,df.columns)   
        for id in range(12,39):
            col = 'id_'+str(id)
            if col in df.columns:
                cat_cols.append('id_'+str(id))
        cols = FeatureSelection.get_feature_names(['id_31_'],df.columns)  
        cat_cols.extend(cols)
        return cat_cols       
    def get_binary_columns(train):
        print("Getting binary columns...")
        binary_columns =  [col for col in train if (len(train[col].value_counts()) > 0) & all(train[col].value_counts().index.isin([0, 1]))]
        print(binary_columns)
        print("Done!")
        return binary_columns

    def drop_features_on_abs_diff(train,test,features):
        print("Drop columns on SOD...")
        sod = pd.DataFrame()
        for f in features:
            v =  np.sum(np.abs(train[f]-test[f]).astype(float))        
            sod = sod.append({'feature': f, 'abs_diff': v}, ignore_index=True)
        
        sod["abs_diff"] = (sod["abs_diff"] - sod["abs_diff"].min()) * (1.0 - 0.0) / (sod["abs_diff"].max() - sod["abs_diff"].min()) + 0.0
        print(sod["abs_diff"])
        sod = sod[sod.abs_diff >= sod["abs_diff"].mean()]
        if "isFraud" in sod.feature:
            sod.remove("isFraud")
        print(len(sod.feature))
        print("Columns to drop:\n",sod.feature)
        
        return sod.feature

    def get_feature_names(feature_group,features):
        all_cols = []
        D_cols = list(range(1,17))
        D_cols = ['D'+str(col) for col in D_cols]
        if isinstance(feature_group, list):
            for g in feature_group:
                cols = [col for col in features if col.startswith(g)]
                if any(elem in D_cols  for elem in cols):
                    cols[:] = [x for x in cols if not x.startswith('Device')]
                all_cols.extend(cols)
        return all_cols
    
    def get_feature_names_on_productcd(train,feature_group,productCD):
        features  = train[train.ProductCD == productCD].columns
        all_cols = []
        D_cols = list(range(1,17))
        D_cols = ['D'+str(col) for col in D_cols]
        if isinstance(feature_group, list):
            for g in feature_group:
                cols = [col for col in features if col.startswith(g)]
                if any(elem in D_cols  for elem in cols):
                    cols[:] = [x for x in cols if not x.startswith('Device')]
                all_cols.extend(cols)
        return all_cols    
    
    def get_drop_columns_manynans_onlyones(train,test,all_cols,threshold):
        print("Nulls/Ones drop columns...")
        one_value_cols = [col for col in all_cols if train[col].nunique() <= 1]
        one_value_cols_test = [col for col in all_cols if test[col].nunique() <= 1]
        one_value_cols == one_value_cols_test
        many_null_cols = [col for col in all_cols if train[col].isnull().sum() / train.shape[0] > threshold]
        many_null_cols_test = [col for col in all_cols if test[col].isnull().sum() / test.shape[0] > threshold]
        big_top_value_cols = [col for col in all_cols if train[col].value_counts(dropna=False, normalize=True).values[0] > threshold]
        big_top_value_cols_test = [col for col in all_cols if test[col].value_counts(dropna=False, normalize=True).values[0] > threshold]
        to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols+ one_value_cols_test))
        if "isFraud" in to_drop:
            to_drop.remove("isFraud")
        print(len(to_drop))
        print('Drop one valued columns from train:\n',one_value_cols)
        print('Drop one valued columns from test:\n',one_value_cols_test)
        print('Drop many null valued columns from train:\n',many_null_cols)
        print('Drop many null valued columns from test:\n',many_null_cols_test)
        print('Drop big top valued columns from train:\n',big_top_value_cols)
        print('Drop big top valued columns from test:\n',big_top_value_cols_test)
        return to_drop
   
    def get_redundant_pairs(df):
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop
    
    def get_top_abs_correlations(df, n=150):
        au_corr = df.corr().abs().unstack()
        labels_to_drop = FeatureSelection.get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:n]
    
        print("Top Absolute Correlations")
        print(FeatureSelection.get_top_abs_correlations(df, 3))
        
    def get_high_correlation_columns(df,threshold,features):
        print("Correlation matrix to drop columns...")
        corr_matrix = df[features].corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))        
        # Find index of feature columns with correlation greater than 0.95
        highly_corr_cols = [column for column in upper.columns if any(upper[column] > threshold)]
        print(len(highly_corr_cols))
        print("Highly correlation columns:\n",highly_corr_cols)       
        return highly_corr_cols
        
    def get_drop_columns_on_correlation(df,corr_threshold,features,show_plots):
        print("creating correlation matrix...")
        features.append('isFraud')
        y = df["isFraud"]
        corr = df[features].corr()
        if show_plots:
            plt.figure(figsize= (30,30))
            sns.heatmap(corr[(corr>=corr_threshold) | (corr <= -corr_threshold)],cmap='coolwarm',vmax=1.0,vmin=-1.0,linewidth=0.1,annot=False,annot_kws={"size":10},square=True)
            plt.show()
        v = corr[(corr>=corr_threshold) | (corr <= -corr_threshold)]
        print('Done!')
        v = v.abs()
        print('Figuring out inter-correlated features...')
        # Select upper triangle of correlation matrix
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
            
        # Find index of feature columns with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] >= corr_threshold)]
        v.drop(to_drop,inplace=True, axis=1)
        
        collected_features = []

        for f in v:
           collected_features.append( list(v[v[f].notnull()].index.values))
        
        
        final_features = []
        for f in collected_features:
            if not f:
                continue;
            if len(f) > 1:
                group = pd.DataFrame(columns=["feature","corr_val"])
                idx = 0
                for sub_f in f:
                    r = stats.pearsonr(df[sub_f],y)[0]
                    print(sub_f , " group vs y correlation" ,r)
                    group.loc[idx]= [sub_f,r]
                    idx = idx + 1
                group.sort_values("corr_val")
                final_features.append(group.iloc[0]["feature"])
            else:
                print(f[0] , " vs y correlation" ,stats.pearsonr(df[f[0]],y)[0])
                final_features.append(f[0])
                       
        print('Done!')        
        droppable_cols = []
        for j in range(len(features)):
            if features[j] not in final_features:
                if 'isFraud' in features[j]:
                    continue
                droppable_cols.append(features[j])
        
        print(len(droppable_cols))
        print("Columns to drop:\n",droppable_cols)       
        return droppable_cols
    
    
    def get_drop_columns_on_percentile_based_feature_selection(train,percent,all_cols):
        print('Percentile based feature selection:',percent)
        X = train.drop(["isFraud",'TransactionID','TransactionDT'],axis = 1)
        y = train["isFraud"]
        
        selector_f = SelectPercentile(f_classif,percentile=percent)
        X_best = selector_f.fit_transform(X,y)
        support = np.asarray(selector_f.get_support())
        #top 20% features
        features = np.asarray(X.columns.values)
        features_with_support = features[support]
        #top 20% f-scores
        fscores = np.asarray(selector_f.scores_)
        fscores_with_support = fscores[support]
        #top 20% p-values
        pvalues = np.asarray(selector_f.pvalues_)
        pvalues_with_support = fscores[support]
        
        top_features = pd.DataFrame({"F-Score":fscores_with_support,"P-Value":pvalues_with_support},index=features_with_support)
        print("Top features best associated features with Y\n Number of features",len(features_with_support))
        print(top_features.sort_values(by='P-Value',ascending = True,inplace=True))
        print('Done!')     
        final_features = top_features.index.values.tolist()
        droppable_cols = []
        for j in range(len(all_cols)):
            if all_cols[j] not in final_features:
                droppable_cols.append(all_cols[j])
        
        print(len(droppable_cols))
        print("Columns to drop:\n",droppable_cols)     
        return droppable_cols
    
    
    def get_drop_columns_on_percentile_then_correlation_feature_selection(train,percent,corr_threshold,show_plots,all_cols):
        print('Percentile then correlation based feature selection:',percent)
        X = train.drop(["isFraud",'TransactionID','TransactionDT'],axis = 1)
        y = train["isFraud"]
        
        selector_f = SelectPercentile(f_classif,percentile=percent)
        X_best = selector_f.fit_transform(X,y)
        support = np.asarray(selector_f.get_support())
        #top 20% features
        features = np.asarray(X.columns.values)
        features_with_support = features[support]
        #top 20% f-scores
        fscores = np.asarray(selector_f.scores_)
        fscores_with_support = fscores[support]
        #top 20% p-values
        pvalues = np.asarray(selector_f.pvalues_)
        pvalues_with_support = fscores[support]
        
        top_features = pd.DataFrame({"F-Score":fscores_with_support,"P-Value":pvalues_with_support},index=features_with_support)
        print("Top features best associated features with Y\n Number of features",len(features_with_support))
        print(top_features.sort_values(by='P-Value',ascending = True,inplace=True))
        print('Done')
        print(top_features)
        # Feature to feature correlation
        print('Caculation feature correlations')
        best_features = train[features_with_support]
        corr = best_features.corr()
        if show_plots:
            plt.figure(figsize= (30,30))
            sns.heatmap(corr[(corr>=corr_threshold) | (corr <= -corr_threshold)],cmap='coolwarm',vmax=1.0,vmin=-1.0,linewidth=0.1,annot=False,annot_kws={"size":10},square=True)
            plt.show()
        v = corr[(corr>=corr_threshold) | (corr <= -corr_threshold)]
        print('Done!')
        v = v.abs()
        print('Figuring out inter-correlated features...')
        # Select upper triangle of correlation matrix
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
            
        # Find index of feature columns with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] >= corr_threshold)]
        v.drop(to_drop,inplace=True, axis=1)
        
        collected_features = []

        for f in v:
           collected_features.append( list(v[v[f].notnull()].index.values))
        
        
        final_features = []
        for f in collected_features:
            if not f:
                continue;
            if len(f) > 1:
                group = pd.DataFrame(columns=["feature","corr_val"])
                idx = 0
                for sub_f in f:
                    r = stats.pearsonr(best_features[sub_f],y)[0]
                    print(sub_f , " group vs y correlation" ,r)
                    group.loc[idx]= [sub_f,r]
                    idx = idx + 1
                group.sort_values("corr_val")
                final_features.append(group.iloc[0]["feature"])
            else:
                print(f[0] , " vs y correlation" ,stats.pearsonr(best_features[f[0]],y)[0])
                final_features.append(f[0])
                       
        print('Done!')        
        droppable_cols = []
        for j in range(len(all_cols)):
            if all_cols[j] not in final_features:
                droppable_cols.append(all_cols[j])
        
        print(len(droppable_cols))
        print("Columns to drop:\n",droppable_cols)       
        return droppable_cols