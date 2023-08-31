# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:10:32 2023

@author: Ivan
"""

import numpy as np
import pandas as pd
np.seterr(divide = 'ignore') 
rng = np.random.RandomState(0)
from sklearn.cross_decomposition import PLSRegression


#%% SAMPLERS and CROSS VALIDATORS
class ActivitySampler():
    def __init__(self,sampler_df,descriptor_df,y_col,n,rng=None,):
        self.sampler = sampler_df.sort_values(y_col).copy()
        self.X = descriptor_df
        self.rng = rng
        self.groups = list()
        self.n_groups = n
        self.y_col = y_col
        
        sampler = self.sampler
        inactive = sampler.loc[sampler[y_col]==0]
        active = sampler.loc[sampler[y_col]>0]
        split_dataframes = np.array_split(active, n)
        average_len = len(active)//n
        len_inactive = len(inactive)

        if len_inactive>average_len:
            inactive = inactive.sample(average_len,random_state=self.rng)

        split_dataframes.insert(0,inactive)
        self.groups = split_dataframes
        
    def sample(self,rng=None):
        if rng is None:
            rng = self.rng
        if rng=="shuffle":
            rng = None
        test = pd.concat([x.sample(1,random_state=rng) for x in self.groups])
        train = pd.concat(self.groups)
        train = train.loc[~train.index.isin(test.index)]
        #train = self.sampler.loc[~self.sampler.index.isin(test.index)]
        X_test = self.X.loc[test.index]
        X_train = self.X.loc[train.index]
        y_test = test[self.y_col]
        y_train = train[self.y_col]
        
        return X_train,X_test,y_train,y_test
        #return test,train,X_test,X_train
    
class CrossValidator():
    def __init__(self,molecule_df,descriptor_df,X_train,y_train,groups=4,rng=rng,n_scores=5,model=PLSRegression()):
        self.X_train = X_train
        self.y_train = y_train
        self.n = n_scores
        self.model = model
        self.models = list()
        self.scores = list()
        #find only indices in X train and y train
        training_samples = self.X_train.index.tolist()
        self.predictor_df = molecule_df.loc[training_samples]
        self.X_df = descriptor_df.loc[training_samples]
        #print(self.X_df.shape,self.predictor_df.shape)
        self.subsampler = ActivitySampler(self.predictor_df,self.X_df,y_train.name,groups,)
     
    def score(self):
        self.scores 
        for i in range(self.n):
            subX_train,subX_test,suby_train,suby_test = self.subsampler.sample(rng="shuffle")
            model = self.model
            #model = PLSRegression()
            model.fit(subX_train,suby_train)
            self.models.append(model)
            self.scores.append(model.score(subX_test,suby_test))
        #todo - either modify this to exclude negative values
        #or do something else with it
        scores = [max(0,x) for x in self.scores]
        return sum(scores)/len(scores)    
    
    def fit_this_validator(self):
        self.model.fit(self.X_train,self.y_train)
        
