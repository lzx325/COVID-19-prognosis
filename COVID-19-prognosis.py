#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import sklearn.model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.linear_model
import sklearn.svm
import scipy.stats
import pickle as pkl
from pprint import pprint
import torch
import torch.nn as nn

import model
import train
# %load_ext autoreload
# %autoreload 2


# In[2]:


def get_and_normalize_data():
    with open("data/dataset.pickle",'rb') as f:
        dataset=pkl.load(f)
    data=dataset['data']
    data_min=data.min(axis=0,keepdims=True)
    data_max=data.max(axis=0,keepdims=True)
    data=(data-data_min)/(data_max-data_min)
    dataset['data']=data
    return dataset
def get_training_configs():
    X=np.random.rand(10,100)
    Y=np.random.rand(10,10)
    net=model.FCModel(n_in_features=X.shape[1],n_out_features=Y.shape[1])
    params={
        'n_epoch':100,
        'lr':1e-3,
        'beta':0,
        'batch_size':8,
        
    }
    weight=np.array([1,1,1,1,1,10.,3.,5.,2.,2.])
    
    return net,X,Y,params,weight


# In[3]:


def cross_validation_loop(X,Y,leave_out=5):
    assert 1<=leave_out<=5 and type(leave_out)==int
    n_features=X.shape[1]
    n_targets=Y.shape[1]
    assert len(X)==len(Y)
    averaged_scores={
        'MSE':np.zeros((n_targets,)),
        'feature_grad':np.zeros((n_targets,n_features)),
        'feature_grad_abs':np.zeros((n_targets,n_features))
    }
    cross_validation_times=len(X)//leave_out
    for i in range(cross_validation_times):
        print("Leave %d out [%d/%d]"%(leave_out,i+1,cross_validation_times))
        test_indices=[m for m in range(i*leave_out,(i+1)*leave_out)]
        train_indices=[i for i in range(len(X)) if i not in test_indices]
        X_train=X[train_indices,:]
        Y_train=Y[train_indices,:]
        X_test=X[test_indices,:]
        Y_test=Y[test_indices,:]

        net=model.FCModel(n_in_features=X.shape[1],n_out_features=Y.shape[1])
        params={
            'n_epoch':200,
            'lr':1e-2,
            'beta':0,
            'batch_size':8,
        }
        train.fit(net,X_train,Y_train,params,verbose=False)
        scores=train.score(net,X_test,Y_test)
        for k,v in scores.items():
            averaged_scores[k]+=scores[k]/cross_validation_times
            
    return averaged_scores

    
def permutation_loop(X,Y,permutation_times=10):
    n_features=X.shape[1]
    n_targets=Y.shape[1]
    assert len(X)==len(Y)
    permutation_scores={
        'performance_gain':np.zeros((n_targets,))
    }
    for i in range(permutation_times):
        print("Permutation [%d/%d]"%(i+1,permutation_times))
        X_train,X_test,Y_train,Y_test=sklearn.model_selection.train_test_split(X,Y,test_size=0.25)
    
        net=model.FCModel(n_in_features=X.shape[1],n_out_features=Y.shape[1])
        params={
            'n_epoch':200,
            'lr':1e-2,
            'beta':0,
            'batch_size':8,
        }
        train.fit(net,X_train,Y_train,params,verbose=False)
        scores_on_original=train.score(net,X_test,Y_test)
        
        perm_indices=np.arange(len(Y))
        np.random.shuffle(perm_indices)
        Y_shuffled=Y[perm_indices].copy()
        
        X_train,X_test,Y_shuffled_train,Y_shuffled_test=sklearn.model_selection.train_test_split(X,Y_shuffled,test_size=0.25)
        
        train.fit(net,X_train,Y_shuffled_train,params,verbose=False)
        scores_on_shuffled=train.score(net,X_test,Y_shuffled_test)
        permutation_scores["performance_gain"]+=(scores_on_shuffled["MSE"]-scores_on_original["MSE"])*(1/permutation_times)
    return permutation_scores


# In[4]:


def interface():
    dataset=get_and_normalize_data()
    data=dataset["data"]
    target_col_indices=np.arange(0,26)
    feature_col_indices=np.arange(26,90)
    
    X=data[:,feature_col_indices]
    Y=data[:,target_col_indices]
    
    cross_validation_scores=cross_validation_loop(X,Y,leave_out=5)
    permutation_scores=permutation_loop(X,Y,permutation_times=20)
    
    print("Targets ranked by MSE: ")
    argsort=cross_validation_scores["MSE"].argsort()
    print(target_col_indices[argsort])
    for i in range(len(target_col_indices)):
        print(dataset["feature_names"][target_col_indices[argsort[i]]],end='\t')
    print()
    print("Targets ranked by Permutation Test: ")
    argsort=(-permutation_scores["performance_gain"]).argsort()
    print(target_col_indices[argsort])
    for i in range(len(target_col_indices)):
        print(dataset["feature_names"][target_col_indices[argsort[i]]],end='\t')
    print()
    for i in range(len(target_col_indices)):
        print("Ranking features for target %d"%(target_col_indices[i]))
        argsort=(-cross_validation_scores["feature_grad_abs"][i]).argsort()
        print(feature_col_indices[argsort]) 
if __name__=="__main__":
    interface()

