# -*- coding: utf-8 -*-
"""
Created on oct 20 23:15:24 2015

@author: marios

Script that makes Xgboost scikit-like.

The initial version of the script came from Guido Tapia (or such is his kaggle name!). I have modified it quite a bit though.

the github from where this was retrieved was : https://github.com/gatapia/py_ml_utils

He has done excellent job in making many commonly used algorithms scikit-like 

"""


from sklearn.base import BaseEstimator, ClassifierMixin
import sys
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix




class XGBoostClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, silent=True,
      use_buffer=True, num_round=10,num_parallel_tree=1, ntree_limit=0,
      nthread=None, booster='gbtree', 
      eta=0.3, gamma=0.01, 
      max_depth=6, min_child_weight=1, subsample=1, 
      colsample_bytree=1,      
      l=0, alpha=0, lambda_bias=0, objective='reg:linear',
      eval_metric='logloss', seed=0, num_class=None,
      max_delta_step=0,classes_=None ,
      colsample_bylevel=1.0 , sketch_eps=0.1 , sketch_ratio=2.0 ,
      opt_dense_col=1, size_leaf_vector=0.0, min_split_loss=0.0,
      cache_opt=1, default_direction =0 , k_folds=0 ,early_stopping_rounds=200 
      ):    
    assert booster in ['gbtree', 'gblinear']
    assert objective in ['reg:linear', 'reg:logistic', 
      'binary:logistic', 'binary:logitraw', 'multi:softmax',
      'multi:softprob', 'rank:pairwise','count:poisson']
    assert eval_metric in [ 'rmse', 'mlogloss', 'logloss', 'error', 
      'merror',  'auc', 'ndcg', 'map', 'ndcg@n', 'map@n', 'kappa']
    if eval_metric=='kappa':
        booster='gblinear'
    self.silent = silent
    self.use_buffer = use_buffer
    self.num_round = num_round
    self.ntree_limit = ntree_limit
    self.nthread = nthread 
    self.booster = booster
    # Parameter for Tree Booster
    self.eta=eta
    self.gamma=gamma
    self.max_depth=max_depth
    self.min_child_weight=min_child_weight
    self.subsample=subsample
    self.colsample_bytree=colsample_bytree
    self.colsample_bylevel=colsample_bylevel
    self.max_delta_step=max_delta_step
    self.num_parallel_tree=num_parallel_tree
    self.min_split_loss=min_split_loss
    self.size_leaf_vector=size_leaf_vector
    self.default_direction=default_direction
    self.opt_dense_col=opt_dense_col
    self.sketch_eps=sketch_eps
    self.sketch_ratio=sketch_ratio
    self.k_folds=k_folds
    self.k_models=[]
    self.early_stopping_rounds=early_stopping_rounds
    
    # Parameter for Linear Booster
    self.l=l
    self.alpha=alpha
    self.lambda_bias=lambda_bias
    # Misc
    self.objective=objective
    self.eval_metric=eval_metric
    self.seed=seed
    self.num_class = num_class
    self.n_classes_ =num_class
    self.classes_=classes_
    

  def set_params(self,random_state=1):
      self.seed=random_state
      
  def build_matrix(self, X, opt_y=None, weighting=None):
    if opt_y==None: 
        if weighting==None:
            return xgb.DMatrix(csr_matrix(X), missing =-999.0)
        else :
            #scale weight
            sumtotal=float(X.shape[0])
            sumweights=np.sum(weighting)            
            for s in range(0,len(weighting)):
                weighting[s]*=sumtotal/sumweights
            return xgb.DMatrix(csr_matrix(X), missing =-999.0, weight=weighting)            
    else:
        if weighting==None:           
            return xgb.DMatrix(csr_matrix(X), label=np.array(opt_y), missing =-999.0)
        else :
            sumtotal=float(X.shape[0])
            sumweights=np.sum(weighting)            
            for s in range(0,len(weighting)):
                weighting[s]*=sumtotal/sumweights             
            return xgb.DMatrix(csr_matrix(X), label=np.array(opt_y), missing =-999.0, weight=weighting)         


  
  def fit(self, X, y,sample_weight=None):    
    
    self.k_models=[]
    
    X1 = self.build_matrix(X, y,weighting= sample_weight)#sample_weight)
    param = {}
    param['booster']=self.booster
    param['objective'] = self.objective
    param['bst:eta'] = self.eta
    param['seed']=  self.seed  
    param['bst:max_depth'] = self.max_depth
    if self.eval_metric!='kappa':
        param['eval_metric'] = self.eval_metric
    param['bst:min_child_weight']= self.min_child_weight
    param['silent'] =  1  
    param['nthread'] = self.nthread
    param['bst:subsample'] = self.subsample 
    param['gamma'] = self.gamma
    param['colsample_bytree']= self.colsample_bytree    
    param['num_parallel_tree']= self.num_parallel_tree   
    param['colsample_bylevel']= self.colsample_bylevel             
    #param['min_split_loss']=self.min_split_loss
    param['default_direction']=self.default_direction    
    param['opt_dense_col']=self.opt_dense_col        
    param['sketch_eps']=self.sketch_eps    
    param['sketch_ratio']=self.sketch_ratio            
    param['size_leaf_vector']=self.size_leaf_vector 

    if self.num_class is not None:
      param['num_class']= self.num_class
    if self.k_folds <2:
            self.bst = xgb.train(param.items(), X1, self.num_round)
    else :
        number_of_folds=self.k_folds
        kfolder2=StratifiedKFold(y, n_folds=number_of_folds,shuffle=True, random_state=self.seed)
        ## we split 64-16 5 times to make certain all the data has been use in modelling at least once 
        for train_indexnew, test_indexnew in kfolder2:
            if sample_weight==None:
                dtrain = xgb.DMatrix(X[train_indexnew], label=y[train_indexnew])
                dtvalid = xgb.DMatrix(X[test_indexnew], label=y[test_indexnew])
            else :
                dtrain = xgb.DMatrix(X[train_indexnew], label=y[train_indexnew], weight=sample_weight[train_indexnew])
                dtvalid = xgb.DMatrix(X[test_indexnew], label=y[test_indexnew], weight=sample_weight[test_indexnew])  
            
            watchlist = [(dtrain, 'train'), (dtvalid, 'valid')]
            gbdt = xgb.train(param.items(), dtrain, self.num_round, watchlist, verbose_eval=False, early_stopping_rounds=self.early_stopping_rounds)#, verbose_eval=250) #, early_stopping_rounds=250, verbose_eval=250) 
                
           #predsnew = gbdt.predict(dtest, ntree_limit=gbdt.best_iteration)  
            self.k_models.append(gbdt)

    return self

  def predict(self, X): 
    if  self.k_models!=None and len(self.k_models)<2:
        X1 = self.build_matrix(X)
        return self.bst.predict(X1)
    else :
        dtest = xgb.DMatrix(X)
        preds= [0.0 for k in X.shape[0]]
        for gbdt in self.k_models:
            predsnew = gbdt.predict(dtest, ntree_limit=(gbdt.best_iteration+1)*self.num_parallel_tree)  
            for g in range (0, predsnew.shape[0]):
                preds[g]+=predsnew[g]
        for g in range (0, len(preds)):
            preds[g]/=float(len(self.k_models))       
  
  def predict_proba(self, X): 
    try:
      rows=(X.shape[0])
    except:
      rows=len(X)
    X1 = self.build_matrix(X)
    if  self.k_models!=None and len(self.k_models)<2:
        predictions = self.bst.predict(X1)
    else :
        dtest = xgb.DMatrix(X)
        predictions= None
        for gbdt in self.k_models:
            predsnew = gbdt.predict(dtest, ntree_limit=(gbdt.best_iteration+1)*self.num_parallel_tree)  
            if predictions==None:
                predictions=predsnew
            else:
                for g in range (0, predsnew.shape[0]):
                    predictions[g]+=predsnew[g]
        for g in range (0, len(predictions)):
            predictions[g]/=float(len(self.k_models))               
        predictions=np.array(predictions)
    if self.objective == 'multi:softprob': return predictions.reshape( rows, self.num_class)
    return np.vstack([1 - predictions, predictions]).T
    



