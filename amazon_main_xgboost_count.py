""" Amazon Access Challenge Code for ensemble

Marios Michaildis script for Amazon .

Uses counts as features  and xgboost

based on Paul Duan's Script.
 


"""
from __future__ import division
import numpy as np
from sklearn import  preprocessing
from sklearn.metrics import roc_auc_score
import XGBoostClassifier as xg
from sklearn.cross_validation import StratifiedKFold
import pandas as pd

SEED = 42  # always use a seed for randomized procedures


def load_datacount(tr,te):
    #w ewill use pandas
    train  = pd.read_csv(tr, sep=',',quotechar='"')
    test  = pd.read_csv(te, sep=',',quotechar='"')
    label=  np.array(train['ACTION']).astype(float)
    train.drop('ACTION', axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)    
    test.drop('ROLE_CODE', axis=1, inplace=True)
    train.drop('ROLE_CODE', axis=1, inplace=True)    

    
    train_s = train
    test_s = test
    result = pd.concat([test_s,train_s])
    headers=[f for f in result.columns]
    
    for i in range(train_s.shape[1]):  
            print headers[i], len(np.unique(result[headers[i]]))    
            cnt = result[headers[i]].value_counts().to_dict()
            #cnt = dict((k, -1) if v < 3  else (k,v) for k, v in cnt.items()  ) # if u want to encode rare values as "special"       
            train_s[headers[i]].replace(cnt, inplace=True)                                  
            test_s[headers[i]].replace(cnt, inplace=True)             
          

    train = np.array(train_s).astype(float)
    test = np.array(test_s).astype(float)
    
  
    print train.shape
    print test.shape    
    
    
    return label, train,test


def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))


def bagged_set(X_t,y_c,model, seed, estimators, xt, update_seed=True):
    
   # create array object to hold predictions 
   baggedpred=[ 0.0  for d in range(0, (xt.shape[0]))]
   #loop for as many times as we want bags
   for n in range (0, estimators):
        #shuff;e first, aids in increasing variance and forces different results
        #X_t,y_c=shuffle(Xs,ys, random_state=seed+n)
          
        if update_seed: # update seed if requested, to give a slightly different model
            model.set_params(random_state=seed + n)
        model.fit(X_t,y_c) # fit model0.0917411475506
        preds=model.predict_proba(xt)[:,1] # predict probabilities
        # update bag's array
        for j in range (0, (xt.shape[0])):           
                baggedpred[j]+=preds[j]
   # divide with number of bags to create an average estimate            
   for j in range (0, len(baggedpred)): 
                baggedpred[j]/=float(estimators)
   # return probabilities            
   return np.array(baggedpred) 
   
   
# using numpy to print results
def printfilcsve(X, filename):

    np.savetxt(filename,X) 
    

def main():
    """
    Fit models and make predictions.
    We'll use one-hot encoding to transform our categorical features
    into binary features.
    y and X will be numpy array objects.
    """
    
    filename="main_xgboos_count" # nam prefix
    #model = linear_model.LogisticRegression(C=3)  # the classifier we'll use
    
    model=xg.XGBoostClassifier(num_round=1000 ,nthread=25,  eta=0.02, gamma=1,max_depth=20, min_child_weight=0.1, subsample=0.9, 
                                   colsample_bytree=0.5,objective='binary:logistic',seed=1) 
    # === load data in memory === #
    print "loading data"
    y, X,X_test = load_datacount('train.csv','test.csv')


    # === one-hot encoding === #
    # we want to encode the category IDs encountered both in
    # the training and the test set, so we fit the encoder on both



    # if you want to create new features, you'll need to compute them
    # before the encoding, and append them to your dataset after

    #create arrays to hold cv an dtest predictions
    train_stacker=[ 0.0  for k in range (0,(X.shape[0])) ]  

    # === training & metrics === #
    mean_auc = 0.0
    bagging=20 # number of models trained with different seeds
    n = 5  # number of folds in strattified cv
    kfolder=StratifiedKFold(y, n_folds= n,shuffle=True, random_state=SEED)     
    i=0
    for train_index, test_index in kfolder: # for each train and test pair of indices in the kfolder object
        # creaning and validation sets
        X_train, X_cv = X[train_index], X[test_index]
        y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
        #print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))

        # if you want to perform feature selection / hyperparameter
        # optimization, this is where you want to do it

        # train model and make predictions 
        preds=bagged_set(X_train,y_train,model, SEED , bagging, X_cv, update_seed=True)   
        

        # compute AUC metric for this CV fold
        roc_auc = roc_auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        mean_auc += roc_auc
        
        no=0
        for real_index in test_index:
                 train_stacker[real_index]=(preds[no])
                 no+=1
        i+=1
        

    mean_auc/=n
    print (" Average AUC: %f" % (mean_auc) )
    print (" printing train datasets ")
    printfilcsve(np.array(train_stacker), filename + ".train.csv")          

    # === Predictions === #
    # When making predictions, retrain the model on the whole training set
    preds=bagged_set(X, y,model, SEED, bagging, X_test, update_seed=True)  

    
    #create submission file 
    printfilcsve(np.array(preds), filename+ ".test.csv")  
    #save_results(preds, filename+"_submission_" +str(mean_auc) + ".csv")

if __name__ == '__main__':
    main()
