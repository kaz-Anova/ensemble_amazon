""" Amazon Access Challenge Code for ensemble

Marios Michaildis script for Amazon .

Uses 3-way interactions and logistic regression

based on Paul Duan's Script.
 

"""
from __future__ import division
import numpy as np
from sklearn import  preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import XGBoostClassifier as xg
from sklearn.cross_validation import StratifiedKFold

SEED = 42  # always use a seed for randomized procedures


def load_data(filename, use_labels=True):
    """
    Load data from CSV files and return them as numpy arrays
    The use_labels parameter indicates whether one should
    read the first column (containing class labels). If false,
    return all 0s. 
    """

    # load column 1 to 8 (ignore last one)
    data = np.loadtxt(open( filename), delimiter=',',
                      usecols=range(1, 9), skiprows=1)
    if use_labels:
        labels = np.loadtxt(open( filename), delimiter=',',
                            usecols=[0], skiprows=1)
    else:
        labels = np.zeros(data.shape[0])
    return labels, data


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
    

# compute all pairs of variables
def Make_3way(X, Xt):
    columns_length=X.shape[1]
    for j in range (columns_length):
        for d in range (j+1,columns_length):  
            print("Adding columns' interraction %d and %d" % (j, d) )
            new_column_train=X[:,j]+X[:,d]
            new_column_test=Xt[:,j]+Xt[:,d]    
            X=np.column_stack((X,new_column_train))
            Xt=np.column_stack((Xt,new_column_test))
    for j in range (columns_length):
        for d in range (j+1,columns_length):  
            for m in range (d+1,columns_length):              
                print("Adding columns' interraction %d and %d and %d" % (j, d, m) )
                new_column_train=X[:,j]+X[:,d]+X[:,m]
                new_column_test=Xt[:,j]+Xt[:,d]+Xt[:,m]      
                X=np.column_stack((X,new_column_train))
                Xt=np.column_stack((Xt,new_column_test))            
            
    return X, Xt
    
    

def main():
    """
    Fit models and make predictions.
    We'll use one-hot encoding to transform our categorical features
    into binary features.
    y and X will be numpy array objects.
    """
    
    filename="main_logit_3way" # nam prefix
    model = LogisticRegression(C=0.7, penalty="l2")  # the classifier we'll use
    
    # === load data in memory === #
    print "loading data"
    y, X = load_data('train.csv')
    y_test, X_test = load_data('test.csv', use_labels=False)
    
    X,X_test= Make_3way(X, X_test)# add interractions
    # === one-hot encoding === #
    # we want to encode the category IDs encountered both in
    # the training and the test set, so we fit the encoder on both
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_test = encoder.transform(X_test)


    # if you want to create new features, you'll need to compute them
    # before the encoding, and append them to your dataset after

    #create arrays to hold cv an dtest predictions
    train_stacker=[ 0.0  for k in range (0,(X.shape[0])) ] 

    # === training & metrics === #
    mean_auc = 0.0
    bagging=1 # number of models trained with different seeds
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
