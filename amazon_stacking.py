# -*- coding: utf-8 -*-
"""
@author: marios

Script that does meta modelling level 1 as in taking the held-out predictions from the previous models and using as features in a new model.

This one uses Extra trees to do this.

"""

import numpy as np
import gc
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold 
from sklearn.ensemble import ExtraTreesClassifier



#load a single file

def loadcolumn(filename,col=4, skip=1, floats=True):
    pred=[]
    op=open(filename,'r')
    if skip==1:
        op.readline() #header
    for line in op:
        line=line.replace('\n','')
        sps=line.split(',')
        #load always the last columns
        if floats:
            pred.append(float(sps[col]))
        else :
            pred.append(str(sps[col]))
    op.close()
    return pred            


#functions to manipulate pickles
    
def load_datas(filename):

    return joblib.load(filename)

def printfile(X, filename):

    joblib.dump((X), filename)
    
def printfilcsve(X, filename):

    np.savetxt(filename,X) 


    
def bagged_set(X,y,model, seed, estimators, xt, update_seed=True):
    
   # create array object to hold predictions 
   baggedpred=[ 0.0  for d in range(0, (xt.shape[0]))]
   #loop for as many times as we want bags
   for n in range (0, estimators):
        #shuff;e first, aids in increasing variance and forces different results
        X_t,y_c=shuffle(X,y, random_state=seed+n)
          
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

 

def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))      
     
def main():


        load_data=True          
        SEED=15
        Use_scale=False # if we want to use standard scaler on the data
        meta_folder="" # this is how you name the foler that keeps the held-out and test predictions to be used later for meta modelling
        # least of meta models. All the train held out predictions end with a 'train.csv' notation while all test set predictions with a 'test.csv'
        meta=["main_xgboost","main_logit_2way","main_logit_3way","main_logit_3way_best","main_xgboos_count","main_xgboos_count_2D","main_xgboos_count_3D"]
        
        bags=1 # helps to avoid overfitting. Istead of 1, we ran 10 models with differnt seed and different shuffling
        ######### Load files (...or not!) ############
       
        y = np.loadtxt("train.csv", delimiter=',',usecols=[0], skiprows=1)
        if load_data:
            Xmetatrain=None
            Xmetatest=None     
            for modelname in meta :
                    mini_xtrain=np.loadtxt(meta_folder + modelname + '.train.csv') # we load the held out prediction of the int'train.csv' model
                    mini_xtest=np.loadtxt(meta_folder + modelname + '.test.csv')   # we load the test set prediction of the int'test.csv' model
                    mean_train=np.mean(mini_xtrain) # we calclaute the mean of the train set held out predictions for reconciliation purposes
                    mean_test=np.mean(mini_xtest)    # we calclaute the mean of the test set  predictions      
                    # we print the AUC and the means and we still hope that everything makes sense. Eg. the mean of the train set preds is 1232314.34 and the test is 0.7, there is something wrong... 
                    print("model %s auc %f mean train/test %f/%f " % (modelname,roc_auc_score(np.array(y),mini_xtrain) ,mean_train,mean_test)) 
                    if Xmetatrain==None:
                        Xmetatrain=mini_xtrain
                        Xmetatest=mini_xtest
                    else :
                        Xmetatrain=np.column_stack((Xmetatrain,mini_xtrain))
                        Xmetatest=np.column_stack((Xmetatest,mini_xtest))
            # we combine with the stacked features
            X=Xmetatrain
            X_test=Xmetatest 
            # we print the pickles
            printfile(X,"xmetahome.pkl")  
            printfile(X_test,"xtmetahome.pkl")     

            X=load_datas("xmetahome.pkl")              
            print("rows %d columns %d " % (X.shape[0],X.shape[1] ))
            #X_test=load_datas("onegramtest.pkl")
            #print("rows %d columns %d " % (X_test.shape[0],X_test.shape[1] ))                   
        else :

            X=load_datas("xmetahome.pkl")              
            print("rows %d columns %d " % (X.shape[0],X.shape[1] ))
            #X_test=load_datas("onegramtest.pkl")
            #print("rows %d columns %d " % (X_test.shape[0],X_test.shape[1] ))    
            
        
        outset="amazon_stacking" # Name of the model (quite catchy admitedly)
        number_of_folds =5 # repeat the CV procedure 5 times and save the holdout predictions


        print("len of target=%d" % (len(y))) # print the length of the target variable because we can
        
        #model we are going to use
                       

        model=ExtraTreesClassifier(n_estimators=10000, criterion='entropy', max_depth=9,  min_samples_leaf=1, max_features=6, n_jobs=30, random_state=1)        
        #model=LogisticRegression(C=0.01)
        train_stacker=[ 0.0  for k in range (0,(X.shape[0])) ] # the object to hold teh held-out preds
        
        # CHECK EVerything in five..it could be more efficient     
        
        #Will hold the average AUC value. I leave it as 'mean_kapa' as I copied the code from the Flower competition.
        #(Always re-use old good code.) Pepperidge Farm remembers.     
        mean_auc = 0.0 
        # cross validation model we are going to use
        kfolder=StratifiedKFold(y, n_folds=number_of_folds,shuffle=True, random_state=SEED)       
        i=0 # iterator counter
        print ("starting cross validation with %d kfolds " % (number_of_folds)) # some words to keep you engaged
        if number_of_folds>0: # this basically means "if we are doing cross validation ", sometimes my model crashed after cv and I had to skip this part in 2nd iteration
            for train_index, test_index in kfolder:
                # get train (set and target variable)
                X_train = X[train_index]    
                y_train= np.array(y)[train_index]
                #talk about it
                print (" train size: %d. test size: %d, cols: %d " % (len(train_index) ,len(test_index) ,(X_train.shape[1]) ))

                if Use_scale:
                    stda=StandardScaler()            
                    X_train=stda.fit_transform(X_train)

                # get validation (set and target variable)
                X_cv= X[test_index]
                y_cv = np.array(y)[test_index]
                
                if Use_scale:
                    X_cv=stda.transform(X_cv)
                
                # bag 10 models of the selected type :)
                preds=bagged_set(X_train,y_train,model, SEED + i, bags, X_cv, update_seed=True)
                # measure AUC, but keep confusing everyone by naming "kapa"
                auc = roc_auc_score(y_cv,preds)                        
                
                # talk about it, hoping people will not actually find out about it
                print "size train: %d size cv: %d AUC (fold %d/%d): %f" % (len(train_index), len(test_index), i + 1, number_of_folds, auc)
             
                mean_auc += auc
                
                #update the held-out array for meta modelling
                no=0
                for real_index in test_index:
                         train_stacker[real_index]=(preds[no])
                         no+=1
                i+=1 # update iterator
                
            if (number_of_folds)>0: # if we did cross validation the print the results
                mean_auc/=number_of_folds
                print (" Average AUC: %f" % (mean_auc) ) # keep calling it AUC (because you can)
                
            print (" printing train datasets in %s" % (meta_folder+ outset + "train.csv")) # print the hold out predictions


        print ("start final modeling")
        
        if Use_scale:
            stda=StandardScaler()            
            X=stda.fit_transform(X)


        #load test data
        X_test=load_datas("xtmetahome.pkl")
        if Use_scale:
            X_test=stda.transform(X_test) 
            
        # giving stats never gets old
        print("rows %d columns %d " % (X_test.shape[0],X_test.shape[1] ))
        # bag 10 models of the selected type :)
        preds=bagged_set(X, y,model, SEED, bags, X_test, update_seed=True)          

        
        X_test=None
        gc.collect()  # collect the garbage   
        #load the id variable to put it in the submission file (if we ant to submit this model in kaggle)
        
        print (" printing test datasets in %s" % (meta_folder + outset + "test.csv"))          
        
        save_results(preds, outset+"_submission_" +str(mean_auc) + ".csv")         

        print("Done.")  
       
       
       



if __name__=="__main__":
  main()
