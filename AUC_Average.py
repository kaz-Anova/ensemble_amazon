
import numpy as np
import operator
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

"""
Script to do the final ensemble via averaging 

using many cross validations schemas for better AUC score

"""


#convert a single 1-dimensional array to rank (e.g sort the score from the smallest to the highest and give them scores as 1,2...len(array))
def ranking(score):
    """ method to create a score into rank"""
    data=[]
    for i in range(len(score)):
        data.append([score[i],i])
    data=sorted(data, key=operator.itemgetter(0), reverse=False)
    value=data[0][0]
    data[0][0]=1
    for i in range(1,len(score)):
        val=data[i][0]
        if val>value :
            value=val
            data[i][0]=(i+1)
        else :
            data[i][0]=data[i-1][0]
    data=sorted(data, key=operator.itemgetter(1), reverse=False)
    final_rank=[]
    for i in range(len(score)):
        final_rank.append(data[i][0])
    return final_rank

#retrieve specific column fron 2dimensional array as a 1dimensional array
def select_column(data, col) :
    array=[]
    for i in range(len(data)):
       array.append(data[i][col])
    return array

# put an array back to the given column j
def putcolumn(data,array,j) :
    for i in range(len(data)):
        data[i][j]=array[i]

# convert each one of the columns in the given array to ranks
def create_ranklist (data ) :
    for j in range(len(data[0])):
        putcolumn( data,ranking(select_column(data,j)),j)
   

# method to load a specific column
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

def printfilcsve(X, filename):

    np.savetxt(filename,X) 

 
def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))
            
def main():
    
        # meta models to be used to assess how much weight to contribute to the final submission
        meta=["main_xgboost","main_logit_2way","main_logit_3way","main_logit_3way_best","main_xgboos_count","main_xgboos_count_2D","main_xgboos_count_3D"]

        y = np.loadtxt("train.csv", delimiter=',',usecols=[0], skiprows=1)

        
        print("len of target=%d" % (len(y))) # reconciliation check
        weights=[1, # all weights to 1, e.g. average
                 1,
                 1,
                 1,
                 1,                 
                 1,
                 1
                 ]     # the weights of the 4 level 3 meta models
                 


        number_of_folds=5 # for cv
        usesccaling_to_0_1=True # some submissions need probas-ish
        use_geo=False #false = uses linear rank average
        Load=True
        use_rank=False # IF we want to use rank
        #basiclaly it says multiple the extra lvl3 model by 1, the xgboost model by 0.05 and the neural net with 0.25
        if Load:
            Xmetatrain=None
            Xmetatest=None   
            #append all the predictions into 1 list (array)
            for modelname in meta :
                mini_xtrain=np.loadtxt(modelname + '.train.csv')
                mini_xtest=np.loadtxt(modelname + '.test.csv')  
                mean_train=np.mean(mini_xtrain)
                mean_test=np.mean(mini_xtest)               
                print("model %s auc %f mean train/test %f/%f " % (modelname,roc_auc_score(y,mini_xtrain) ,mean_train,mean_test)) 
                if Xmetatrain==None:
                    Xmetatrain=mini_xtrain
                    Xmetatest=mini_xtest
                else :
                    Xmetatrain=np.column_stack((Xmetatrain,mini_xtrain))
                    Xmetatest=np.column_stack((Xmetatest,mini_xtest))
            # convert my scores to list
    
            X=Xmetatrain
            X_test=Xmetatest
            joblib.dump((X,X_test),"METADUMP.pkl" )
        else :
            X,X_test=joblib.load("METADUMP.pkl")
            
        outset="AUC_Average" # Output base name


        seedlist=[87, 111, 1337, 42 , 201628] # many seeds for more accurate results
        train_stacker=[0.0 for i in range (0,len(X))]
        mean_auc = 0.0
        for seeder in seedlist:
            print("kfolding seed %d " % (seeder) )
            kfolder=StratifiedKFold(y, n_folds=number_of_folds,shuffle=True, random_state=seeder)
            #number_of_folds=0
            #X,y=shuffle(X,y, random_state=SEED) # Shuffle since the data is ordered by time
            i=0 # iterator counter
            print ("starting cross validation with %d kfolds " % (number_of_folds))
            if number_of_folds>0:
                for train_index, test_index in kfolder:
                    # creaning and validation sets
                    X_train, X_cv = X[train_index], X[test_index]
                    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
                    print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))
                    
    
                    minmax=MinMaxScaler(feature_range=(0, 1))
                    X_cv=X_cv.tolist()
                    if use_rank:
                        create_ranklist(X_cv)
                    
                    #X_cv= minmax.fit_transform((X_cv))
                    #print X_cv
    
                    if use_geo: # use geo mean
                        preds=[1.0 for s in range (0,len(X_cv))]
                        for i in range (0,len(X_cv)) :
                            for j in range (0,len(weights)) :
                                preds[i]*=X_cv[i][j]**weights[j] 
                    else :
                        preds=[0.0 for s in range (0,len(X_cv))]
                        for i in range (0,len(X_cv)) :
                            for j in range (0,len(weights)) :
                                preds[i]+=X_cv[i][j]*weights[j]
                    
                    if usesccaling_to_0_1:
                        preds= minmax.fit_transform(preds)
    
                    # compute Loglikelihood metric for this CV fold
                    #scalepreds(preds)     
                    AUC = roc_auc_score(y_cv,preds)
                    print "size train: %d  CV : %d AUC (fold %d/%d): %f" % ((X_train.shape[0]), len(X_cv), i + 1, number_of_folds, AUC)
                 
                    mean_auc += AUC
                    #save the results
                    no=0
                    for real_index in test_index:
                             train_stacker[real_index]=(preds[no])
                             no+=1
                    i+=1
  
        mean_auc/=(len(seedlist)*5.0)
        print ("Average AUC: %f" % mean_auc)
        minmax=MinMaxScaler(feature_range=(0, 1))
        X_test=X_test.tolist()
        if use_rank:
            create_ranklist(X_test)    

        # combine all the ranked scores in a weighted manner for the test lvl 3 out-of-fold predictions 
       
        
        if use_geo: # use geo mean
            preds=[1.0 for s in range (0,len(X_test))]
            for i in range (0,len(X_test)) :
                for j in range (0,len(weights)) :
                    preds[i]*=X_test[i][j]**weights[j]      
        else : # linear wighted rank average
            preds=[0.0 for s in range (0,len(X_test))]
            for i in range (0,len(X_test)) :
                for j in range (0,len(weights)) :
                    preds[i]+=X_test[i][j]*weights[j]

        if usesccaling_to_0_1:
            preds= minmax.fit_transform(preds)  
            
        
        #convert to numpy
        preds=np.array(preds)
        #write the results

        save_results(preds, outset+"_submission_" +str(mean_auc) + ".csv")   
        print("Done.")  

       
       



if __name__=="__main__":
  main()
