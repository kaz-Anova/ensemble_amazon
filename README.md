# ensemble_amazon
Code to share different ensemble techniques with focus on meta-stacking , using data from Amazon.com - Employee Access Challenge kaggle competition

This code is part of the EE381V Large-Scale Machine Learning PhD level course in the University of Texas (Taught by Alexandros G. Dimakis) and aims to show different ensemble techniques for AUC type of problems (classification).

The code is for education purposes and did not aim to achieve a high score.

# Requirements

- Python 2.7
- Xgboost
- Sklearn
- numpy
- scipy
- pandas

download the train.csv and test.csv data from the kaggle competition :  Amazon.com - Employee Access Challenge
Link: https://www.kaggle.com/c/amazon-employee-access-challenge

# The ensemble methods

- The code initially creates a couple of models on different transformations of the data and saves the out-of-fold predictions
- We start testing different ensemble techniques as:
        * Simple average
        * Weighted average based on cv
        * Weighted Rank AVerage based on cv
        * Use Xgboost 


