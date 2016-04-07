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

* The code initially creates a couple of models on different transformations of the data and saves the out-of-fold predictions
* We start testing different ensemble techniques as:
       - Simple average
       - Weighted average based on cv
       - Weighted Rank Average based on cv
       - Geomean Weighted Rank Average based on cv
       - Use another model (ExtraTreesClassifier from sklearn) to perform meta-stacking 

# Replicate solution

Inisde a folder that the train.csv and test.csv are present :

* Run amazon_main_xgboost_count_2D.py
* Run amazon_main_logit_3way_best.py
* Run amazon_main_logit_2D.py
* Run amazon_main_xgboost.py
* Run amazon_main_logit_3way.py
* Run amazon_main_xgboost_count.py
* Run amazon_main_xgboost_count_3D.py

This will yield the following results in Kaggle's Private Leaderboard and internal 5-fold cv

Model name | AUC - Private LB | AUC- CV 5-fold
--- | --- | ---
main_xgboost | 0.89096 | 0.876971
amazon_main_logit_2D | 0.89534 | 0.877267
main_logit_3way | 0.89554 | 0.878507
main_logit_3way_best | 0.89792 | 0.882932
main_xgboos_count | 0.88187 | 0.870671
main_xgboos_count_2D | 0.90127 | 0.888981
main_xgboos_count_3D | **0.904** | **0.893425**


* Run AUC_Average.py
* Run AUC_Weighted_Average.py
* Run AUC_Rank_Weighted_Average.py
* Run AUC_Geo_Rank_Weighted_Average.py
* Run amazon_stacking.py

This will yield:

Model name | AUC - Private LB | AUC- CV 5-fold
--- | --- | ---
AUC_Average | 0.90725 | 0.893209
AUC_Weighted_Average | 0.91121 | 0.899529
AUC_Rank_Weighted_Average | 0.90916 | 0.897925
AUC_Geo_Rank_Weighted_Average | 0.90988 | 0.898586
amazon_stacking | **0.91206** | **0.899851**







