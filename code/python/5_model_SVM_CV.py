#!/usr/bin/python

"""
Authors: Elena Garcia
Course: Data Mining Techniques
"""

# Import required libraries
# from sklearn import cross_validation
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import metrics # cross-validation metric
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score, permutation_test_score, StratifiedKFold, GridSearchCV, cross_val_predict, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, mean_absolute_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from itertools import cycle
from sklearn.svm import SVR
import pickle

def file_to_pandas(datafile):
    ''' This function opens the dataset and converts it to Pandas'''
    train = pd.read_csv(datafile)
    return train

# Tuning parameters of SVM
def tuning_parameters(X_train, y_train):
    # Set the parameters by cross-validation

    # parameters to be tuned
    model = SVR()
    c_values  = [0.1, 1, 2, 10]
    gamma_values = [1e-1, 1e-2, 1e-5]
    parameters = {'kernel':('linear', 'poly','rbf', 'sigmoid'), 'C': c_values, 'gamma':gamma_values }


    clf = GridSearchCV(model, param_grid=parameters, cv=10, scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)
    # 10-Fold Cross validation
    # scores =  cross_val_score(clf, X_train, y_train, cv=10,scoring='neg_mean_squared_error')
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print "Grid scores on development set: "

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    	print "%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params)

    with open('results/best_param_svm.txt', 'w') as f:
        f.write(clf.best_params_)

    return clf.best_params_

def train_RF(parameters,X_train,y_train,X_test,y_test):

    model = SVR(n_estimators=parameters['n_estimators'], max_depth=parameters['max_depth'])
    k_fold = KFold(n_splits=10, random_state=True)
    model.fit(X_train, y_train)

    print model.predict(X_test)
    print y_test

    #distance = svr_rbf.decision_function(X_test)
    predicted_labels = model.predict(X_test)
    pred_lb = predicted_labels.tolist()
    RMSE = mean_absolute_error(y_test,predicted_labels)
    print 'rmse is', RMSE
    filename = 'RF_model_1.sav'
    pickle.dump(model, open(filename, 'wb'))


# Remember --> subsetdf = df['srch_ID' == srch_ID]
def assign_relevance_score(subsetdf):
    # Create a relevance_score column for assigning the relevance srch_query_affinity_score
    # Assign a 0 relevance score to all rows
    subsetdf['relevance_score'] = 0
    # Assign a relevance score of 5 to all rows having booking_bool = 1 (bookings)
    indices_bookings = subsetdf[subsetdf['booking_bool']==1].index.tolist()
    subsetdf.loc[indices_bookings,'relevance_score'] = 5
    # Assign a relevance score of 1 to all rows having booking_bool = 0 AND click_bool = 1 (clicks without booking)
    indices_clicks = subsetdf[(subsetdf['booking_bool']==0) & (subsetdf['click_bool']==1)].index.tolist()
    subsetdf.loc[indices_clicks,'relevance_score'] = 1
    return subsetdf

# Deletes non-desired columns from the dataframe
def delete_useless_columns(dataframe):
    useless_variables = ['Unnamed: 0', 'Unnamed: 0.1', 'srch_id','year','click_bool','position','booking_bool']
    for variable in useless_variables:
        del dataframe[variable]
    return dataframe

# this function yields 2 dataframes:
# 1 - Dataframe containing input vectors for a model
# 2 - Dataframe containing output values (variable that is going to be predicted) y a model
def separate_input_and_target_variable(dataframe):
    y = dataframe['relevance_score']
    dataframe = dataframe.drop('relevance_score', 1)
    X = dataframe.as_matrix()
    return  X, y

def prepare_df(filename,trainingset_do=False):
    train = file_to_pandas(filename)
    train = assign_relevance_score(train)
    train = delete_useless_columns(train)
    train = train.dropna()
    if trainingset_do:
        # Shuffle rows from the training set
        train = train.sample(frac=1)
    X, y = separate_input_and_target_variable(train)
    return X, y

# Main function
def main():
    print 'Please wait...'
    print 'Getting the train set'
    train = file_to_pandas('data/training_final1.csv')
    # Assign the relevance score (TARGET VARIABLE) to the dataframe
    train = assign_relevance_score(train)
    train = delete_useless_columns(train)
    train = train.dropna()

    print 'Getting the train set'
    test = file_to_pandas('data/test_final_notdownsampled.csv')
    # Assign the relevance score (TARGET VARIABLE) to the dataframe
    test = assign_relevance_score(test)
    test = delete_useless_columns(test)
    test = test.dropna()

    X_test, y_test = separate_input_and_target_variable(test)


    # Shuffle rows from the training set
    train = train.sample(frac=1)
    # Obtain X to be trained and predicted values (y_train)
    # X_train, y_train = separate_input_and_target_variable(train)
    X_train, y_train = prepare_df('data/training_final1.csv',True)

    best_parameters = tuning_parameters(X_train, y_train)
    train_RF(best_parameters,X_train,y_train,X_test,y_test)



if __name__ == "__main__":main()
