###### "Data mining" course, Vrije Universiteit Amsterdam #################
###### Authors: Elena Garcia Lara, Alberto Gil, Marie Corradi #############

# Import desired libraries
import random
import os
import matplotlib
import pylab as plt
from operator import itemgetter
import numpy as np
from datetime import datetime, date
import time
import pandas as pd
from random import randint
import random
import sklearn
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import math

# small_trainingset = 'data/selected_train.csv' #0.1%
# medium_trainingset = 'data/train_medium_noNA.csv' #1%
# big_trainingset = 'data/complete_train_touse.csv' #100%
# onepercent = 'data/onepercent_noNA.csv' #1%
# trainingset = 'data/training_set.csv' #100% & original
training = 'data/training.csv'
#
def file_to_pandas(filename):
    ''' This function opens the dataset and converts it to Pandas'''
    # train = pd.read_csv(small_trainingset)
    # train = pd.read_csv(medium_trainingset)
    # train = pd.read_csv(training)
    # train = pd.read_csv(big_trainingset)

    train = pd.read_csv(filename)

    # print 'Train dataset shape:', train.shape
    return train

def defining_sets(train, format_):
    ''' Define the training and test sets/labels
    For now, I just split the training set, but this can be improved'''

    if format_ == 'svm':
        #Train
        train_train = train.loc[1:2000]
        train_train = train_train.fillna(0)
        X = train_train.drop(['booking_bool', 'click_bool', 'position'], 1)
        y = train_train.position
        f = 'data/training_svmlight.dat'
        sklearn.datasets.dump_svmlight_file(X, y, f, query_id=train_train.srch_id)

        #Val
        val_train = train.loc[2001:3000]
        val_train = val_train.fillna(0)
        X = val_train.drop(['booking_bool', 'click_bool', 'position'], 1)
        y = val_train.position
        f = 'data/val_svmlight.dat'
        sklearn.datasets.dump_svmlight_file(X, y, f, query_id=val_train.srch_id)

        #Test
        test_train = train.loc[3001:49583]
        test_train = test_train.fillna(0)
        X = test_train.drop(['booking_bool', 'click_bool', 'position'], 1)
        y = test_train.position
        f = 'data/test_svmlight.dat'
        sklearn.datasets.dump_svmlight_file(X, y, f, query_id=test_train.srch_id)

    elif format_ == 'pandas':
        train_train = train.loc[1:49583]
        test_train = train.loc[3001:49583]
        global Ytrain
        Ytrain = train_train['booking_bool']
        global Xtrain
        Xtrain = train_train.drop(['booking_bool', 'click_bool', 'position'], 1)
        # Xtrain = train_train[['month', 'prop_location_score2']]

        global Ytest
        Ytest = test_train['booking_bool']
        global Xtest
        Xtest = test_train.drop(['booking_bool', 'click_bool', 'position'], 1)
        # Xtest = test_train[['month', 'prop_location_score2']]

        Ytrain = Ytrain.fillna(0)
        Xtrain = Xtrain.fillna(0)
        Ytest = Ytest.fillna(0)
        Xtest = Xtest.fillna(0)


def tree_and_accuracy():
    """ Builds a tree with Xtrain (data) and Ytrain (labels) set. Calculates
    the accuracy (misclassifications) for the Xtest and Ytest"""
    accuracyLabel, rf = mainRF()
    print accuracyLabel

def forest_classifier(Xtrain, Ytrain):
    """ Generate the random forest,
    called by mainRF"""

    rf = RandomForestRegressor(n_estimators = 100)#, max_depth = 10)
    rf = rf.fit(Xtrain, Ytrain)
    return rf

# def dcg_score(y_true, y_score, k=10, gains="exponential"):
#     ''' Called by error_NDCG
#     '''
#     order = np.argsort(y_score)[::-1]
#     y_true = y_true.tolist()
#     y_true = np.take(y_true, order[:k], )
#
#     gains = 2 ** relevance_function(y_true, y_score) - 1
#
#     # highest rank is 1 so +2 instead of +1
#     discounts = np.log2(np.arange(len(y_true)) + 2)
#     return np.sum(gains / discounts)
#
# def relevance_function(y_true, y_pred):
#     ##complete
#     relevance = 0
#     # # if booking_bool == 1 for this search_id in TRUE labels
#     #     # if booking_bool == 1 for this search_id in PREDICTED labels
#     #         relevance += 5
#     #
#     # # if booking_bool == 1 for this search_id in TRUE labels
#     #     # if booking_bool == 1 for this search_id in PREDICTED labels
#     #         relevance += 5
#
#
# def error_NDCG(Xtest, Ytest, rf):
#     ''' Called by mainRF '''
#     Ypred = rf.predict(Xtest)
#     print Ypred
#     exit()
#
#     #### List the results by the 'probability' of the hotel beeing booked
#
#
#     best = dcg_score(Ytest, Ytest, 10, "exponential")
#     actual = dcg_score(Ytest, Ypred, 10, 'exponential')
#     error = actual / best
#     return error
#
#     #### AVERAGE OVER ALL QUERIES



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

# Remember --> subsetdf = df['srch_ID' == srch_ID]
def calculate_dcg_score(subsetdf):
    # Sort the rows by the predicted probabilities of being booked
    """NOTE!!! I HAVE PUT HERE AN ARBITRARY NAME TO THE COLUMN (predicted_booking). CHANGE IF REQUIRED"""
    """NOTE!!! I have assumed that subsetdf contains also the relevance_score column. CHANGE IF REQUIRED"""
    sorting = subsetdf.sort_values(['predicted_booking'], ascending=False)['relevance_score'].tolist()
    # Calculate the DCG
    DCG = 0
    position = 1
    for score in sorting:
        DCG += (math.pow(2,score) - 1) / math.log(position + 1, 2)
        position += 1
    return DCG

# Remember --> subsetdf = df['srch_ID' == srch_ID] AND it already has the 'relevance_score' column assigned
# Calculates the DCG score of the IDEAL ORDERING in a search (gold standard)
def calculate_ideal_dcg_score(subsetdf):
    optimum_sorting = subsetdf.sort_values(['relevance_score'], ascending=False)['relevance_score'].tolist()
    DCG = 0
    position = 1
    for score in optimum_sorting:
        DCG += (math.pow(2,score) - 1) / math.log(position + 1, 2)
        position += 1
    return DCG

""" MAIN EXAMPLE for implementing calculation of NDCG """
def main():
    train = file_to_pandas('data/SUBSET_processed_testset.csv')
    # Create a list for appending the NDCG score of every search
    NDCG_list = []
    for search_ID in train.srch_id.unique().tolist():
        # this if is for avoiding further errors
        if math.isnan(search_ID):
            pass
        else:
            # Create a dataframe with the info of that search ID
            nd = train[train['srch_id'] == search_ID]
            # Assign relevance score to every row according to booking_bool and
            # click_bool information (5 == (book_bool = 1), 1 == (click_bool = 1 AND book_bool=0), 0 == otherwise
            nd = assign_relevance_score(nd)
            # Calculate the score of the IDEAL ORDERING IDCG (Ideal DCG)
            idcg = calculate_ideal_dcg_score(nd)
            # Only compute the DCG score for search_ID which at least have one click and/or one booking (otherwise IDCG = 0)
            if idcg > 0:
                # Calculate the DCG score
                obtained_score = calculate_dcg_score(nd)
                # Calculate normalized DCG score (NDCG)
                NDCG = float(obtained_score) / float(idcg)
                # Append the NDCG obtained to the NDCG list
                NDCG_list.append(NDCG)
    # Calculate the average of the NDCG obtained over all searches
    mean_NDCG = np.mean(NDCG_list)


def mainRF():
    """ Called by tree_and_accuracy"""
    rf = forest_classifier(Xtrain, Ytrain)

    error = error_NDCG(Xtest, Ytest, rf)
    return error, rf



def main_good():
    ## 1. Get the file (preprocessed!)
    train = file_to_pandas()

    ## 2. Get the test and training sets (if not given separately)
    defining_sets(train, format_='pandas')
            # Format_ can be 'svm' or 'pandas'


    ## 3.
    tree_and_accuracy()








if __name__ == "__main__":
    main()
