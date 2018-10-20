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
from sklearn.ensemble import RandomForestClassifier

# small_trainingset = 'data/selected_train.csv' #0.1%
# medium_trainingset = 'data/train_medium_noNA.csv' #1%
# big_trainingset = 'data/complete_train_touse.csv' #100%
# onepercent = 'data/onepercent_noNA.csv' #1%
# trainingset = 'data/training_set.csv' #100% & original
training = 'data/training.csv'

def file_to_pandas():
    ''' This function opens the dataset and converts it to Pandas'''
    # train = pd.read_csv(small_trainingset)
    # train = pd.read_csv(medium_trainingset)
    train = pd.read_csv(training)
    # train = pd.read_csv(trainingset)

    # print 'Train dataset shape:', train.shape
    return train

def defining_sets(train):
    ''' Define the training and test sets/labels
    For now, I just split the training set, but this can be improved'''

    train_train = train.sample(1000)
    test_train = train.sample(100)
    global Ytrain
    Ytrain = train_train['position']
    global Xtrain
    Xtrain = train_train.drop(['booking_bool', 'click_bool', 'position'], 1)

    global Ytest
    Ytest = test_train['position']
    global Xtest
    Xtest = test_train.drop(['booking_bool', 'click_bool', 'position'], 1)

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

    rf = RandomForestClassifier(n_estimators = 100, max_depth = 10)
    rf = rf.fit(Xtrain, Ytrain)
    return rf


####################################
###### NCDG score trial ############


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    ''' Called by error_NDCG
    '''
    order = np.argsort(y_score)[::-1]
    y_true = y_true.tolist()
    y_true = np.take(y_true, order[:k], )

    gains = 2 ** y_true - 1

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def error_NDCG(Xtest, Ytest, rf):
    ''' Called by mainRF '''

    Ypred = rf.predict(Xtest)

    best = dcg_score(Ytest, Ytest, 10, "exponential")
    actual = dcg_score(Ytest, Ypred, 10, 'exponential')
    error = actual / best
    return error

def mainRF():
    """ Called by tree_and_accuracy"""
    rf = forest_classifier(Xtrain, Ytrain)

    error = error_NDCG(Xtest, Ytest, rf)
    return error, rf



def main():
    ## 1. Get the file (preprocessed!)
    train = file_to_pandas()

    ## 2. Get the test and training sets (if not given separately)
    defining_sets(train)

    ## 3.
    tree_and_accuracy()








if __name__ == "__main__":
    main()
