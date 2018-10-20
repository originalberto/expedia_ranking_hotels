###### "Data mining" course, Vrije Universiteit Amsterdam #################
###### Author: Alberto Gil ################################

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
# from data_exploration_v2alb import file_to_pandas
import random


small_trainingset = 'data/selected_train.csv' #0.1%
medium_trainingset = 'data/mediumsize_selected_train.csv' #1%
trainingset = 'data/training_set.csv' #100%

def file_to_pandas():
    ''' This function opens the dataset and converts it to Pandas'''
    # train = pd.read_csv(small_trainingset)
    # train = pd.read_csv(medium_trainingset)
    train = pd.read_csv(trainingset)

    # print 'Train dataset shape:', train.shape
    return train

def create_testset(original_df):
    ## Look for the rows which have month = 5 or 6 and year = 2013 (TEST SET)
    new_df = original_df[(original_df['year'] == 2013) & ((original_df['month'] == 5) | ((original_df['month'] == 6)))]
    indices = original_df[(original_df['year'] == 2013) & ((original_df['month'] == 5) | ((original_df['month'] == 6)))].index.tolist()

    # remove the retrieved rows in "new_df" from the orignal dataframe
    original_df.drop(original_df.index[indices], inplace=True)

    # save new training and test set
    new_df.to_csv('data/unprocessed_testset.csv')
    original_df.to_csv('data/unprocessed_trainingset.csv')
    #return final_df

def main():
    train = file_to_pandas()
    create_testset(train)
    #print new_df

if __name__ == "__main__":
    main()
