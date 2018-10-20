#!/usr/bin/python

"""
Authors: Elena Garcia, Alberto Gil Jimenez, Marie Corradi
Course: Data Mining Techniques
"""

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
from ast import literal_eval

small_trainingset = 'data/selected_train.csv' #0.1%
medium_trainingset = 'data/mediumsize_selected_train.csv' #1%
trainingset = 'data/training_set.csv' #100%

def file_to_pandas():
    ''' This function opens the dataset and converts it to Pandas'''
    # train = pd.read_csv(small_trainingset)
    train = pd.read_csv(medium_trainingset)
    # train = pd.read_csv(trainingset)

    # print 'Train dataset shape:', train.shape
    return train

def list_time(df):
    # Order the table by time
    df = df.sort_value('date_time')

    # List star rating
    totalrating = []
    for hotel in df.prop_id.unique():
        rows = df.loc[df['prop_id'] == hotel]
        rating = rows['prop_starrating'].tolist()
        totalrating.append(rating)
    print totalrating

    with open('results/time_starrating.txt', 'w') as f:
        f.write(str(totalrating))

    # List review rating
    totalrating = []
    for hotel in df.prop_id.unique():
        rows = df.loc[df['prop_id'] == hotel]
        rating = rows['prop_review_score'].tolist()
        totalrating.append(rating)
    print totalrating

    with open('results/time_reviews.txt', 'w') as f:
        f.write(str(totalrating))

def check_time():
    ''' Check if a hotel has a different starrating in two time points
    '''
    y = []
    n = 0
    with open('results/time_starrating.txt', 'r') as f:
        for line in f:
            for hotel in line.split(','):
                n += 1
                list_=[]
                hotel = hotel.strip('[')
                hotel = hotel.strip(']')
                for i in range(0,len(hotel)):
                    if hotel[i] not in ['[', ']', ' ', '\n']:
                        list_.append(int(hotel[i]))
                result = len(set(list_)) <= 1
                if result == False:
                    y.append(n)
    print y
    # Because printed y equals '[]', it means that all of the
    # hotels have the same starrating in all times.

    ''' Checking review scores
    --- doesn't work because of the points like 3.5'''
    # y = []
    # n = 0
    # with open('results/time_reviews.txt', 'r') as f:
    #     for line in f:
    #         for hotel in line.split(','):
    #             n += 1
    #             list_=[]
    #             hotel = hotel.strip('[')
    #             hotel = hotel.strip(']')
    #             for i in range(0,len(hotel)):
    #                 if hotel[i] not in ['[', ']', ' ', '\n']:
    #                     print hotel[i]
    #                     list_.append(int(hotel[i]))
    #             result = len(set(list_)) <= 1
    #             if result == False:
    #                 y.append(n)
    # print y


def main():
    ## 1. Import dataset and transform to Pandas
    # train = file_to_pandas()

    ## 2. List stars and reviews through time
    # list_time(train)

    ## 3. check similarity in stars
    check_time()

    #--- not finished ---
    ## Once the time_starrating.txt file is created
    ## 3. Plot the starrating and review per time
    # plot_time()
    ## 4. Find the hotels that appear the most in the searches



if __name__ == "__main__":
    main()
