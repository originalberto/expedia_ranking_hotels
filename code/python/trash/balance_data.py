###### "Data mining" course, Vrije Universiteit Amsterdam #################
###### Author: Elena Garcia Lara (student ID 2604289) ################################

# Import desired libraries

import os
import matplotlib
import pylab as plt
from operator import itemgetter
import numpy as np
from datetime import datetime, date
import time
import pandas as pd
import random
from random import randint
import sklearn
from data_exploration import file_to_pandas

from sklearn.cluster import KMeans
import sklearn.metrics as sm
from sklearn import preprocessing



def group_by_searchid(train):
    ''' This function calculates the search_id that are repeated'''
    # Worked
    print len(train['srch_id'].value_counts())
    print len(train['srch_id'])

    # Doesn't work
    # grouped_train = train.groupby('srch_id').count()
    # grouped_train = train.sort_values(by='srch_id', axis=0)
    # print grouped_train


def count(train):
    ''' This function calculates
    #bookings/totalsearch and #clicks/totalsearch'''
    booking = train.groupby('booking_bool').count()
    clicking = train.groupby('click_bool').count()

    # Values calculated when printing the two lines above
    global is_book, is_click, not_book, not_click, totalsearch
    is_book = booking['srch_id'][1]
    not_book = booking['srch_id'][0]
    is_click = clicking['srch_id'][1]
    not_click = clicking['srch_id'][0]
    totalsearch = is_book + not_book
    Yes = [is_book, is_click]
    Not = [not_book, not_click]

    ind = np.arange(2)    # the x locations for the groups
    width = 0.5      # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, Not, width, color='#F4561D')
    p2 = plt.bar(ind, Not, width, bottom=Yes, color='#F1BD1A')
    plt.ylabel('# instances')
    plt.title('Booking and clicking count')
    plt.xticks(ind, ('Booking', 'Click'))
    plt.legend((p1[0], p2[0]), ('Is', 'Is not'))

    # plt.show()
    # plt.savefig('results/book_click_count.png')

def downsample(train):
    # Select how many instances in the data you want
    # It can be is_book or is_click
    n = is_book

    # Select all rows with is_book = 1
    df_book = train.loc[train['booking_bool']==1]

    # Select n rows with is_book = 0
    train_not_book = train.loc[train['booking_bool']==0]
    df_notbook = train_not_book.sample(n)

    # Put rows together
    frames = [df_book, df_notbook]
    train_downsample = pd.concat(frames)

    return train_downsample

def main():
    # 1. Import dataset and transform to Pandas
            # (with NaN)
    train = file_to_pandas()

    # And group it by search_id
        # I just managed to count search_id, not group them
    # GroupedTrain = group_by_searchid(train)

    # 2.Count negative/positive data (booking/click)
    count(train)

    # 3. Downsample dataste to end with a 50-50 neg/pos
    new_train = downsample(train)



if __name__ == "__main__":
    main()
