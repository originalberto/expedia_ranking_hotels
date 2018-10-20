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
#from data_exploration_v2alb import file_to_pandas
import random



small_trainingset = 'data/selected_train.csv' #0.1%
medium_trainingset = 'data/mediumsize_selected_train.csv' #1%
trainingset = 'data/training_set.csv' #100%

def file_to_pandas(filename):
    ''' This function opens the dataset and converts it to Pandas'''
    train = pd.read_csv(small_trainingset)
    # train = pd.read_csv(medium_trainingset)
    #train = pd.read_csv(trainingset)

    train = pd.read_csv(filename)

    # print 'Train dataset shape:', train.shape
    return train

#  To still do: -Sample taking into account time! (or filter data after sampling)
            #   - create the test set from This

# Samples by srch_ID, and returns a dataframe in which 50% of the srch_ID have
# one of the rows with book_bool = 1 and 50% of the srch_ID have all their rows
# with book_bool = 0
def downsampling_v1(original_df):
    # print ((original_df['booking_bool'] == 1)).sum()
    # print 'unique search ID'
    # print len(original_df['srch_id'].unique())

    ## Look for the srch_IDs which ended up in a booking_bool=1 (in list srch_ID_booking)
    srch_ID_booking = original_df[original_df['booking_bool'] == 1].srch_id.tolist()
    # Create a new df storing the info from these srch_ID's
    new_df = original_df[original_df['srch_id'].isin(srch_ID_booking)]
    # remove the retrieved rows in "new_df" from the orignal dataframe
    original_df = original_df[~original_df['srch_id'].isin(srch_ID_booking)]
    # Sample the same number of different srch_id present in new_df from the original_df
    #     (in this case, the sampled users will have all book_bool = 0)
    non_sampled_srch_ID = original_df.srch_id.unique().tolist() # list of non-sampled srch_id's (yet)
    print len(non_sampled_srch_ID)
    print len(srch_ID_booking)
    to_be_sampled_srch_ID = random.sample(non_sampled_srch_ID, len(srch_ID_booking))
    new_df_2 = original_df[original_df['srch_id'].isin(to_be_sampled_srch_ID)]

    # Final dataframe: 50% users which booked, 50% user which didn't book
    final_df = pd.concat([new_df,new_df_2])
    final_df.to_csv('data/sampled_trainingset_v1.csv')

    return final_df

# Samples by row, and returns a dataframe in which 50% of the book_bool values = 1
# and 50% of the book_bool values = 0
## possible modiifcations: 33% book_bool & click_bool = 1, 33% book_bool = 0 & click_bool = 1; 33% book_bool & click_bool = 0
def downsampling_v2(original_df):
    ## Look for the rows which have book_bool = 1 and create a new df
    new_df = original_df[original_df['booking_bool'] == 1]
    number_book_bool = ((original_df['booking_bool'] == 1)).sum() # count how many rows were samples

    # remove the retrieved rows in "new_df" from the orignal dataframe
    # booking_bool = 0 AND random_bool = 1
    original_df = original_df[(original_df['booking_bool'] == 0) & (original_df['random_bool'] == 1)]


    # Sample the same number of rows matching "book_bool = 1" present in new_df from the original_df
    #     (in this case, the sampled rows will have all book_bool = 0)
    new_df_2 = original_df.sample(n=number_book_bool)


    # Final dataframe: 50% rows book_bool = 1, 50% rows book_bool = 0
    final_df = pd.concat([new_df,new_df_2])
    final_df.to_csv('data/processed_trainingset_downsampled.csv')
    return final_df

def main():
    train = file_to_pandas('data/processed_trainingset.csv')
    new_df = downsampling_v2(train)
    #print new_df



if __name__ == "__main__":
    main()
