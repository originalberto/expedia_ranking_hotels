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

small_trainingset = 'data/selected_train.csv' #0.1%
medium_trainingset = 'data/train_medium_noNA.csv' #1%
big_trainingset = 'data/complete_train_touse.csv' #100%
onepercent = 'data/onepercent_noNA.csv' #1%
trainingset = 'data/training_set.csv' #100% & original

def file_to_pandas():
    ''' This function opens the dataset and converts it to Pandas'''
    # train = pd.read_csv(small_trainingset)
    # train = pd.read_csv(medium_trainingset)
    train = pd.read_csv(onepercent)
    # train = pd.read_csv(trainingset)

    # print 'Train dataset shape:', train.shape
    return train

def user_minus_average(df):
    ''' Create new variables in pdataframe
    '''
    # 1. starrating_diff = |visitor_hist_starrating - prop_starrating|
    ########### BETTER FILLING NAs ##############
    df['visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(0)
    df['prop_starrating'] = df['prop_starrating'].fillna(0)

    df['starrating_diff'] = abs(df['visitor_hist_starrating'] - df['prop_starrating'])

    # 2. usd_diff = |visitor_hist_adr_usd - price_usd|
    ########### BETTER FILLING NAs ##############
    df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(0)
    df['price_usd'] = df['price_usd'].fillna(0)

    df['usd_diff'] = abs(df['visitor_hist_adr_usd'] - df['price_usd'])

    # 3. prop_starating_monotonic = |prop_starrating - mean(prop_starrating[booking_bool])|
    ########### BETTER FILLING NAs ##############
    df['prop_starrating'] = df['prop_starrating'].fillna(0)

    df_bool0 = df.loc[df['booking_bool'] == 0]
    mean_bool0 = df_bool0['prop_starrating'].mean()
    df_bool1 = df.loc[df['booking_bool'] == 1]
    mean_bool1 = df_bool1['prop_starrating'].mean()

    df.ix[df.booking_bool == 0, 'prop_starating_monotonic'] = abs(df['prop_starrating'] - mean_bool0)
    df.ix[df.booking_bool == 1, 'prop_starating_monotonic'] = abs(df['prop_starrating'] - mean_bool1)

    return df

def calc_norm(df, indicator, new_col, old_col):
    # indicator -- the reference for the normalization (e.g. srch_id, prop_id)
    # new_col -- the name of the new column to hold normalized values
    # old_col -- column that holds the values to be normalized
    # new_col and old_col can be the same, but it will overwrite.

    if indicator == 'srch_id':
        for item in df.srch_id.unique():
            df.ix[df.srch_id == item, new_col] = (df[old_col] - df[old_col].loc[df.srch_id == item].mean()) / (df[old_col].loc[df.srch_id == item].max() - df[old_col].loc[df.srch_id == item].min())

    if indicator == 'prop_id':
        for item in df.prop_id.unique():
            df.ix[df.prop_id == item, new_col] = (df[old_col] - df[old_col].loc[df.prop_id == item].mean()) / (df[old_col].loc[df.prop_id == item].max() - df[old_col].loc[df.prop_id == item].min())
    if indicator == 'srch_destination_id':
        for item in df.srch_destination_id.unique():
            df.ix[df.srch_destination_id == item, new_col] = (df[old_col] - df[old_col].loc[df.srch_destination_id == item].mean()) / (df[old_col].loc[df.srch_destination_id == item].max() - df[old_col].loc[df.srch_destination_id == item].min())
    if indicator == 'prop_country_id_id':
        for item in df.prop_country_id_id.unique():
            df.ix[df.prop_country_id_id == item, new_col] = (df[old_col] - df[old_col].loc[df.prop_country_id_id == item].mean()) / (df[old_col].loc[df.prop_country_id_id == item].max() - df[old_col].loc[df.prop_country_id_id == item].min())
    if indicator == 'month':
        for item in df.month.unique():
            df.ix[df.month == item, new_col] = (df[old_col] - df[old_col].loc[df.month == item].mean()) / (df[old_col].loc[df.month == item].max() - df[old_col].loc[df.month == item].min())
    if indicator == 'srch_booking_window':
        for item in df.srch_booking_window.unique():
            df.ix[df.srch_booking_window == item, new_col] = (df[old_col] - df[old_col].loc[df.srch_booking_window == item].mean()) / (df[old_col].loc[df.srch_booking_window == item].max() - df[old_col].loc[df.srch_booking_window == item].min())

    return df

def normalization(df):
    ''' Choose the columns to be normalized
    '''
    # Normalize log10(price_usd) with respect to serch_id
    df['log10price_usd'] = df['price_usd'].apply(np.log10)
    df = calc_norm(df, 'srch_id', 'price_norm', 'log10price_usd')

    # comp1_rate, comp1_rate_percent_diff, comp_inv wrt ...srch_destination_id, month, prop_id ???
    df = calc_norm(df, 'srch_destination_id', 'comp_rate_norm', 'comp_rate')

    # prop_review_score wrt prop_id
    df = calc_norm(df, 'prop_id', 'score_norm', 'prop_review_score')

    # prop_log_historical_price wrt prop_id, or prop_country_id ??, month??
    df = calc_norm(df, 'prop_country_id', 'hist_price_norm', 'prop_log_historical_price')

    # srch_length_of_stay wrt srch_id
    df = calc_norm(df, 'srch_id', 'length_norm', 'srch_length_of_stay')

    # srch_booking_window wrt srch_id, month??
    df = calc_norm(df, 'srch_id', 'window_norm', 'srch_booking_window')

    # srch_query_affinity_score --- be careful with extreme values. normalize wrt prop_id
    df = calc_norm(df, 'prop_id', 'affinity_norm', 'srch_query_affinity_score')

def month_booked(train):
    train['month_booked'] = train['month'] + (train['srch_booking_window']/30).astype(int)
    train['month_booked'] = train['month_booked'].replace([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    return train

def aggregate(train):
    train_srchID = train.groupby(by='month')
    train_srchID.to_csv('train_onepercent_aggSearchID.csv')
    train_propID.to_csv('train_onepercent_aggPropID.csv')




def main():
    train = file_to_pandas()

    ## 1.Create new variables for ratings, prices, etc.
    # train = user_minus_average(train)

    ## 2. Normalize log10(price_usd) w.r.t. srch_id
    # normalization(train)

    ## 3. Month_booked : (month_booking + srch_booking_window/30)
    train = month_booked(train)

    ## 4. Aggregate data frames by search_id and prop_id
    # ---- not finished, looking for a way to aggregate them
    # ---- in the tutorials they only do it by mean()... but
    # ---- i just want a list or sth
    # aggregate(train)
    print train








if __name__ == "__main__":
    main()
