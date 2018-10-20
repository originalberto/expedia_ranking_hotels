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
# from random import randint
import random
# import sklearn
import math
import csv

def file_to_pandas(datafile):
    ''' This function opens the dataset and converts it to Pandas'''
    train = pd.read_csv(datafile)
    # Change time information to year and month columns
    train["date_time"] = pd.to_datetime(train["date_time"])
    train["year"] = train["date_time"].dt.year
    train["month"] = train["date_time"].dt.month

    return train

def remove_outliers(df):
    ''' Remove hotels with price > 10.000 US$ and with less than 5 stars
    '''
    indices_to_remove = df[df['price_usd'] > 5000].index.tolist()
    df = df.drop(indices_to_remove)
    return df

def fill1(df):
    ''' Competitors 1-8'''
    ## com1-8_rate
    suma = df[['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate']].fillna(0).sum(axis=1)
    df['comp_rates'] = suma
    df = df.drop('comp1_rate', 1)
    df = df.drop('comp2_rate', 1)
    df = df.drop('comp3_rate', 1)
    df = df.drop('comp4_rate', 1)
    df = df.drop('comp5_rate', 1)
    df = df.drop('comp6_rate', 1)
    df = df.drop('comp7_rate', 1)
    df = df.drop('comp8_rate', 1)

    ## com1-8_inv
    suma = df[['comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv', 'comp7_inv', 'comp8_inv']].fillna(value=0).sum(axis=1)
    df['comp_invs'] = suma
    df = df.drop('comp1_inv', 1)
    df = df.drop('comp2_inv', 1)
    df = df.drop('comp3_inv', 1)
    df = df.drop('comp4_inv', 1)
    df = df.drop('comp5_inv', 1)
    df = df.drop('comp6_inv', 1)
    df = df.drop('comp7_inv', 1)
    df = df.drop('comp8_inv', 1)

    ## com1-8_rate_percent_diff
    median = df[['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 'comp7_rate_percent_diff', 'comp8_rate_percent_diff']].fillna(value=0).median(axis=1)
    df['comp_percent_diffs'] = median
    df = df.drop('comp1_rate_percent_diff', 1)
    df = df.drop('comp2_rate_percent_diff', 1)
    df = df.drop('comp3_rate_percent_diff', 1)
    df = df.drop('comp4_rate_percent_diff', 1)
    df = df.drop('comp5_rate_percent_diff', 1)
    df = df.drop('comp6_rate_percent_diff', 1)
    df = df.drop('comp7_rate_percent_diff', 1)
    df = df.drop('comp8_rate_percent_diff', 1)

    return df

def fill2(df,testset):
    '''srch_query_affinity_score'''
    # Replace the NaN by the 25% lowest number in that column of the dataset
    affinities = df['srch_query_affinity_score']
    total = len(affinities.dropna())
    if not testset:
        af = affinities.nsmallest(total/4).mean()
        f = open('data/data_preprocessing_values/af_srch_query_affinity_score.txt','w')
        f.write('%s'%(str(af)))
        f.close()
    if testset:
        f = open('data/data_preprocessing_values/af_srch_query_affinity_score.txt')
        af = f.readlines()[0].strip('\n')

    df['srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(af)
    return df

def fill3(df,testset):
    """prop_log_historical_price"""

    # Replace the NaN by the median of the hotels (depending on the number of stars of that hotel)
    if not testset:
        histprice_1star_median = df[(df['prop_log_historical_price'] != 0) & (df['prop_starrating'] == 1) ]['prop_log_historical_price'].median()
        histprice_2star_median = df[(df['prop_log_historical_price'] != 0) & (df['prop_starrating'] == 2) ]['prop_log_historical_price'].median()
        histprice_3star_median = df[(df['prop_log_historical_price'] != 0) & (df['prop_starrating'] == 3) ]['prop_log_historical_price'].median()
        histprice_4star_median = df[(df['prop_log_historical_price'] != 0) & (df['prop_starrating'] == 4) ]['prop_log_historical_price'].median()
        histprice_5star_median = df[(df['prop_log_historical_price'] != 0) & (df['prop_starrating'] == 5) ]['prop_log_historical_price'].median()

        f = open('data/data_preprocessing_values/prop_log_historical_price.txt','w')
        f.write('%s\n'%(str(histprice_1star_median)))
        f.write('%s\n'%(str(histprice_2star_median)))
        f.write('%s\n'%(str(histprice_3star_median)))
        f.write('%s\n'%(str(histprice_4star_median)))
        f.write('%s\n'%(str(histprice_5star_median)))
        f.close()
    if testset:
        f = open('data/data_preprocessing_values/prop_log_historical_price.txt')
        lines = f.readlines()
        histprice_1star_median = lines[0].strip('\n')
        histprice_2star_median = lines[1].strip('\n')
        histprice_3star_median = lines[2].strip('\n')
        histprice_4star_median = lines[3].strip('\n')
        histprice_5star_median = lines[4].strip('\n')
        f.close()

    # df.loc[df['prop_log_historical_price'] == 0, 'prop_log_historical_price'] = np.nan


    df.loc[(df['prop_starrating'] == 0), 'prop_log_historical_price'] = df.loc[(df['prop_starrating'] == 0), 'prop_log_historical_price'].fillna(histprice_1star_median)
    df.loc[(df['prop_starrating'] == 1), 'prop_log_historical_price'] = df.loc[(df['prop_starrating'] == 1), 'prop_log_historical_price'].fillna(histprice_1star_median)
    df.loc[(df['prop_starrating'] == 2), 'prop_log_historical_price'] = df.loc[(df['prop_starrating'] == 2), 'prop_log_historical_price'].fillna(histprice_2star_median)
    df.loc[(df['prop_starrating'] == 3), 'prop_log_historical_price'] = df.loc[(df['prop_starrating'] == 3), 'prop_log_historical_price'].fillna(histprice_3star_median)
    df.loc[(df['prop_starrating'] == 4), 'prop_log_historical_price'] = df.loc[(df['prop_starrating'] == 4), 'prop_log_historical_price'].fillna(histprice_4star_median)
    df.loc[(df['prop_starrating'] == 5), 'prop_log_historical_price'] = df.loc[(df['prop_starrating'] == 5), 'prop_log_historical_price'].fillna(histprice_5star_median)

    ### Old code in folder 1.Exploration

    return df

def fill4(df,testset):
    """visitor_hist_starrating"""
    ''' substitute by the actual price/rating that the person has spend
     on the booking, or if it has not made any booking, substitute by
     the average value in the dataset?
     df[df['booking_bool'] == 1]'''
    if not testset:
        median_starrating = df['visitor_hist_starrating'].median()
        median_hist_adr_usd = df['visitor_hist_adr_usd'].median()
        f = open('data/data_preprocessing_values/median_starrating.txt','w')
        f.write('%s\n'%(str(median_starrating)))
        f.write('%s\n'%(str(median_hist_adr_usd)))
        f.close()
    if testset:
        f = open('data/data_preprocessing_values/median_starrating.txt')
        lines = f.readlines()
        median_starrating = lines[0].strip('\n')
        median_hist_adr_usd = lines[1].strip('\n')


    df.loc[df['visitor_hist_starrating'] == 0] = np.nan
    if not testset:
        index_users = df.loc[df['booking_bool']==1].srch_id.tolist()
        # The users that have a booking will get this value
        for user in index_users:
            value1 = df.loc[(df['srch_id']==user)].prop_starrating.mean()
            df.loc[(df['srch_id']==user), 'visitor_hist_starrating'] = df.loc[(df['srch_id']==user), 'visitor_hist_starrating'].fillna(value1)

        # The users not in index_users will be still NaNs
        #They will get a common value
        df['visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(median_starrating)
    if testset:
        df['visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(median_starrating)

    """visitor_hist_adr_usd"""
    if not testset:
        index_users = df.loc[df['booking_bool']==1].srch_id.tolist()

        # The users that have a booking will get this value
        for user in index_users:
            value = df.loc[(df['srch_id']==user)].gross_bookings_usd.mean()
            df.loc[(df['srch_id']==user), 'visitor_hist_adr_usd'] = df.loc[(df['srch_id']==user), 'visitor_hist_adr_usd'].fillna(value)

        # The users not in index_users will be still NaNs
        #They will get a common value
        df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(median_hist_adr_usd)
    if testset:
        df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(median_hist_adr_usd)

    return df

def fill5(df,testset):
    """orig_destination_distance"""
    ## Old code available in filling_mp_old4.py

    ''' Fill the missing values by the median of the searches matching the
    same visitor_location_country_id '''
    # print len(df['orig_destination_distance'].isnull())
    no_ordest_distance = df.loc[df['orig_destination_distance'].isnull()]
    sum_list = []
    list_countries= []
    for visitorid in df['visitor_location_country_id'].unique():
        if not testset:
            median_tofill = df[df['visitor_location_country_id'] == visitorid]['orig_destination_distance'].median()
            if not math.isnan(median_tofill):
                sum_list.append(median_tofill)
    for visitorid in df['visitor_location_country_id'].unique():
        if not testset:
            median_tofill = df[df['visitor_location_country_id'] == visitorid]['orig_destination_distance'].median()
            if not math.isnan(median_tofill):
                list_countries.append(visitorid)
                #sum_list.append(median_tofill)
                f = open('data/data_preprocessing_values/orig_destination_distance_%s.txt'%(str(visitorid)),'w')
                f.write('%s\n'%(str(median_tofill)))
                f.close()
            else:
                # Calculate the mean of the list of medians
                # median_tofill = sum(sum_list)/len(sum_list)
                median_tofill = np.median(sum_list)
                f = open('data/data_preprocessing_values/orig_destination_distance_mean.txt','w')
                f.write(str(median_tofill))
                f.close()
            # Write down the list of countries with individual orig_destination_distance
            f = open('data/data_preprocessing_values/orig_destination_distance_list_countries.txt','w')
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(list_countries)
            f.close()

        if testset:
            # Check if the contry has individual or 'pooled' value
            try:
                f = open('data/data_preprocessing_values/orig_destination_distance_%s_.txt'%(str(visitorid)))
            except:
                f = open('data/data_preprocessing_values/orig_destination_distance_mean.txt')
            median_tofill = f.readlines()[0].strip('\n')
        no_ordest_distance_selected = no_ordest_distance.loc[no_ordest_distance['visitor_location_country_id'] == visitorid]
        # for index in no_ordest_distance_selected.index.tolist():
        #     #print median_tofill
            # df.ix[index,'orig_destination_distance'] = median_tofill
        df.loc[(df['visitor_location_country_id'] == visitorid), 'orig_destination_distance'] = df.loc[(df['visitor_location_country_id'] == visitorid), 'orig_destination_distance'].fillna(median_tofill)


    return df

def fill6(df,testset=False):
    """ prop_review_score; prop_location_score1; prop_location_score2; """
    new = df['prop_review_score'].fillna(0)
    df['prop_review_score'] = new

    new = df['prop_location_score1'].fillna(0)
    df['prop_location_score1'] = new

    new = df['prop_location_score2'].fillna(0)
    df['prop_location_score2'] = new

    return df

def calling_functions(dataset, is_test):
    ## 1. Import dataset and transform to Pandas
    print 'Please wait...'
    train = file_to_pandas(dataset)
    # train = remove_outliers(train)

    #Delete column date-time
    # train = train.drop(['date_time', 'Unnamed: 0'], 1)
    train = train.drop(['date_time'], 1)


    ## A series of functions to fill the missing points
    ## is_test=True/false
    print 1
    train = fill1(train)
    print 2
    train = fill2(train,is_test)
    print 3
    train = fill3(train,is_test)
    print 4
    train = fill4(train,is_test)
    print 5
    train = fill5(train,is_test)
    print 6
    train = fill6(train,is_test)

    #Delete column date-time
    # train = train.drop('gross_bookings_usd', 1)

    # Check if there is any missing value
    print 'Missing values of set:', len(train.isnull())

    print train.isnull().sum()
    print train.shape

    # Convert to csv
    train.to_csv('data/processed_assignment_testset3.csv')
    # if not is_test:    ## Train set
    #     train.to_csv('data/processed_trainingset.csv')
    # else:    ## Test set
    #     train.to_csv('data/processed_testset.csv')

def main():
    # trainingset = 'data/unprocessed_trainingset.csv'
    # testset = 'data/unprocessed_testset.csv'
    train = 'data/assignment_test_set.csv'
    print train.isnull().sum()
    train = 'data/assignment_test_set.csv'
    print train.isnull().sum()


    # Both training and test set
    # is_test = False
    # for dataset in [trainingset, testset]:
    #     calling_functions(dataset, is_test)
    #     is_test = True

    # Only test set
    is_test = True
    calling_functions(testset, is_test)


if __name__ == "__main__":
    main()
