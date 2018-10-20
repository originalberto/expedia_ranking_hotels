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



def file_to_pandas(dataset):
    ''' This function opens the dataset and converts it to Pandas'''
    train = pd.read_csv(dataset)

    return train

def user_minus_average(df):
    ''' Create new variables in pdataframe
    '''
    # 1. starrating_diff = |visitor_hist_starrating - prop_starrating|
    df['starrating_diff'] = abs(df['visitor_hist_starrating'] - df['prop_starrating'])

    # 2. usd_diff = |visitor_hist_adr_usd - price_usd|
    df['usd_diff'] = abs(df['visitor_hist_adr_usd'] - df['price_usd'])

    # 3. prop_starating_monotonic = |prop_starrating - mean(prop_starrating[booking_bool])|
    # df_bool0 = df.loc[df['booking_bool'] == 0]
    # mean_bool0 = df_bool0['prop_starrating'].mean()
    # df_bool1 = df.loc[df['booking_bool'] == 1]
    # mean_bool1 = df_bool1['prop_starrating'].mean()

    # For test set, values calculated before with testset (created by us, not the assignment one)
    mean_bool0 = 3.14163898475
    mean_bool1 = 3.26851311953

    # df.ix[df.booking_bool == 0, 'prop_starating_monotonic'] = abs(df['prop_starrating'] - mean_bool0)
    # df.ix[df.booking_bool == 1, 'prop_starating_monotonic'] = abs(df['prop_starrating'] - mean_bool1)
    df['mean_bool'] = mean_bool1
    df['prop_starating_monotonic'] = abs(df['prop_starrating'] - df['mean_bool'])
    df = df.drop('mean_bool', 1)


    return df



def month_booked(train):
    train['month_booked'] = train['month'] + (train['srch_booking_window']/30).astype(int)
    train['month_booked'] = train['month_booked'].replace([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    return train

def season_booked(train):
    winter = [1, 2, 12]
    spring = [3, 4, 5]
    summer = [6, 7, 8]
    autumn = [9, 10, 11]

    train['season_booked'] = np.nan

    train.loc[train['month_booked'].isin(winter), 'season_booked'] = 1
    train.loc[train['month_booked'].isin(spring), 'season_booked'] = 2
    train.loc[train['month_booked'].isin(summer), 'season_booked'] = 3
    train.loc[train['month_booked'].isin(autumn), 'season_booked'] = 4

    return train

def merge_prop_scores(train):
    train.loc[train['prop_starrating']==0, 'prop_starrating'] = 0.01
    train.loc[train['prop_review_score']==0, 'prop_review_score'] = 0.01

    train['scores_merged'] = train['prop_starrating'] * train['prop_review_score']

    train.loc[train['prop_starrating']==0.01, 'prop_starrating'] = 0
    train.loc[train['prop_review_score']==0.01, 'prop_review_score'] = 0

    return train

def advertise_vs_real(train):
    train['ad_vs_real'] = train['prop_starrating'] - train['prop_review_score']
    return train


def probability_hotel(train, testset):
    ''' Fill the missing values by the probablity of the hotel,
    or 0 if not seen before. '''
    if not testset:
        ## Count_hotel
        train['count_hotel'] = train.groupby('prop_id')['prop_id'].transform('count')

        ## Prob_book
        train['prob_book'] = np.nan
        ## count how many times each hotel is booked
        for hotel in train.prop_id.unique().tolist():
            list_booking = train.loc[(train['prop_id']==hotel)]['booking_bool'].value_counts().tolist()
            # Some hotels are never booked, then the list is only [6], e.g. it appears 6 times but never booked
            if len(list_booking) > 1:
                booked_hotels = list_booking[1]
                total_times = sum(list_booking)
                prob_booked = booked_hotels / float(total_times)
                train.loc[(train['prop_id']==hotel), 'prob_book'] = prob_booked
        train['prob_book'] = train['prob_book'].fillna(0)

        ## Prob_click
        train['prob_click'] = np.nan
        ## count how many times each hotel is booked
        for hotel in train.prop_id.unique().tolist():
            list_clicking = train.loc[(train['prop_id']==hotel)]['click_bool'].value_counts().tolist()
            # Some hotels are never clicked, then the list is only [6], e.g. it appears 6 times but never clicked
            if len(list_clicking) > 1:
                clicked_hotels = list_clicking[1]
                total_time = sum(list_clicking)
                prob_clicked = clicked_hotels / float(total_time)
                train.loc[(train['prop_id']==hotel), 'prob_click'] = prob_clicked
        train['prob_click'] = train['prob_click'].fillna(0)


        for hotel in train['prop_id'].unique():
            #Hotel_count
            hotel_count = train[train['prop_id'] == hotel]['count_hotel']
            #Prob_booked
            prob_b = train[train['prop_id'] == hotel]['prob_book']
            #Prob_click
            prob_c = train[train['prop_id'] == hotel]['prob_click']

            list_ = [hotel_count, prob_b, prob_c]

            f = open('data/data_preprocessing_values/count_hotel_%s.txt'%(str(hotel)),'w')
            f.write('%s\n'%(str(list_)))
            f.close()

            # f = open('data/data_preprocessing_values/prob_book_%s.txt'%(str(hotel)),'w')
            # f.write('%s\n'%(str(prob_b)))
            # f.close()
            #
            # f = open('data/data_preprocessing_values/prob_click_%s.txt'%(str(hotel)),'w')
            # f.write('%s\n'%(str(prob_c)))
            # f.close()


    if testset:
        for hotel in train['prop_id'].unique():
            if testset:
                try:
                    f = open('data/data_preprocessing_values/orig_destination_distance_%s_.txt'%(str(visitorid)))
                    line = f.readlines()[0].strip('\n').strip('[').strip(']')
                    line = line.split(',')
                    count_h = line[0]
                    book_prb = line[1]
                    click_prb = line[2]
                    median_tofill = f.readlines()[0].strip('\n')
                except:
                    count_h = 0
                    book_prb = 0
                    click_prb = 0

                train.loc[(train['prop_id'] == hotel), 'count_hotel'] = count_h
                train.loc[(train['prop_id'] == hotel), 'prob_book'] = book_prb
                train.loc[(train['prop_id'] == hotel), 'prob_click'] = click_prb
                # Check if the contry has individual or 'pooled' value



            # for visitorid in df['visitor_location_country_id'].unique():
            # else:
            #     # Calculate the mean of the list of medians
            #     # median_tofill = sum(sum_list)/len(sum_list)
            #     median_tofill = np.median(sum_list)
            #     f = open('data/data_preprocessing_values/orig_destination_distance_mean.txt','w')
            #     f.write(str(median_tofill))
            #     f.close()
            # # Write down the list of countries with individual orig_destination_distance
            # f = open('data/data_preprocessing_values/orig_destination_distance_list_countries.txt','w')
            # wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            # wr.writerow(list_countries)
            # f.close()


    return train



def boolean_countries(train):
    # 1 if the countries are the same
    train['bool_same_country'] = 0
    train.loc[(train['visitor_location_country_id'] == train['prop_country_id']), 'bool_same_country'] = 1
    return train

def price_individual(train):
    # Individual price per night
    # price_for_one = price_usd / [(srch_adults_count + srch_children_count) * srch_length_of_stay]
    train['price_for_one'] = train['price_usd'] / ((train['srch_adults_count'] + train['srch_children_count']) * train['srch_length_of_stay'])
    return train

def calculate_ranks(train):
    # Calculate the rank in a query of price and star rating
    train['rank_price'] = 1
    train['rank_scores'] = 1
    # for search in train.srch_id.unique().tolist():
    #     train.loc[train['srch_id']==search, 'rank_price'] = train.loc[train['srch_id']==search, 'price_usd'].rank()
    #     train.loc[train['srch_id']==search, 'rank_scores'] = train.loc[train['srch_id']==search, 'prop_starrating'].rank()

    return train

def composed_features(train):
    #Compose features roomcount_bookwindow, adultcount_childrencount by F1_F2 = F1*max(Max(F2)) + F2

    train['roomcount_bookwindow'] = train['srch_room_count']*max(train['srch_booking_window']) + train['srch_booking_window']
    train['adultcount_childrencount'] = train['srch_adults_count']*max(train['srch_children_count']) + train['srch_children_count']

    return train

def calling_functions(dataset, is_test):
    print 'Please wait...'

    train = file_to_pandas(dataset)

    ## 1.Create new variables for ratings, prices, etc.
    print 1
    # train = user_minus_average(train)
    # train = train.fillna(0)

    ## 3. Month_booked : (month_booking + srch_booking_window/30)
    print 2
    # train = month_booked(train)
    # train = season_booked(train)

    ## 4. Merge prop_starrating and prop_review_score
    print 3
    # train = merge_prop_scores(train)

    ## 5.Prop_starrating - prop_review_score (difference advertised/real)
    print 4
    # train = advertise_vs_real(train)

    ## 6. probability hotel X is booked or clicked
    print 5
    train = probability_hotel(train, is_test)
    print train

    ## 7. Boolean visitor_location_country_id == prop_country_id
    print 6
    # train = boolean_countries(train)

    ## 8. price by person and by night
    print 7
    # train = price_individual(train)

    ## 9. Calculate the rank in a query of price and rating (scores_merged)
    print 8
    # train = calculate_ranks(train)

    ## 10. Suggested features
    print 9
    # train = composed_features(train)

    #
    # ## Save the file to a csv file
    # if number == 0:
    #     train.to_csv('data/training_final2.csv')
    # elif number == 1:
    #     train.to_csv('data/test_final_notdownsampled.csv')
    # elif number == 2:
    #     train.to_csv('data/training_final1.csv')
    # print train.isnull().sum()
    # print train.shape
    # train.to_csv('data/assignment_final_testset_semi3.csv')
    train.to_csv('data/assignment_finalfinal_testset6.csv')


def main():
    # # Both training and test set
    # trainingset1 = 'data/downsampled_trainingset2.csv'
    # trainingset2 = 'data/downsampled_trainingset3.csv'
    # testset = 'data/processed_testset.csv'


    ### TRAINING SET
    # dataset = 'data/downsampled_trainingset2.csv'

    ### TEST SET
    dataset = 'data/assignment_finalfinal_testset5.csv'

    # number = 0
    # for dataset in [trainingset2, testset, trainingset1]:
    #     calling_functions(dataset, number)
    #     number +=1
    calling_functions(dataset, True)

    print 'Done!'


if __name__ == "__main__":
    main()
