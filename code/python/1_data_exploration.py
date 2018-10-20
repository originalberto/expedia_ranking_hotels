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

small_trainingset = 'data/selected_train.csv' #0.1%
medium_trainingset = 'data/mediumsize_selected_train.csv' #1%
big_trainingset = 'data/complete_train_touse.csv' #100%
trainingset = 'data/training_set.csv' #100% & original

def file_to_pandas():
    ''' This function opens the dataset and converts it to Pandas'''
    # train = pd.read_csv(small_trainingset)
    # train = pd.read_csv(medium_trainingset)
    train = pd.read_csv(trainingset)

    # print 'Train dataset shape:', train.shape
    return train

def dataset_reduction(train):
    ''' This function selects 1/10 of the raw dataset,
    takes care (and graphs) the distribution of target hotel and time.
    '''
    # Change time information to year and month columns
    train["date_time"] = pd.to_datetime(train["date_time"])
    train["year"] = train["date_time"].dt.year
    train["month"] = train["date_time"].dt.month

    #Delete column date-time
    train = train.drop('date_time', 1)

    ## Calculate the distributions before 10% selection
    # distributions(train)

    ## Print the number of rows that each month has
    # for month in range(1,7):
    #     print (train[(train['year']==2013 )& (train['month']==month)]).shape
    # exit()

    # Select randomly 10% of the dataset
    unique_users = train.srch_id.unique()
    total_size = 4958347
    n_selected = total_size/100
    # We can select more later on, I selected little data for the beginning
    sel_user_id = train.sample(n_selected)

    # Calculate the distributions after 10% selection
    # distributions(sel_user_id)
    print 'Done'
    sel_user_id.to_csv('onepercent_train.csv')


def distributions(train):
    ''' Graph the distributions of country and time
    Used 1- before, 2- after the 10% selection of the data
    (called from dataset_reduction function)'''

    # Get dictionary for country
    list_country = []
    list_countryvalues = []
    for country in train.prop_country_id.unique():
        rows = train.loc[train['prop_country_id'] == country]
        list_country.append(country)
        list_countryvalues.append(len(rows))

    # Get dictionary for time (month, year)
    list_year = []
    list_yearvalues = []
    for year in [2012, 2013]: # there's only these 2 years in the data
        for months in train.month.unique():
            rows = train.loc[train['month'] == months]
            realrows = rows.loc[train['year']==year]
            list_year.append(str(year)+'-'+ str(months))
            list_yearvalues.append(len(realrows))

    # Pie chart of distributions of country
    labels = list_country
    sizes = list_countryvalues
    # colors = ...
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()


    # Pie chart of distributions of country
    labels = list_year
    sizes = list_yearvalues
    # colors = ...
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()

def missing_points(df):
    ''' Calculate and plot the missing points '''
    ''' 1. missing points per variable'''

    # Calculate missing points
    mp_variable = []
    points_var = df.count(axis=0).tolist()
    for item in points_var:
        mp = 4958 - item
        mp_variable.append(mp)

    # Plot
    fig, ax = plt.subplots()
    rects1 = ax.bar(range(0,len(list(df))), mp_variable)
    ax.set_xticklabels(list(df), rotation=45)
    # Theres a problem with labels...

    # This line represents the maximum datapoints
    plt.plot(range(0,len(list(df))), [4958]*56)

    plt.xlim([0, 55])
    plt.show()

    ''' 2. missing points per search_id (grouped)'''
    # Calculate missing points
    mp_search = []
    points_s = df.count(axis=1).tolist()
    for item in points_s:
        mp = 56 - item
        mp_search.append(mp)

    # Group search_id in 10 groups (% of missing points)
    total = 56
    suma=0
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for item in mp_search:
        if item < total/10: p1+=item
        elif item>total/10 and item<2*total/10: p2+=1
        elif item>2*total/10 and item<3*total/10: p3+=1
        elif item>3*total/10 and item<4*total/10: p4+=1
        elif item>4*total/10 and item<5*total/10: p5+=1
        elif item>5*total/10 and item<6*total/10: p6+=1
        elif item>6*total/10 and item<7*total/10: p7+=1
        elif item>7*total/10 and item<8*total/10: p8+=1
        elif item>8*total/10 and item<9*total/10: p8+=1
        elif item>9*total/10: p10+=1
        suma+=1
    mp_grouped = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
    labels = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']

    # Plot
    fig, ax = plt.subplots()
    # plt.plot(range(0,10), [suma/10]*10) Line that represents equal distr
    plt.bar(range(0,10), mp_grouped)
    plt.xlim([0,9])
    ax.set_xticklabels(labels)
    plt.show()

def plot_variables(df,var1,var2):
    plt.figure()
    plt.scatter(df[var1],df[var2])
    plt.xlabel('%s'%(var1)), plt.ylabel('%s'%(var2))
    plt.tight_layout()
    plt.savefig('results/correlation_%s_%s.png'%(var1,var2))
    plt.tight_layout()
    plt.show()

def remove_outliers(df):
    ''' Remove hotels with price > 10.000 US$ and with less than 5 stars
    NOT BASED IN ANYTHING
    But maybe it is useful for other removal
    '''
    indices_to_remove = df[(df['prop_starrating'] < 5) & (df['price_usd'] > 10000)].index.tolist()
    df = df.drop(indices_to_remove)
    return df

def study_effect_random(df):
    '''Study the effect of the random sorting in Expedia
        (whether this can influence) the final decision of final booking'''
    rsortings = len(df[df['random_bool'] == 1].index.tolist())
    print len(df[df['random_bool'] == 1].index.tolist()), 'random sortings'
    rsortings_click = len(df[(df['random_bool'] == 1) & (df['click_bool'] == 1)].index.tolist())
    rsortings_book = len(df[(df['random_bool'] == 1) & (df['booking_bool'] == 1)].index.tolist())
    print 'Random sort + click', len(df[(df['random_bool'] == 1) & (df['click_bool'] == 1)].index.tolist())
    print 'Random sort + booking', len(df[(df['random_bool'] == 1) & (df['booking_bool'] == 1)].index.tolist())
    normalsortings = len(df[df['random_bool'] == 0].index.tolist())
    print len(df[df['random_bool'] == 0].index.tolist()), 'normal sortings'
    normalsortings_click = len(df[(df['random_bool'] == 0) & (df['click_bool'] == 1)].index.tolist())
    normalsortings_book = len(df[(df['random_bool'] == 0) & (df['booking_bool'] == 1)].index.tolist())
    print 'Normal sort + click', len(df[(df['random_bool'] == 0) & (df['click_bool'] == 1)].index.tolist())
    print 'Normal sort + booking', len(df[(df['random_bool'] == 0) & (df['booking_bool'] == 1)].index.tolist())
    clicks_ratio = [float(rsortings_click)/float(rsortings),float(normalsortings_click)/float(normalsortings)]
    bookings_ratio = [float(rsortings_book)/float(rsortings),float(normalsortings_book)/float(normalsortings)]
    variables = ['Random sorting', 'Normal sorting']
    # Plot the results
    ind = np.arange(2)  # the x locations for the groups
    width = 0.4
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, clicks_ratio, width, color='r',label='Clicks')
    rects2 = ax.bar(ind + width, bookings_ratio, width, color='y',label='Bookings')
    ax.set_xticklabels(('random_bool=1','random_bool=0'))
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Click and booking ratio')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('random_bool = 1', 'random_bool = 0'))
    ax.legend((rects1[0], rects2[0]), ('Clicks', 'Bookings'))
    plt.ylabel('Click and booking ratio')
    plt.legend(prop={'size':6})
    plt.tight_layout()
    plt.savefig('results/study_effect_random.png')
    # Save the obtained ratios in a file
    f = open('results/random_sorting_studio.txt','w')
    f.write('###1st line: clicks_ratio; 2nd line: bookings ratio; 1st element 1st line:: Clicks; 2nd element 2nd line: Bookings\n')
    for element in clicks_ratio:
        f.write('%s\t'%(str(element)))
    f.write('\n')
    for element in bookings_ratio:
        f.write('%s\t'%(str(element)))

def histogram_rating_hotels(df,nstars):
    plt.hist(df[df['prop_starrating'] == nstars]['prop_review_score'].dropna(),5)
    plt.tight_layout()
    plt.title('Histogram of reviews from %s star hotel'%(str(nstars)))
    plt.xlabel('Review value (prop_review_score)'), plt.ylabel('Frequency')
    plt.savefig('results/histogram_%sstarshotel.png'%(str(nstars)))
    plt.tight_layout()
    #plt.show()

def histogram_position_vs_booking(df):
    ''' Plots #booking vs position
    1) When position is not random (random_bool=0)
    '''
    df0 = df.loc[df['random_bool'] == 0]
    pos_list = []
    bookings_no = []
    click_no = []
    for pos in range(50):
        pos_list.append(pos)
        bookings_no.append(df0[df0['position']==pos]['booking_bool'].tolist().count(1))
        click_no.append(df0[df0['position']==pos]['click_bool'].tolist().count(1))
    fig, ax = plt.subplots()
    width = 0.35
    ax.bar(pos_list,bookings_no,width,color='r',label='booking_bool')
    pos_list = [x+width for x in pos_list]
    ax.bar(pos_list,click_no,width,label='click_bool')
    plt.xlabel('Position'), plt.ylabel('Counts')
    plt.legend()
    plt.title('Book/Click vs position (not random)')
    plt.tight_layout()
    plt.savefig('results/hotel_position_vs_click_booking_notrandom.png')
    # plt.show()
    plt.close()

    ''' 2) When position is random (random_bool=1)
    '''
    df1 = df.loc[df['random_bool'] == 1]
    pos_list = []
    bookings_no = []
    click_no = []
    for pos in range(50):
        pos_list.append(pos)
        bookings_no.append(df1[df1['position']==pos]['booking_bool'].tolist().count(1))
        click_no.append(df1[df1['position']==pos]['click_bool'].tolist().count(1))
    fig, ax = plt.subplots()
    width = 0.35
    ax.bar(pos_list,bookings_no,width,color='r',label='booking_bool')
    pos_list = [x+width for x in pos_list]
    ax.bar(pos_list,click_no,width,label='click_bool')
    plt.xlabel('Position'), plt.ylabel('Counts')
    plt.legend()
    plt.title('Book/Click vs position (random)')
    plt.tight_layout()
    plt.savefig('results/hotel_position_vs_click_booking_random.png')
    # plt.show()
    plt.close()

    print 'Figures created'

def histogram_all(df):
    ''' Plots and saves a histogram (ordered by frequency)
    of all the variables in the dataframe'''

    # For loop to study all variables
    df_comp_rate = None
    df_comp_inv= None
    df_comp_diff = None
    for var in list(df)[2:]:
        # For most of the variables
        if var[0:4] != 'comp':
            plt.figure()
            plt.hist(df[var].dropna()) #, bins=50)
            plt.title(var)
            plt.xlabel('Value'), plt.ylabel('Frequency')
            plt.savefig('results/hist/hist_'+var+'.png')
            plt.close()

        # For the variables about the competitors, join comp1-comp8
        elif var[len(var)-4:len(var)] == 'rate':
            frames = [df_comp_rate, df[var]]
            df_comp_rate = pd.concat(frames)

        elif var[len(var)-4:len(var)] == '_inv':
            frames = [df_comp_inv, df[var]]
            df_comp_inv = pd.concat(frames)

        elif var[len(var)-4:len(var)] == 'diff':
            frames = [df_comp_diff, df[var]]
            df_comp_diff = pd.concat(frames)

    yrate = df_comp_rate.value_counts().tolist()
    yinv = df_comp_inv.value_counts().tolist()
    ydiff = df_comp_diff.value_counts()

    fig, ax = plt.subplots()
    width = 0.35
    ax.bar([1, 0, -1],yrate,width)
    plt.xlabel('Value'), plt.ylabel('Counts')
    plt.title('Expedia prices vs comp1-8 (1=lower, 0=equal)')
    plt.savefig('results/hist/comp1-8_rate.png')
    plt.close()

    fig, ax = plt.subplots()
    width = 0.35
    ax.bar([1, 0, -1],yinv,width)
    # WHY are there -1 here ???
    plt.xlabel('Value'), plt.ylabel('Counts')
    plt.title('comp1-8 availability')
    plt.savefig('results/hist/comp1-8_inv.png')
    plt.close()

    plt.hist(ydiff)
    plt.title('comp1-8 price difference')
    plt.xlabel('Value'), plt.ylabel('Frequency')
    plt.savefig('results/hist/comp1-8_pdiff.png')
    plt.close()

    # Finally, ahist of price_usd without 'outliers' (>5000$)
    plt.figure()
    dflow = df.drop(df[df['price_usd'] > 2000].index)
    plt.hist(dflow['price_usd'].dropna(), bins=100)
    plt.title('price_usd (<2000$)')
    plt.xlabel('Value'), plt.ylabel('Frequency')
    plt.savefig('results/hist/hist_price_usd_NO2000.png')
    plt.close()

    print 'Histograms saved'


def clicked_booked_histograms(df,clicked):
    ''' This function checks if missing data has an impacton book/click percentage
    '''

    if clicked == True:
        target_to_study = 'click_bool'
    else:
        target_to_study = 'booking_bool'

    # Variables to study
    hotel_descriptor_variables = ['prop_starrating','prop_review_score','prop_brand_bool','prop_location_score1', \
    'prop_location_score2','prop_log_historical_price','srch_query_affinity_score', 'orig_destination_distance']

    booked_ratio, booked_NA_ratio= [], []
    for variable in hotel_descriptor_variables:
        # count() counts non null values of an object
        if variable in ['prop_review_score','prop_location_score2','srch_query_affinity_score', 'orig_destination_distance']:
            booked = df[df[target_to_study] == 1][variable].count()
            non_booked = df[df[target_to_study] == 0][variable].count()
        else:
            ndf = df[df[target_to_study] == 1][variable] != 0
            booked = ndf.sum()
            ndf_non = df[df[target_to_study] == 0][variable] != 0
            non_booked = ndf_non.sum()
        booked_NA = len(df[df[target_to_study] == 1][variable]) - booked
        non_booked_NA = len(df[df[target_to_study] == 0][variable]) - non_booked
        booked_ratio.append(float(booked)/float(non_booked))
        if non_booked_NA == 0:
            booked_NA_ratio.append(0)
        else:
            booked_NA_ratio.append(float(booked_NA)/float(non_booked_NA))

    # Save results in a file
    f = open('results/missing_info_%s_ratio.txt','w')
    f.write('1st line: %s_ratio; 2ndline: %s_NA_ratio;3rdline: variables'%(str(target_to_study),str(target_to_study)))
    for element in booked_ratio:
        f.write('%s\t'%(str(element)))
    f.write('\n')
    for element in booked_NA_ratio:
        f.write('%s\t'%(str(element)))
    f.write('\n')
    for element in hotel_descriptor_variables:
        f.write('%s\t'%(str(element)))
    f.write('\n')

    # Plot the results
    ind = np.array(range(len(booked_ratio)))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, booked_ratio, width, color='r',label='no NA')
    rects2 = ax.bar(ind + width, booked_NA_ratio, width, color='y',label='NA')
    ax.set_xticklabels(hotel_descriptor_variables,rotation='vertical')
    plt.title('%s ratio study of missing data'%(str(target_to_study)))
    plt.ylabel('%s ratio'%(str(target_to_study)))
    plt.legend(prop={'size':6})
    plt.tight_layout()
    plt.savefig('results/missing_info_%s_ratio.png'%(str(target_to_study)))
    #plt.show()
    plt.close()


def plot_outliers(df):
    ''' Creates boxplots charts of outliers.
    Price/distance variables are grouped on one graph
    Other numerical variables are grouped on the other graph
    '''
    plt.figure()
    df.boxplot(column = ['prop_starrating','prop_review_score','prop_location_score1','prop_location_score2','srch_adults_count','srch_children_count','srch_room_count'])
    plt.show()
    plt.figure()
    df.boxplot(column = ['prop_log_historical_price','price_usd','orig_destination_distance'])
    plt.show()


def main():
    ## 1. Import dataset and transform to Pandas
    train = file_to_pandas()

    ## 2. Select 1/10 of the dataset, randomly.
        ## checking for same distribution of target hotel AND same of time
        ## Select only if we're changing the dataset
    dataset_reduction(train)

    ## 3. Plot distributions in pie chart - not used
    # distributions(train)

    ## 4. Plot the missing points
    # missing_points(train)

    ## 5. Study relationship between variables
    # plot_variables(train,'position','booking_bool')

    ## 6. Study the effect of the random sorting in Expedia (whether this can influence)
            ## the final decision of final booking
    #study_effect_random(train)

    ## 7. Study relationship between hotel stars and average hotel reviews
    # histogram_rating_hotels(train,0)

    ## 8. Study relationship between hotel position and booking
    # histogram_position_vs_booking(train)

    ## 9. Plot histograms for all variables
    # histogram_all(train)

    ## 10. Study how the missing data affected the click/booking
    # clicked_booked_histograms(train,False)
    # clicked_booked_histograms(train,True)

    ## 11. Boxplot of the data points, looking for outliers
    plot_outliers(train)




if __name__ == "__main__":
    main()
