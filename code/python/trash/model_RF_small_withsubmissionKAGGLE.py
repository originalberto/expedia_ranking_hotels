# Import required libraries
# from sklearn import cross_validation
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import metrics # cross-validation metric
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score, permutation_test_score, StratifiedKFold, GridSearchCV, cross_val_predict, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, mean_absolute_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from itertools import cycle
from sklearn.svm import SVR
import pickle
import math

def file_to_pandas(datafile):
    ''' This function opens the dataset and converts it to Pandas'''
    train = pd.read_csv(datafile)
    return train

# Tuning parameters of SVM
def tuning_parameters(X_train, y_train):
    # Set the parameters by cross-validation
    model = RandomForestRegressor() #Initialize with whatever parameters you want to

    # parameters to be tuned
    # n_estimators_values = [10,100,1000]
    # max_depth_values = [10,20]
    # max_depth_values = [5,50,150,200]

    # n_estimators_values = [150]
    # min_samples_leaf_values = [100]
    n_estimators_values = [300]
    min_samples_leaf_values = [20]

    parameters = {'n_estimators': n_estimators_values,'min_samples_leaf':min_samples_leaf_values}


    clf = GridSearchCV(model, param_grid=parameters, cv=10, scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)
    # 10-Fold Cross validation
    # scores =  cross_val_score(clf, X_train, y_train, cv=10,scoring='neg_mean_squared_error')
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print "Grid scores on development set: "

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    	print "%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params)

    return clf.best_params_

def train_RF(parameters,X_train,y_train,X_test,y_test):

    # model = RandomForestRegressor(n_estimators=parameters['n_estimators'], max_depth=parameters['max_depth'])
    model = RandomForestRegressor(n_estimators=parameters['n_estimators'], min_samples_leaf=parameters['min_samples_leaf'])
    k_fold = KFold(n_splits=10, random_state=True)
    model.fit(X_train, y_train)

    print model.predict(X_test)
    print y_test

    #distance = svr_rbf.decision_function(X_test)
    predicted_labels = model.predict(X_test)
    pred_lb = predicted_labels.tolist()
    RMSE = mean_absolute_error(y_test,predicted_labels)
    print 'rmse is', RMSE
    filename = 'RF_model_1_small_test.sav'
    pickle.dump(model, open(filename, 'wb'))
    return predicted_labels


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

# Deletes non-desired columns from the dataframe
def delete_useless_columns(dataframe):
    useless_variables = ['Unnamed: 0', 'Unnamed: 0.1', 'srch_id','year','click_bool','position','booking_bool']
    for variable in useless_variables:
        del dataframe[variable]
    return dataframe

# this function yields 2 dataframes:
# 1 - Dataframe containing input vectors for a model
# 2 - Dataframe containing output values (variable that is going to be predicted) y a model
def separate_input_and_target_variable(dataframe,computeNDCG=False):
    y = dataframe['relevance_score']
    if not computeNDCG:
        dataframe = dataframe.drop('relevance_score', 1)
        X = dataframe.as_matrix()
    if computeNDCG:
        X = dataframe
    return  X, y

def prepare_df(filename,trainingset_do=False,computeNDCG=False):
    train = file_to_pandas(filename)
    train = assign_relevance_score(train)
    if not computeNDCG:
        train = delete_useless_columns(train)
    train = train.dropna()
    if trainingset_do:
        # Shuffle rows from the training set
        train = train.sample(frac=1)
    if not computeNDCG:
        X, y = separate_input_and_target_variable(train)
    if computeNDCG:
        X, y = separate_input_and_target_variable(train,True)
    return X, y

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

def compute_NDCG(testsetfile,y_predicted):
    # Open the test set as a dataframe
    test, y_test = prepare_df(testsetfile,False,True)

    # Append the predicted values to the test set dataframe
    test['predicted_booking'] = y_predicted

    # Create a list for appending the NDCG score of every search
    NDCG_list = []

    # Loop through all the src_ID present in the test set
    for search_ID in test.srch_id.unique().tolist():
        # this if is for avoiding further errors
        if math.isnan(search_ID):
            pass
        else:
            # Create a dataframe with the info of that search ID
            nd = test[test['srch_id'] == search_ID]
            # Assign relevance score to every row according to booking_bool and
            # click_bool information (5 == (book_bool = 1), 1 == (click_bool = 1 AND book_bool=0), 0 == otherwise
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
    return mean_NDCG


def prepare_df_submissiontestset(filename):
    train = file_to_pandas(filename)
    srch_IDs = train.srch_id.tolist()
    hotel_IDs = train.prop_id.tolist()
    useless_variables = ['Unnamed: 0','Unnamed: 0.1', 'srch_id','year']
    for variable in useless_variables:
        del train[variable]
    train = train.dropna()
    print train.columns.values.tolist()

    X = train.as_matrix()
    return X, srch_IDs, hotel_IDs

# Main function
def main_model():
    # train = file_to_pandas('data/training_final1.csv')
    # test = file_to_pandas('data/SUBSET_test_final_notdownsampled.csv')

    # Load the training set and test set datasets
    X_train, y_train = prepare_df('data/training_final1.csv',True)
    print 'Loaded training set'
    X_test, y_test = prepare_df('data/SUBSET_test_final_notdownsampled.csv',False)
    print 'Loaded training set'

    # Compute best parameters of model
    print 'Estimating best parameters of model'
    best_parameters = tuning_parameters(X_train, y_train)
    # Train model with best parameters and predict the prob. of being booked for the test set
    print 'Best parameters obtained! Training model with best parameters'
    testset_predicted = train_RF(best_parameters,X_train,y_train,X_test,y_test)

    # Save memory
    print 'Deleting old variables for saving memory'
    del X_train, y_train, X_test

    # Compute the normalized DCG
    print 'Computing mean NCDG for the predictions'
    mean_NDCG = compute_NDCG('data/SUBSET_test_final_notdownsampled.csv',testset_predicted)
    print 'Mean NDCG is: ', mean_NDCG

def main():
    # Predict the output for the REAL test set (the one from the competition)
    real_testset_file = 'data/kaggle_2.csv'

    # Fill mp for test set submission?
    # preprocess test set with values (medians, etc) from test set file

    # Load the dataset input values and the corresponding srch_IDs and property IDs
    X_sub_testset, srch_IDs, hotel_IDs =  prepare_df_submissiontestset(real_testset_file)

    # load saved model
    saved_model = 'RF_model_1_small_v4_LESSFEATURES.sav'
    loaded_model = pickle.load(open(saved_model, 'rb'))
    predicted_values = loaded_model.predict(X_sub_testset)
    print loaded_model.get_params()
    print 'predictions done!'
    # Delete variables for saving memory
    del X_sub_testset

    # Create a df with the searchIDs, hotelIDs, and probability to be booked
    ndf = pd.DataFrame({'srch_id': srch_IDs, 'prop_id': hotel_IDs, 'predicted_probability': predicted_values})
    ndf.srch_id = ndf.srch_id.astype(int)
    ndf.prop_id = ndf.prop_id.astype(int)

    first = True
    for search_ID in ndf.srch_id.unique().tolist():
        print search_ID
        sorted_srch_df = ndf[ndf['srch_id'] == search_ID].sort_values(['predicted_probability'], ascending=False)
        del sorted_srch_df['predicted_probability']
        if first:
            global_sorted_df = sorted_srch_df
            first = False
        else:
            global_sorted_df = pd.concat([global_sorted_df,sorted_srch_df])
    # ndf = ndf.sort_values(['predicted_probability'], ascending=False)
    # print global_sorted_df
    global_sorted_df = global_sorted_df[['srch_id', 'prop_id']]
    global_sorted_df.to_csv('submission_predictions_2_v3.csv',index=False,header=False)

if __name__ == "__main__":main()
