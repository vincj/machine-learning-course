import numpy as np
#import scipy
#import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt

# load the data
from helpers import load_data, preprocess_data
path_dataset = "data/data_train.csv"
ratings = load_data(path_dataset)

# plot number of ratings per movie and user
from plots import plot_raw_data
num_items_per_user, num_users_per_item = plot_raw_data(ratings)

# split data into train and test set
from helpers import split_data
valid_ratings, train, test = split_data(
    ratings, num_items_per_user, num_users_per_item, min_num_ratings=10, p_test=0.1)

from helpers import calculate_mse, calculate_error

# Use the global mean as prediction
def baseline_global_mean(real,prediction):
    """baseline method: use the global mean."""
    nz_row, nz_col = prediction.nonzero()
    nz_pred = list(zip(nz_row, nz_col))
    np.random.shuffle(nz_pred)
    gl_mean = round(real.sum()/len(real.nonzero()[0])) # sligthly different from mean() function because we only count nonzero values
    for d, n in nz_pred:
        prediction[d,n] = gl_mean

    print('global rmse', calculate_error(real,prediction))
    print('mse for global mean',calculate_mse(real,prediction))
    return prediction

pred_glob = baseline_global_mean(train.copy(), test.copy())
    
# Take user means as prediction
def baseline_user_mean(real, prediction):
    """baseline method: use the user means as the prediction."""
    nz_row, nz_col = prediction.nonzero()
    nz_pred = list(zip(nz_row, nz_col))
    np.random.shuffle(nz_pred)
    current_row = 0 # integer used to store value of current row
    u_mean = round(float(real[0].sum())/real[0].count_nonzero()) # user mean for user corresponding to row zero
    for d, n in nz_pred:
        if (d != current_row): #we compute the user mean only when reaching a new row, instead of computing for every nz_pred value
            current_row = d
            u_mean = round(float(real[d].sum())/real[d].count_nonzero()) # user mean for user corresponding to row d

        prediction[d,n] = u_mean
    
    print('user mean rmse', calculate_error(real,prediction))
    print('mse for user mean',calculate_mse(real,prediction))
    return prediction

pred_user = baseline_user_mean(train.copy(), test.copy())

# Take item means as predictions
def baseline_item_mean(real, prediction):
    """baseline method: use the item means as the prediction."""
    nz_row, nz_col = prediction.nonzero()
    nz_pred = list(zip(nz_row, nz_col))
    np.random.shuffle(nz_pred)
    nz_pred.sort(key=lambda line: line[1]) # order the coordinates by columns

    current_col = 0 # integer used to store value of current col
    i_mean = round(float(real[:,0].sum())/real[:,0].count_nonzero()) # item mean for item corresponding to column zero
    for d, n in nz_pred:
        if (n != current_col): #we compute the item mean only when reaching a new column, instead of computing for every nz_pred value
            current_col = n
            i_mean = round(float(real[:,n].sum())/real[:,n].count_nonzero()) # item mean for item corresponding to column zero
        prediction[d,n] = i_mean
    
    print('item mean rmse', calculate_error(real,prediction))
    print('mse for item mean',calculate_mse(real,prediction))
    return prediction

pred_item = baseline_item_mean(train.copy(), test.copy())
'''
# Make submission

# load data for submission
from helpers import load_data, preprocess_data

path_dataset = "data/sampleSubmission.csv"
prediction = load_data(path_dataset)

from helpers import create_csv_submission_proj2
# replace "baseline_..._mean" by the function you wish to test for submission
prediction = baseline_user_mean(ratings.copy(), prediction)
    
# create submission
create_csv_submission_proj2(prediction, 'data/prediction_user_mean.csv')
'''