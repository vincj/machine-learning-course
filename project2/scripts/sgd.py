import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Load the data
from helpers import load_data, preprocess_data

path_dataset = "data/data_train.csv"
ratings = load_data(path_dataset)

# Number of ratings per movie and user 
from plots import plot_raw_data
num_items_per_user, num_users_per_item = plot_raw_data(ratings)

print("min # of items per user = {}, min # of users per item = {}.".format(
        min(num_items_per_user), min(num_users_per_item)))

# split data into train and test
from helpers import split_data
valid_ratings, train, test = split_data(
    ratings, num_items_per_user, num_users_per_item, min_num_ratings=1, p_test=0.1)
# small subset for faster testing of the script
train_ = train[:100,:100]
test_ = test[:100,:100]

# initialize matrix factorization
def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    num_user = train.shape[1]
    num_item = train.shape[0]
    std = 0.1
    user_features = std * np.random.randn(num_features,num_user) #user_features.shape (20,1000)
    item_features = std * np.random.randn(num_item, num_features) #item_features.shape (10000,20)
    return user_features, item_features

def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    sum_err = 0
    for d, n in nz:
        err = data[d,n] - np.dot(item_features[d,:],user_features[:,n])
        sum_err += err**2
    rmse = 0.5*sum_err/len(nz)
    return rmse

from helpers import calculate_mse

def matrix_factorization_SGD(train, test, K_feat, epochs):
    """matrix factorization by SGD."""
    # define parameters
    gamma = 0.01
    num_features = K_feat   # K in the lecture notes
    lambda_user = 0.1
    lambda_item = 0.7
    num_epochs = epochs     # number of full passes through the train set
    errors = [0]
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train: # iterate over non zero elements
            e_dn = train[d,n] - np.dot(item_features[d,:],user_features[:,n])
            for k in range(num_features): # subtract the gradient for each feature
                user_features[k,n] = user_features[k,n] + gamma * e_dn * item_features[d,k] #- lambda_user * user_features[k,n]  
                item_features[d,k] = item_features[d,k] + gamma * e_dn * user_features[k,n] #- lambda_ * item_features[d,k]
        
        rmse_train = compute_error(train, user_features, item_features, nz_train) # for each epoch, last rmse_train value is returned
        errors.append(rmse_train)

    print("iter: {}, RMSE on training set: {}.".format(it, rmse_train))
    # evaluate the test error
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    rmse_test = compute_error(test, user_features, item_features, nz_train) 
    print("RMSE on test data: {}.".format(rmse_test))
    return user_features, item_features, rmse_train, rmse_test

user_features, item_features, rmse_train, rmse_test = matrix_factorization_SGD(train, test,20,10)

from helpers import build_k_indices, split_data_cross_validation
from plots import cross_validation_visualization
'''
# Plot the error for different values of K with cross-validation
def cross_validation():
    seed = 1
    k_fold = 4
    #Ks_feat = np.array([20, 25, 30, 35, 40, 45, 50])
    Ks_feat = np.array([20, 25])
    epochs = 4
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    for K_feat in Ks_feat:
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for k in range(k_fold):
            tr_, te_ = split_data_cross_validation(train_, k_fold, seed, k)
            user_features, item_features, loss_tr, loss_te = matrix_factorization_SGD(tr_, te_, K_feat, epochs)
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))
    
    cross_validation_visualization(Ks_feat, rmse_tr, rmse_te)
    
cross_validation()
'''
# assign predicted values to prediction matrix
def create_prediction_matrix(prediction):
    nz_row, nz_col = prediction.nonzero()
    nz_pred = list(zip(nz_row, nz_col))
    for d, n in nz_pred:
        pred = np.dot(item_features[d,:],user_features[:,n]) # matrices item_features and user_features are already computed above
        prediction[d,n] = round(pred)
    return prediction

# Make submission
# load test data
from helpers import load_data, preprocess_data

path_dataset = "data/sampleSubmission.csv"
prediction = load_data(path_dataset)

from helpers import create_csv_submission_proj2
# estimate prediction using method of sgd
prediction = create_prediction_matrix(prediction)
    
# create submission
create_csv_submission_proj2(prediction, 'data/prediction_SGD_2.csv')
