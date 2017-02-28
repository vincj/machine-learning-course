import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt


# Load the data
from helpers import load_data, preprocess_data

path_dataset = "data/data_train.csv"
ratings = load_data(path_dataset)

# number of ratings per movie and user
from plots import plot_raw_data
num_items_per_user, num_users_per_item = plot_raw_data(ratings)

print("min # of items per user = {}, min # of users per item = {}.".format(
        min(num_items_per_user), min(num_users_per_item)))

# split the data
from helpers import split_data
valid_ratings, train, test = split_data(
    ratings, num_items_per_user, num_users_per_item, min_num_ratings=10, p_test=0.1)
# small subsets for test ing the script
train_ = train[:100,:100]
test_ = test[:100,:100]

# initialize matrix factorization
def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    num_user = train.shape[1]
    num_item = train.shape[0]
    user_features = np.random.rand(num_features,num_user) # user_features shape (20,943)
    item_features = np.random.rand(num_item, num_features) # item_features shape (1152,20)
    return user_features, item_features

def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    sum_err = 0
    for d, n in nz:
        err = data[d,n] - np.dot(item_features[d,:],user_features[:,n])
        sum_err += err**2
    rmse = 0.5*sum_err/len(nz)
    return rmse

def update_user_feature(
        train, user_features, item_features, lambda_user,
        nz_user_itemindices, I):
    """update user feature matrix."""
    for d, user_d in enumerate(nz_user_itemindices): # iterate over non zero users
        nnz_items_per_user = len(user_d[1]) # Number of items user d has rated
        if (nnz_items_per_user == 0): nnz_items_per_user = 1
            
        # Least squares solution
        A_d = np.dot(item_features[user_d[1]].T, item_features[user_d[1]]) + lambda_user * nnz_items_per_user * I
        V_d = np.dot(item_features[user_d[1]].T, train[user_d[1],user_d[0]].todense())
        user_features[:,user_d[0]] = np.linalg.solve(A_d,V_d)
    

def update_item_feature(
        train, item_features, user_features, lambda_item,
        nz_item_userindices, I):
    """update item feature matrix."""
    for n, item_n in enumerate(nz_item_userindices):
        nnz_users_per_item = len(item_n[1]) # Number of users who rated item n
        if (nnz_users_per_item == 0): nnz_users_per_item = 1
        # Least squares solution
        A_n = np.dot(user_features[:,item_n[1]], user_features[:,item_n[1]].T) + lambda_item * nnz_users_per_item * I
        V_n = np.dot(user_features[:,item_n[1]], train.T[item_n[1],item_n[0]].todense())
        #if (n%3 == 0): print("item_n: {}".format(item_n[0]), np.linalg.det(A_n))
        if (np.linalg.det(A_n) != 0): item_features.T[:,item_n[0]] = np.linalg.solve(A_n,V_n)
        else: 
            A_n[0,0] += 1; A_n[1,1] += 1; A_n[2,2] += 1; A_n[3,3] += 1; A_n[4,4] += 1; A_n[5,5] += 1 # if matrix A_n is singular, slightly modify several values
            item_features.T[:,item_n[0]] = np.linalg.solve(A_n,V_n)


from helpers import build_index_groups

def ALS(train, test):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    num_features = 20   # K in the lecture notes
    lambda_user = 0.1
    lambda_item = 0.7
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]
    
    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    user_features = np.matrix(user_features) # convert ndarray into matrix for subsequent operations
    item_features = np.matrix(item_features)
    
    item_features[:,0] = train.mean(axis=1) #Initialize the item matrix by assigning the average rating as the first row
    num_epochs = 5;
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)
    I = np.eye(num_features) # (k x k)-dimensional idendity matrix
    
    for it in range(num_epochs):
        print("starting iter: {}".format(it))
        
        # fix item_features and estimate user_features
        update_user_feature(train, user_features, item_features, lambda_user, nz_user_itemindices, I)

        # fix user_features and estimate item_features
        update_item_feature(train, item_features, user_features, lambda_item, nz_item_userindices, I)
        
        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
            
    # evaluate the test error
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(rmse))

ALS(train_, test_)