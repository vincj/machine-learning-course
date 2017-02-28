# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import scipy.sparse as sp
import csv


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices

def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
        min_num_ratings: 
            all users and items we keep must have at least min_num_ratings per user and per item. 
    """
    # set seed
    np.random.seed(988)
    
    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][: , valid_users]  
    
    num_rows, num_cols = valid_ratings.shape 
    train = sp.lil_matrix((num_rows, num_cols)) 
    test = sp.lil_matrix((num_rows, num_cols)) 
    
    nz_items, nz_users, _ = sp.find(valid_ratings)
    
    for u in set(nz_users): 
        row, col, _ = sp.find(valid_ratings[:, u])
        test_rows = np.random.choice(row, int(len(row) * p_test)) 
        train_rows = list(set(row) - set(test_rows)) 
        # add to train 
        train[train_rows, u] = valid_ratings[train_rows, u] 
        # add to test 
        test[test_rows, u] = valid_ratings[test_rows, u] 
    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test

def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return t.power(2).mean() # originally 1.0 * t.dot(t.T).mean() but the latest impementation is more appropriate


def calculate_error(real, pred):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    nz_row, nz_col = real.nonzero()
    nz = list(zip(nz_row, nz_col))
    
    sum_err = 0
    for d, n in nz:
        err = real[d,n] - pred[d,n]
        sum_err += err**2
    rmse = 0.5*sum_err/len(nz)
    return rmse

def build_k_indices(y, k_fold, seed): # from lab 4
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def split_data_cross_validation(ratings, k_fold, seed, k):
    """............."""
    num_rows, num_cols = ratings.shape 
    train = sp.lil_matrix((num_rows, num_cols)) 
    test = sp.lil_matrix((num_rows, num_cols)) 
    nz_items, nz_users, _ = sp.find(ratings)
    
    for u in set(nz_users): 
        row, col, _ = sp.find(ratings[:, u])
        k_indices = build_k_indices(row, k_fold, seed) # from lab 4, split data in k fold
        test_rows = row[k_indices[k]]
        train_rows = list(set(row) - set(test_rows))
        # add to train 
        train[train_rows, u] = ratings[train_rows, u] 
        # add to test 
        test[test_rows, u] = ratings[test_rows, u] 
    return train, test

import csv
def create_csv_submission_proj2(prediction, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: prediction (predicted ratings matrix)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        
        nz_row, nz_col = prediction.nonzero()
        nz_test = list(zip(nz_row, nz_col))
        nz_test.sort(key=lambda line: line[1]) # order the coordinates by columns
        for d, n in nz_test:
            pred = prediction[d,n]
            writer.writerow({'Id':'r'+'{}'.format(d+1)+'_c'+'{}'.format(n+1),'Prediction':float(pred)})