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

# in the user-item matrix, compute cosine similarity between users
from sklearn.metrics.pairwise import pairwise_distances
ratings_T = ratings.T
user_distance = pairwise_distances(ratings_T, metric='cosine')
user_similarity = 1 - user_distance

# use the similarity to predict missing values in the matrix
mean_user_rating = ratings_T.mean(axis=1)
ratings_diff = (ratings_T - mean_user_rating)
pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
pred = pred.T

# create prediction matrix
def create_prediction_matrix(prediction):
    nz_row, nz_col = prediction.nonzero()
    nz_pred = list(zip(nz_row, nz_col))
    for d, n in nz_pred:
        if(pred[d,n] < 0.5) : pred[d,n] = 1
        elif(pred[d,n] > 5.4) : pred[d,n] = 5
        prediction[d,n] = round(pred[d,n])
    return prediction

# Make submission
from helpers import load_data, preprocess_data
path_dataset = "data/sampleSubmission.csv"
prediction = load_data(path_dataset)

from helpers import create_csv_submission_proj2
# estimate prediction matrix using method of similarity
prediction = create_prediction_matrix(prediction)
# create submission
create_csv_submission_proj2(prediction, 'data/prediction_similarity_1.csv')
