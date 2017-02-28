# -*- coding: utf-8 -*-
"""some functions for plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_raw_data(ratings):
    """plot the statistics result on raw rating data."""
    # do statistics.
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    sorted_num_movies_per_user = np.sort(num_items_per_user)[::-1]
    sorted_num_users_per_movie = np.sort(num_users_per_item)[::-1]

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sorted_num_movies_per_user, color='blue')
    ax1.set_xlabel("users")
    ax1.set_ylabel("number of ratings (sorted)")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sorted_num_users_per_movie)
    ax2.set_xlabel("items")
    ax2.set_ylabel("number of ratings (sorted)")
    ax2.set_xticks(np.arange(0, 10000, 2000))
    ax2.grid()

    plt.tight_layout()
    plt.savefig("images/stat_ratings")
    plt.show()
    # plt.close()
    return num_items_per_user, num_users_per_item

def plot_train_test_lambda(train_errors, test_errors, lambdas): # from lab 3
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("best lambda using ALS")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("images/best_lambda_using_ALS_to_rename")
    
def plot_train_test_K(train_errors, test_errors, lambdas): # from lab 3
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    """
    plt.plot(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.plot(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("K")
    plt.ylabel("RMSE")
    plt.title("best K using ALS")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("images/best_K_using_ALS_to_rename")
    
    
def plot_train_test_K_CV(errors, lambdas): # from lab 3
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    """
    plt.plot(lambdas, errors, color='b', marker='*', label="error")
    plt.xlabel("K")
    plt.ylabel("RMSE")
    plt.title("best K using ALS")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("images/best_K_using_ALS_CV_to_rename")
    
def plot_train_test_lambda_CV(errors, lambdas): # from lab 3
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    """
    plt.semilogx(lambdas, errors, color='b', marker='*', label="error")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("best lambda using ALS")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("images/best_K_using_ALS_CV_to_rename")
    
def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.plot(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.plot(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("K")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("images/cross_validation")