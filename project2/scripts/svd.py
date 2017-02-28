import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from helpers import load_data, preprocess_data

path_dataset = "data/data_train.csv"
ratings = load_data(path_dataset)

#get SVD components from train matrix. Choose k.
u, s, vt = svds(ratings.copy(), k = 40)
s_diag_matrix=np.diag(s)
pred = np.dot(np.dot(u, s_diag_matrix), vt)

from helpers import calculate_error
print(calculate_error(ratings.copy(), pred))