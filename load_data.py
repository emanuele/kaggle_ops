"""Simple loader and shuffler of the numpy datasets.

Copyright 2012, Emanuele Olivetti.
BSD license, 3 clauses.
"""

import pickle
import gzip
import numpy as np

def load(filename='all_data.pickle.gz', shuffle_train=False):
    """Load dataset. Shuffle train data if requested
    """
    f = gzip.open(filename)
    all_data = pickle.load(f)
    X_train = all_data['X_train']
    X_test = all_data['X_test']
    ys = all_data['ys']
    ids = all_data['ids']
    idx = np.arange(X_train.shape[0])
    if shuffle_train:
        idx = np.random.permutation(idx)
        X_train = X_train[idx, :]
        ys = ys[idx, :]
    return X_train, X_test, ys, ids, idx
