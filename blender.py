"""Very simple blender of the desired regressors and across months.

Copyright 2012, Emanuele Olivetti.
BSD license, 3 clauses.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import load_data
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
import pickle
import gzip

if __name__ == '__main__':

    np.random.seed(0)
    n_folds = 5

    X, X_submission, ys, ids, idx = load_data.load()

    # Smart transformation to avoid logscale in evaluation:
    ys = np.log(ys/500.0 + 1.0)

    y_submission = np.zeros((X_submission.shape[0], 12))

    # regs = [RandomForestRegressor(n_estimators=100, n_jobs=-1, max_features='auto'),
    #         ExtraTreesRegressor(n_estimators=100, n_jobs=-1, max_features='auto'),
    #         GradientBoostingRegressor(learn_rate=0.01, subsample=0.5, max_depth=6, n_estimators=5000)]

    # My best submission used just this one:
    regs = [GradientBoostingRegressor(learn_rate=0.001, subsample=0.5, max_depth=6, n_estimators=20000)]

    dataset_blend_train = np.zeros((X.shape[0], 12*len(regs)), dtype=np.double)
    dataset_blend_submission = np.zeros((X_submission.shape[0], 12*len(regs), n_folds), dtype=np.double)

    for i in range(12):
        print "Month", i
        y = ys[:,i]
        kfcv = KFold(n=X.shape[0], k=n_folds)
        for j, (train, test) in enumerate(kfcv):
            print "Fold", j
            for k, reg in enumerate(regs):
                print reg
                reg.fit(X[train], y[train])
                dataset_blend_train[test,12*k+i] = reg.predict(X[test])
                dataset_blend_submission[:,12*k+i,j] = reg.predict(X_submission)

    
    dataset_blend_submission_final = dataset_blend_submission.mean(2)
    print "dataset_blend_submission_final:", dataset_blend_submission_final.shape

    print "Blending."
    for i in range(12):
        print "Month", i, '-',
        y = ys[:,i]
        reg = RidgeCV(alphas=np.logspace(-2,4,40))
        reg.fit(dataset_blend_train, y)
        print "best_alpha =", reg.best_alpha
        y_submission[:,i] = reg.predict(dataset_blend_submission_final)
                
    # transforming back outcomes to the original scale:
    y_submission = (np.exp(y_submission) - 1.0) * 500.0
    
    print "Saving results."
    np.savetxt("test.csv", np.hstack([ids[:,None], y_submission]), fmt="%d", delimiter=',')
