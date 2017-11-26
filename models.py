import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def generate_X(ts, length):
    X = []
    for pos in range(length, len(ts)):
        X.append(ts[pos-length:pos])
    return np.array(X)

def train_test_score(model, ts, y, train_size=0.75, length=2 * 60 * 60, plot=False, return_proba=False):
    y = y[length:]

    if plot:
        plot_with_target(ts,y)
    X = generate_X(ts,length)
    train_len = int(len(X) * train_size)
    X_train, X_test = X[:train_len], X[train_len:]
    y_train, y_test = y[:train_len], y[train_len:]
    model.fit(X_train, y_train)
    
    train_proba = model.predict_proba(X_train)[:,1]
    test_proba = model.predict_proba(X_test)[:,1]

    roc_auc_test = roc_auc_score(y_score=test_proba, y_true=y_test)
    roc_auc_train = roc_auc_score(y_score=train_proba, y_true=y_train)
    if return_proba:
        return (roc_auc_train, roc_auc_test), (train_proba, test_proba)
    return roc_auc_train, roc_auc_test