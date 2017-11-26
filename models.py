import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score

def generate_X(ts, length):
    X = []
    ts = np.array(ts)
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

def train_test_run(model_y, model_sigma, ts, sigmas, y, train_size=0.75,
    length=2 * 60 * 60, plot=False, return_proba=False):
    y = y[length:]
    sigmas = sigmas[length:]

    if plot:
        plot_with_target(ts[ts.column[0]],y)

    X = None

    for col in ts:
        features = generate_X(np.array(ts[col]),length)
        if X is None:
            X = features
        else:
            X = np.concatenate((X, features), axis=1)

    train_len = int(len(X) * train_size)
    X_train, X_test = X[:train_len], X[train_len:]
    y_train, y_test = y[:train_len], y[train_len:]
    model_y.fit(X_train, y_train)

    train_proba = model_y.predict_proba(X_train)[:,1]
    test_proba = model_y.predict_proba(X_test)[:,1]

    y_probas = np.array(model_y.predict_proba(X)[:, 1])
    Xy = np.concatenate((X, y_probas.reshape(len(y_probas), 1)), axis=1)

    Xy_train, Xy_test = Xy[:train_len], Xy[train_len:]
    sigma_train, sigma_test = sigmas[:train_len], sigmas[train_len:]

    model_sigma.fit(X_train, sigma_train)

    train_sigma_pred = model_sigma.predict(X_train)
    test_sigma_pred = model_sigma.predict(X_test)

    r2_train = r2_score(y_true=sigma_train, y_pred=train_sigma_pred)
    r2_test = r2_score(y_true=sigma_test, y_pred=test_sigma_pred)

    roc_auc_test = roc_auc_score(y_score=test_proba, y_true=y_test)
    roc_auc_train = roc_auc_score(y_score=train_proba, y_true=y_train)

    if return_proba:
        return (roc_auc_train, roc_auc_test, r2_train, r2_test),
        (train_proba, test_proba,train_sigma_pred, test_sigma_pred)

    return roc_auc_train, roc_auc_test, r2_train, r2_test
