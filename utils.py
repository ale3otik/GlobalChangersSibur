import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import gaussian_mixture_anomaly_detection as ad
import models
import ewma
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import matplotlib.style
import matplotlib as mpl 

def clip(ts,treshold=3.0):
    plt.figure(figsize=(12,8))
    ts1 = deepcopy(ts)
    mean = np.mean(ts)
    diff = np.abs(ts - mean)
    sigm = np.std(ts)
    print(sigm)
    # plt.plot(range(len(ts)), diff)
    indices = np.where(diff > treshold * sigm)
    ts[indices] = mean + np.sign(ts[indices]) * treshold * sigm
    plt.plot(range(len(ts)), ts, alpha=0.5)
    plt.plot(range(len(ts)), ts1, alpha=0.5)
    plt.show()
    return ts

def read_ts(file):
    df = pd.read_csv(file,delimiter=';')
    values = [tofloat(v) for v in df[' value']]
    df[' value'] = values
    return df

def bad_to_mean(df):
    mean_good = df[df[' quality'] == 'Good'][' value'].mean()
    df[' value'][df[' quality'] != 'Good'] = mean_good
    return df

def tofloat(v):
    if isinstance(v, float):
        return v
    if v.find(',') < 0:
        return float(v)

    while True:
        p = v.find(',')
        if v[p+1:].find(',') < 0:
            break
        v = v[p+1:]
    return float(v.replace(',','.'))

def get_targets_with_mixture(data,horizont,halflife,T=None,top=0.0005):
    if T is None:
        T = data.shape[0]
    return ad.extract_anomaly_target(data,frame_period=T,halflife=halflife,horizont=horizont,top=top)

def get_dropped(ts, lag=10):
    return np.array([ts[i*lag] for i in range(len(ts)//lag)])
    
def get_expanded_features(ts, freq= 11, log=True, lag_to_drop=None, plot=False):
    if log:
        ts = np.log(ts + 1.0)
    res = sm.tsa.seasonal_decompose(ts, freq=freq)
    if plot:
        res.plot()
        plt.show()
    res_dict = dict()
    trend = res.trend
    season = res.seasonal
    nans = np.isnan(trend)
    trend[nans] = np.nanmean(trend)
    nans = np.isnan(season)
    season[nans] = np.nanmean(season)
    observed = res.observed
    diff = np.concatenate([[0.0], np.diff(trend)])

    if not lag_to_drop is None:
        trend = get_dropped(trend, lag_to_drop)
        season = get_dropped(season, lag_to_drop)
        diff = get_dropped(diff, lag_to_drop)
    df = pd.DataFrame(data = np.concatenate([[trend], [np.abs(diff)] , [diff]], axis=0).T,
                                columns=['trend', 'absdiff', 'diff'])
    return df

def unite_features(list_df):
    return pd.concat(list_df)
    
def plot_with_target(ts, target):
    plt.figure(figsize=(12,8))
    X = np.arange(len(ts))
    Y = ts
    indices = [i for i in range(len(target)) if target[i]==1]
    mean = np.mean(Y)
    sigm = np.std(Y)
    plt.ylim(mean - 7.0 * sigm, mean + 7.0 * sigm)
    plt.plot(X,Y)
    plt.plot(X[indices],Y[indices],'o',color='red')
    plt.show()

# def get_model_score(target, ts, model=LogisticRegression(),X_length = 50,return_proba=False):
#     return models.train_test_score(model, ts, target, length=X_length, return_proba=return_proba)

def end_to_end(path, 
        lag_to_drop=70, 
        target_extract_method='mixture', 
        clip_treshold=None,
        horizont = 60 * 11,
        halflife=10,
        X_length=50,
        top=0.005,
        plot=False):
    print('Reading data from \"' + path + '\"...')

    train = bad_to_mean(read_ts(path))
    train = bad_to_mean(train)
    origin_ts = train[' value'].values
    if not clip_treshold is None:
        origin_ts = clip(origin_ts,clip_treshold)
    
    print('TSA decomposition...')
    print('lag_to_drop',lag_to_drop)
    df = get_expanded_features(origin_ts, lag_to_drop=lag_to_drop)
    horizont = horizont//lag_to_drop
    if target_extract_method == 'mixture':
        targets = get_targets_with_mixture(df,horizont=horizont,halflife=halflife,top=top)
        if plot:
            plot_with_target(df['trend'].values, targets)
        sigmas = ewma.moving_dispersion(df['trend'].values, window=10)
        score = models.train_test_run(df[['trend','absdiff']], sigmas, targets,length=X_length)
        return score 

    # if target_extract_method == 'ewma':
    #     targets = ewma.get_target_future(df[['trend']],horizont=horizont, top=top, halflife=halflife)
    #     features = np.abs(df['trend'].values)
    #     score = get_model_score(targets, features, X_length=X_length)
    #     return score 

def plot_targets_colored(series, y_true, y_pred, size=500, edges=False):
    true_index = set()
    # plt.style.use('ggplot')
    plt.figure(figsize=(15,5))
    if edges:
        plt.plot(np.arange(len(series)), series)
    for i, y in enumerate(y_true):
        if y > 0:
            true_index.add(i)
    false_index = set(np.arange(len(series))) - true_index
    series_true = [x for i, x in enumerate(series) if i in true_index]
    series_false = [x for i, x in enumerate(series) if i in false_index]
    
    y_pred_true = [y for i, y in enumerate(y_pred) if i in true_index]
    y_pred_false = [y for i, y in enumerate(y_pred) if i in false_index]
    cmap = plt.cm.get_cmap('seismic')
    sc = plt.scatter(sorted(true_index), series_true, c=1 - np.array(y_pred_true),
                cmap = cmap, edgecolors='red', linewidth=5,s=size)
    plt.scatter(sorted(false_index), series_false, c=1 - np.array(y_pred_false), 
                cmap=cmap,s=size)
    plt.colorbar(sc)
    # plt.show()