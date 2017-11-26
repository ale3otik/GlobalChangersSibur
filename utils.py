import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import gaussian_mixture_anomaly_detection as ad
from copy import deepcopy

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
    if v.find(',') < 0:
        return float(v)

    while True:
        p = v.find(',')
        if v[p+1:].find(',') < 0:
            break
        v = v[p+1:]
    return float(v.replace(',','.'))

def get_targets_with_mixture(data,horizont,T=None,top=0.0005):
    if T is None:
        T = data.shape[0]
    return ad.extract_anomaly_target(data,frame_period=T,halflife=10,horizont=horizont,top=top)

def get_dropped(ts, lag=10):
    return np.array([ts[i*lag] for i in range(len(ts)//lag)])
    
def get_expanded_features(ts, freq= 11, log=True, lag_to_drop=None, plot=False):
    if log:
        ts = np.log(ts + 10.0)
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
    df = pd.DataFrame(data = np.concatenate([[trend], [season], [diff]], axis=0).T,
                                columns=['trend', 'season', 'diff'])
    return df

def unite_features(list_df):
    return pd.concat(list_df)
    
def plot_with_target(ts, target):
    plt.figure(figsize=(12,8))
    X = np.arange(len(ts))
    Y = ts
    indices = [i for i in range(len(target)) if target[i]==1]
    plt.plot(X,Y)
    plt.plot(X[indices],Y[indices],'o',color='red')
    plt.show()