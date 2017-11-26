import os
import sys
import numpy as np
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt

def get_target(data, top=.05, **kwargs):
    '''accepts dataframe of numeric, returns np.array of shape (data.shape[0], data.shape[1]) with 1s and 0s'''
    ewma = np.array(data.ewm(**kwargs).var())
    ewma_k_stat = []
    for i in range(ewma.shape[1]):
        a = ewma[:, i]
        a = a[np.isfinite(a)]
        k = int((1.0 - top) * a.shape[0])
        ewma_k_stat.append(np.partition(a, k)[:k].max())
    ewma_k_stat = np.array(ewma_k_stat)
    target = np.array([[1 if np.isfinite(ewma[i, j]) and ewma[i, j] >= ewma_k_stat[j] else 0
               for j in range(ewma.shape[1])]
              for i in range(ewma.shape[0])])
    return target

def rolling_sum(a, n=1000) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret

################ RUN THIS ###################
def get_target_future(data, horizont=1000, min_anomaly=50, top=.1, **kwargs):
    target = get_target(data, top, **kwargs)
    dim = target.shape[1]
    target = np.vectorize(lambda v: 1 if v >= min(dim / 2, 1) else 0)(target.sum(axis=1))
    target = rolling_sum(target, n=horizont)
    target_future = target
    target_future[:-horizont] = target[horizont:]
    target_future[-horizont:] = np.full(shape=(horizont,), fill_value=target[-1])
    return np.vectorize(lambda s: 1 if s >= min_anomaly else 0)(target_future)



#target = get_target(train[['efficiency', 'energy_cons', 'quality']].iloc[:100000], halflife=100)
#print(target[-80:])
# target = get_target_future(train[['efficiency', 'energy_cons', 'quality']],
#                            horizont=1000, min_anomaly=100, top=.01,
#                            halflife=100)



# lightweight # 
def moving_dispersion(x, window):
    disp = []
    length = len(x)
    x = np.concatenate([x,np.ones(window, dtype=np.float32) * np.mean(x[-10:])]) 
    for pos in range(length):
        mean = np.mean(x[pos:pos + window])
        diff = np.array(x[pos:pos + window]) - mean
        disp.append(np.mean(diff**2))
    return np.array(disp)
def get_ma_target(df, window=50, top=0.05):
    ma = moving_dispersion(df['trend'].values,window=window)
    args = np.argsort(ma)
    ans = np.zeros(len(args), dtype=np.float32)
    ans[args[-int(top * len(args)):]] = 1
    return ans
