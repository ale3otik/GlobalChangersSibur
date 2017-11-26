import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def read_ts(file='data/eff_train/eff_train2.csv'):
    df = pd.read_csv(file,delimiter=';')
    values = [tofloat(v) for v in df[' value']]
    df[' value'] = values
    return df

def bad_to_mean(df):
    mean_good = df[df[' quality'] == 'Good'][' value'].mean()
    df[' value'][df[' quality'] != 'Good'] = mean_good
    return df

def tofloat(v):
    while True:
        p = v.find(',')
        if v[p+1:].find(',') < 0:
            break
        v = v[p+1:]
    return float(v.replace(',','.'))