# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:52:27 2024

@author: Xiao Xia Liang
"""

import pandas as pd
import numpy as np
from os.path import join
import os

def train_val_split(df, train_percent, val_percent):
    
    num_samples = df.shape[0]
    num_val = round(num_samples * val_percent)
    num_train = round(num_samples *train_percent)
    num_test = num_samples - num_val - num_train
    
    time = df.index
    
    train, val, test = df[:num_train], df[num_train: num_train + num_val], df[-num_test:]
   
    train_time, val_time, test_time = time[:num_train], time[num_train: num_train + num_val], time[-num_test:]
    
    time_dic = {"train_time":np.array(train_time), "val_time":np.array(val_time), "test_time":np.array(test_time)} 
    
    return np.array(train), np.array(val), np.array(test), time_dic


def sequence_data_preparation(data, data_time, seq_length_x, seq_length_y, shift):

    num_samples = len(data)
    x, y, time_list = [], [], []
    #shift = seq_length_x
    max_t = num_samples - (seq_length_y+shift)
   
    for t in range(0, max_t, shift):  # t is the index of the last observation.
        
        total_window_len = data[t:seq_length_x+seq_length_y+t]
        x.append(total_window_len[:seq_length_x, :])
        y.append(total_window_len[seq_length_x:, :]) 
        
        time_window_len = data_time[t:seq_length_x+seq_length_y+t]
        time_list.append(time_window_len[seq_length_x:]) # Only collecting the time for y dataset
        
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    time_y = np.stack(time_list, axis=0)
    
    return x, y, time_y

def destandardize_pred(data_path, yhat, realy, time, seq_length_x):
    
    df = pd.read_csv(data_path, parse_dates=True, index_col=0)
    std = df.std()
    mean = df.mean()
    
    yhat = pd.DataFrame(yhat)
    realy = pd.DataFrame(realy)
    
    yhat = (yhat*std.values)+mean.values
    realy = (realy*std.values)+mean.values
    
    yhat.index = time
    realy.index = time
    # # To get back the missing sequence for the train, val, test y dataset
    # pad_nan = pd.DataFrame(np.nan, index=np.arange(seq_length_x), columns=yhat.columns)
    
    return yhat, realy

def standardize_df(data_path):
    
    df = pd.read_csv(data_path, parse_dates=True, index_col=0)
    std = df.std()
    mean = df.mean()
    
    df_std = (df-mean.values)/std.values
    
    return df_std