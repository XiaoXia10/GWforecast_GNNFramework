# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:07:07 2023
This  script inverse tranforms PCA transformed data
@author: Xiao Xia Liang
"""

import pandas as pd
import numpy as np
from os import chdir

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

chdir(r'G:\My Drive\PhD\Mercier_model\final_codes')
from utils import compile_data, min_max_norm, train_test_split, sequence_data_preparation

def denormalization_(compiled_data_, predicted_train_data_, predicted_val_data_, predicted_test_data_):
    # Denormalized back to the standization values used in PCA 
    predicted_train_data = predicted_train_data_
    predicted_test_data = predicted_test_data_
    predicted_val_data = predicted_val_data_
    compiled_data = compiled_data_
    
    
    train_pred_denormal = predicted_train_data*(compiled_data.max()-compiled_data.min())+compiled_data.min()
    val_pred_denormal = predicted_val_data*(compiled_data.max()-compiled_data.min())+compiled_data.min()
    test_pred_denormal = predicted_test_data*(compiled_data.max()-compiled_data.min())+compiled_data.min()

    return train_pred_denormal, val_pred_denormal, test_pred_denormal


def prediction_pca_inverse(resampled_data_, compiled_pca_pumping_rate_, compiled_pca_head_, predicted_train_data_, predicted_val_data_, predicted_test_data_, train_rate, val_rate, seq_len, pre_len, n_components=2):
                          
    resampled_data = resampled_data_
    compiled_pca_pumping_rate = compiled_pca_pumping_rate_
    compiled_pca_head = compiled_pca_head_
    predicted_train_data, predicted_test_data, predicted_val_data = predicted_train_data_, predicted_test_data_, predicted_val_data_
    
    train_pred_denormal, val_pred_denormal, test_pred_denormal = denormalization_(compiled_pca_head, predicted_train_data, predicted_val_data, predicted_test_data)
    
    norm_data = min_max_norm(resampled_data)

    real_heads = resampled_data['Heads (m)']
    real_pump = resampled_data['Pumping Rates (m^3/s)']

    compiled_sim_heads = compile_data(norm_data, real_heads)
    compiled_sim_pumping_rate = compile_data(norm_data, real_pump)

    compiled_sim_heads = pd.DataFrame(compiled_sim_heads)
    compiled_sim_pumping_rate = pd.DataFrame(compiled_sim_pumping_rate)

    compiled_pca_pumping_rate = pd.DataFrame(compiled_pca_pumping_rate)

    train_pump, val_pump, test_pump = train_test_split(compiled_sim_pumping_rate, train_rate, val_rate)
    trainX_pump, trainY_pump, valX_pump, valY_pump, testX_pump, testY_pump = sequence_data_preparation(
        seq_len, pre_len, train_pump, val_pump, test_pump
    )


    train_head, val_head, test_head = train_test_split(compiled_sim_heads, train_rate, val_rate)
    trainX_head, trainY_head, valX_head, valY_head, testX_head, testY_head = sequence_data_preparation(
        seq_len, pre_len, train_head, val_head, test_head
    )

    
    train_pump, val_pump, test_pump = train_test_split(compiled_pca_pumping_rate, train_rate, val_rate)
    trainX_pump_saved, trainY_pump_saved, valX_pump_saved, valY_pump_saved, testX_pump_saved, testY_pump_saved = sequence_data_preparation(
        seq_len, pre_len, train_pump, val_pump, test_pump
    )
    
    
    pca = PCA(n_components=n_components)

    all_pred_head_train = np.array([])
    all_real_head_train = np.array([])
    # train_pred_inverse = []
    # train_head_pump = []
    for i in range(len(trainY_head)):
        head = trainY_head[i, :]
        pump = trainY_pump[i, :]
        pump_saved = trainY_pump_saved[i,:]
        
        head_pump = np.concatenate([[head, pump]]).T
        
        mu = np.mean(np.vstack((head, pump)), axis =1) 
        std = np.std(np.vstack((head, pump)), axis =1)
        
        head_pump_stand = StandardScaler().fit_transform(head_pump)
        pca.fit_transform(head_pump_stand)
        
        head_pred = train_pred_denormal[i,:]
        head_pump_pred = np.concatenate([[head_pred, pump_saved]]).T
        
        pred_inverse = pca.inverse_transform(head_pump_pred)
        
        pred_inverse = std*pred_inverse + mu

        # train_head_pump.append(head_pump)
        # train_pred_inverse.append(pred_inverse)        

        all_real_head_train = np.append(all_real_head_train, head)
        all_pred_head_train = np.append(all_pred_head_train, pred_inverse[:,0])
        
    all_real_head_train = np.reshape(all_real_head_train, trainY_head.shape)
    all_pred_head_train = np.reshape(all_pred_head_train, trainY_head.shape)
        
    
    all_pred_head_val = np.array([])
    all_real_head_val = np.array([])
    # test_pred_inverse = []
    # test_head_pump = []
    for i in range(len(valY_head)):
        head = valY_head[i, :]
        pump = valY_pump[i, :]
        pump_saved = valY_pump_saved[i,:]
        
        head_pump = np.concatenate([[head, pump]]).T
        
        mu = np.mean(np.vstack((head, pump)), axis =1) 
        std = np.std(np.vstack((head, pump)), axis =1)
        
        head_pump_stand = StandardScaler().fit_transform(head_pump)
        pca.fit_transform(head_pump_stand)
        
        head_pred = val_pred_denormal[i,:]
        head_pump_pred = np.concatenate([[head_pred, pump_saved]]).T
        
        pred_inverse = pca.inverse_transform(head_pump_pred)
        
        pred_inverse = std*pred_inverse + mu

        # test_head_pump.append(head_pump)
        # test_pred_inverse.append(pred_inverse)
        
        all_real_head_val = np.append(all_real_head_val, head)
        all_pred_head_val = np.append(all_pred_head_val, pred_inverse[:,0])
        
    all_real_head_val = np.reshape(all_real_head_val, valY_head.shape)
    all_pred_head_val = np.reshape(all_pred_head_val, valY_head.shape)

    
    all_pred_head_test = np.array([])
    all_real_head_test = np.array([])
    # test_pred_inverse = []
    # test_head_pump = []
    for i in range(len(testY_head)):
        head = testY_head[i, :]
        pump = testY_pump[i, :]
        pump_saved = testY_pump_saved[i,:]
        
        head_pump = np.concatenate([[head, pump]]).T
        
        mu = np.mean(np.vstack((head, pump)), axis =1) 
        std = np.std(np.vstack((head, pump)), axis =1)
        
        head_pump_stand = StandardScaler().fit_transform(head_pump)
        pca.fit_transform(head_pump_stand)
        
        head_pred = test_pred_denormal[i,:]
        head_pump_pred = np.concatenate([[head_pred, pump_saved]]).T
        
        pred_inverse = pca.inverse_transform(head_pump_pred)
        
        pred_inverse = std*pred_inverse + mu

        # test_head_pump.append(head_pump)
        # test_pred_inverse.append(pred_inverse)
        
        all_real_head_test = np.append(all_real_head_test, head)
        all_pred_head_test = np.append(all_pred_head_test, pred_inverse[:,0])
        
    all_real_head_test = np.reshape(all_real_head_test, testY_head.shape)
    all_pred_head_test = np.reshape(all_pred_head_test, testY_head.shape)
    
    return all_real_head_train, all_real_head_val, all_real_head_test, all_pred_head_train, all_pred_head_val, all_pred_head_test




