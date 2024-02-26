# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:46:05 2023

@author: Xiao Xia Liang
"""
import pandas as pd
import numpy as np
from os import chdir
import matplotlib.pyplot as plt

def plot_results(well_num, sampling_days, trainY, valY, testY, y_pred_pca,  y_pca, y_pred_head, y_head):
    #%matplotlib qt5
    train_days = list(range(0, (trainY.shape[0])*sampling_days, sampling_days))
    val_days = list(range(train_days[-1], (valY.shape[0]+1)*sampling_days+train_days[-1], sampling_days))
    test_days = list(range(val_days[-1], (testY.shape[0]+1)*sampling_days+val_days[-1], sampling_days))

    days =  list(range(sampling_days, ((trainY.shape[0]+1)+valY.shape[0]+testY.shape[0])*sampling_days,sampling_days))


    fig, ax = plt.subplots(2,1,figsize=(25,10))

    ax[0].plot(days, y_pred_pca, 'r-', label='Predicted', linewidth=4)
    ax[0].plot(days, y_pca, 'k--',label='Simulated', linewidth=4)
    
    ax[0].fill_between(train_days, max(y_pca)+np.std(y_pca), min(y_pca)-np.std(y_pca), facecolor = 'yellow', alpha = 0.2 )
    ax[0].fill_between(val_days, max(y_pca)+np.std(y_pca), min(y_pca)-np.std(y_pca), facecolor = 'green', alpha = 0.2 )
    ax[0].fill_between(test_days, max(y_pca)+np.std(y_pca), min(y_pca)-np.std(y_pca), facecolor = 'red', alpha = 0.2 )
    ax[0].set_title('Well '+str(well_num), fontsize=35)
    ax[0].set_ylabel('PCA Signal', fontsize=30)
    ax[0].set(xticks=[])
    ax[0].tick_params(axis='y', labelsize=15)
    ax[0].legend(loc='lower left',fontsize=30) 

    
 
    ax[1].plot(days, y_pred_head, 'r-',label='Predicted', linewidth=4)
    ax[1].plot(days, y_head, 'k--',label='Simulated', linewidth=4)
    ax[1].tick_params(axis='x', labelsize=20)
    ax[1].tick_params(axis='y', labelsize=15)

    ax[1].fill_between(train_days, max(y_head)+np.std(y_head), min(y_head)-np.std(y_head), facecolor = 'yellow', alpha = 0.2 )
    ax[1].fill_between(val_days, max(y_head)+np.std(y_head), min(y_head)-np.std(y_head), facecolor = 'green', alpha = 0.2 )
    ax[1].fill_between(test_days, max(y_head)+np.std(y_head), min(y_head)-np.std(y_head), facecolor = 'red', alpha = 0.2 )
    ax[1].set_xlabel('Time (days)',fontsize=25)
    ax[1].set_ylabel('ASL (m)', fontsize=30)
    ax[1].legend(loc='lower left',fontsize=30) 




