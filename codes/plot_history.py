# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 19:05:40 2024

@author: Xiao Xia Liang
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from os.path import join, isfile, exists
from os import listdir

# load history training info
path = path = r'G:\My Drive\PhD\Mercier_model\final_callback_models\sim1_adj025_2\history'

files = [f for f in listdir(path) if isfile(join(path, f))]   
files.sort()

fig, ax = plt.subplots(2,1,figsize=(7,5))

ax[0].set_title('Dataset 1', fontsize=15)
ax[0].set_ylabel('Training Loss', fontsize=15)
ax[1].set_ylabel('Validation Loss', fontsize=15)
ax[1].set_xlabel('Epoch', fontsize=15)

for filename in files:
    with open(join(path, filename), 'rb') as file:
        name = filename.split('.')
        history = pickle.load(file)
        train_mse = history['loss']
        val_mse = history['val_loss']
        
        ax[0].plot(train_mse, 'g', alpha=0.5)
        ax[1].plot(val_mse, 'r', alpha=0.5)





