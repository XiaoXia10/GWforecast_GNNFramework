# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:17:22 2023
Train GCN_LSTM model
@author: Xiao Xia Liang
"""
from os import chdir 
import numpy as np

chdir(r"G:\My Drive\PhD\Mercier_model\final_codes\github")
from gnn_model import gnn_model

main_path = r'G:\My Drive\PhD\Mercier_model\final_callback_models\sim1_adj025_2\\'
data_path = r'G:\My Drive\PhD\Mercier_model\final_data\sim1_resampled30_ObsWells44_pca_head.npy'

adj = np.load(r'G:\My Drive\PhD\Mercier_model\final_data\adaptive_fixed_adj_44wells_025.npy')

af1 = "tanh"
af2 = "tanh"
batch_size=15 
learning_rate =0.001
gc_layer_sizes = [5,5]
lstm_layer_sizes = [600, 600]
train_rate = 0.7
val_rate = 0.15
seq_len = 4
pre_len = 1

model_name = 'trained_gnn_model.hdf5'
    
gnn = gnn_model(adj, seq_len, learning_rate, af1, af2, gc_layer_sizes, lstm_layer_sizes, batch_size, train_rate)

gnn_model = gnn.create_model()

train_gnn_model = gnn.train_gnn_model(gnn_model, main_path, model_name, data_path)
    





