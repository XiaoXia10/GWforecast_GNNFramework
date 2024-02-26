# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:33:33 2023

@author: Xiao Xia Liang
"""
from os import chdir
import numpy as np 
import pandas as pd

chdir(r".\")
from utils import adj_matrix, min_max_norm, resample_data 
from get_pca_signal import standardize_pca_transform


#%%
steady_path = r'.\steady_heads.csv'
trans_path = r'.\real26_sim3.csv'

#%%

############## Get the sampled raw data with specific timestep ###############

timestep = 30
resampled_data = resample_data(trans_path, timestep)

#################### Get Adj matrix ##########################
radius = 0.25

steady_data = adj_matrix(steady_path, trans_path, radius)
data = steady_data.steady_data_compilation()
adj = steady_data.adaptive_adj(data)

################### Get PCA embedded signal #####################

compiled_trans_head, compiled_trans_pump = standardize_pca_transform(resampled_data)    
