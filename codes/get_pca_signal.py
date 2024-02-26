# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 20:35:48 2023

@author: Xiao Xia Liang
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from os import chdir

chdir(r"G:\My Drive\PhD\Mercier_model\final_codes\github")
from utils import compile_data, min_max_norm

def add_head_noise_(head_, noise_ratio = 0.1):
    head = head_
    #head = (head-np.mean(head))/np.std(head)
    
    head_noise_std = noise_ratio * np.std(head)
    print(head_noise_std)
    #pumping_noise_std = 0.1* np.std(pumping)

    gaussian_noise_head = np.random.normal(loc=0.0, scale=head_noise_std, size=head.shape)
    #gaussian_noise_pumping = np.random.normal(loc=0, scale=pumping_noise_std, size=pumping.shape)

    # Add the Gaussian noise to the original data
    head_noise = head + gaussian_noise_head
    #pumping_noise = pumping + gaussian_noise_pumping
    return head_noise

def standardize_pca_transform(resampled_data_, n_components = 2, add_noise=False):
    resampled_data = resampled_data_.copy()
    norm_data = min_max_norm(resampled_data_)
    pca = PCA(n_components=n_components)

    compiled_head = compile_data(norm_data, resampled_data['Heads (m)'])
    compiled_pump = compile_data(norm_data, resampled_data['Pumping Rates (m^3/s)'])
    
    if add_noise == True:
        all_head_noise =[]
        
        for i in range(compiled_head.shape[0]):
            head = compiled_head[i,:]
            head_noise = add_head_noise_(head)
            
            all_head_noise.append(head_noise)

        head_noise = np.array(all_head_noise)
        
    else:
        head_noise = compiled_head
    
    
    all_trans_pump = np.array([])
    all_trans_head = np.array([])
    
    for i in range(compiled_head.shape[1]):
        
        x = pd.concat([pd.Series(head_noise[:,i]), pd.Series(compiled_pump[:,i])], axis = 1)
        x = StandardScaler().fit_transform(x)
        
        x_trans = pca.fit_transform(x)    
        trans_head = x_trans[:,0]
        trans_pump = x_trans[:,1]
        
        all_trans_head = np.append(all_trans_head, trans_head)
        all_trans_pump = np.append(all_trans_pump, trans_pump)
        
    compiled_trans_head = compile_data(norm_data, all_trans_head)    
    compiled_trans_pump = compile_data(norm_data, all_trans_pump)
    
    return compiled_trans_head, compiled_trans_pump












