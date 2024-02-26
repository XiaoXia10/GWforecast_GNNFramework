# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:41:46 2023

@author: Xiao Xia Liang
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean

def resample_data(trans_path, timestep):
        
    timestep_ = timestep
    cut_off = 500 #cut the simulations data that is not yet stable
        
    raw_data = pd.read_csv(trans_path)
    raw_data = raw_data.astype({'Timestep (day)':'int'})
    raw_data = raw_data.set_index('Timestep (day)')
    raw_data = raw_data.loc[cut_off:, :]
        
    resampled = [] 
        
    while timestep_ <= raw_data.index[-1]:
        resampled.extend([time for time in raw_data.index if time >= timestep_ and time < timestep_+1])
        timestep_ += timestep 
                
        if timestep_ > raw_data.index[-1]:
            break
        
    resampled_time = pd.Series(resampled).unique()
        
    resampled_data = raw_data.loc[raw_data.index.isin(resampled_time)]
        
    ind_len = []
    for i in resampled_time:
        ind_len.append(len(resampled_data.loc[i])) 
    wells_len = min(ind_len)
        
    df = pd.DataFrame()
    for i in range(len(ind_len)):
        temp = resampled_data.loc[resampled_time[i]]
        df = pd.concat([df, temp[-wells_len:]])
        
    return df

    
def min_max_norm(df):
    df_norm_hp = df.drop(columns = ['X', 'Y', 'Z', 'Kx (m/s)', 'Ky (m/s)', 'Kz (m/s)'])   

    minz = -46.330953
    maxz = 69.816757
                
    maxpump = df_norm_hp['Pumping Rates (m^3/s)'].max()
    minpump = 0

    df_norm_hp['Heads (m)'] = (df_norm_hp['Heads (m)'] - minz)/(maxz-minz)
            
    df_norm_hp['Pumping Rates (m^3/s)'] = (df_norm_hp['Pumping Rates (m^3/s)'] - minpump)/(maxpump-minpump)
            
    return df_norm_hp       
            
def min_max_norm_all(df):
    df_norm = df
        
    minx = 280113.500252
    maxx = 298138.017578
    miny = 5003861.97363
    maxy = 5023378.99976
    minz = -46.330953
    maxz = 69.816757
                
    minKx = np.log10(1e-10)
    maxKx = np.log10(0.05)
    minKy = np.log10(1e-10)
    maxKy = np.log10(0.05)
                
    minKz = np.log10(1.9864e-12)
    maxKz = np.log10(0.0404149)
                
    df_norm['X'] = (df_norm['X'] - minx)/(maxx-minx)
    df_norm['Y'] = (df_norm['Y'] - miny)/(maxy-miny)
    df_norm['Z'] = (df_norm['Z'] - minz)/(maxz-minz)
    df_norm['Heads (m)'] = (df_norm['Heads (m)'] - minz)/(maxz-minz)
            
            
    df_norm['Kx (m/s)'] = (np.log10(df_norm['Kx (m/s)']) - minKx)/(maxKx-minKx) 
    df_norm['Ky (m/s)'] = (np.log10(df_norm['Ky (m/s)']) - minKy)/(maxKy-minKy)
    df_norm['Kz (m/s)'] = (np.log10(df_norm['Kz (m/s)'])- minKz)/(maxKz-minKz)
            
    return df_norm

class adj_matrix():
    
    def __init__(self, steady_path, trans_path, radius):
        self.steady_path = steady_path
        self.trans_path = trans_path
        self.radius = radius
        
    def steady_data_compilation(self):
        
        if self.trans_path.endswith('.pkl'):
            trans_data = pd.read_pickle(self.trans_path)
        else:
            trans_data = pd.read_csv(self.trans_path)
        

        if self.steady_path.endswith('.pkl'):
            steady_data = pd.read_pickle(self.steady_path)
        else:
            steady_data = pd.read_csv(self.steady_path)
        

        time_groups = trans_data.groupby('Timestep (day)')

        group1 = time_groups.get_group(trans_data.iloc[0,0])

        group1 = group1.drop(columns=['Timestep (day)', 'Pumping Rates (m^3/s)', 'Heads (m)'])

        frame = [group1, steady_data]

        df_steady = pd.concat(frame, axis=1)
        
        return df_steady
    
    def adaptive_adj(self, df_steady):

        k = 1
        algo = 'brute'
        steady_norm = min_max_norm_all(df_steady)
        
        X_norm = steady_norm.values.tolist() #convert scaled and sorted data to a list of list

        neigh = NearestNeighbors(n_neighbors=k, algorithm=algo).fit(X_norm)
        A = neigh.kneighbors_graph()
        fixed_adj = A.toarray() # adjancey matrix 
        fixed_adj = fixed_adj+np.identity(len(steady_norm))

        steady_norm = steady_norm.to_numpy()
        steady_norm_tran = steady_norm.T
        l2 = []
        
        # can do a upper or lower mat decomp instead of calculating the entier matrix
        for i in range(len(steady_norm)):
            temp = []
            for j in range(len(steady_norm)):
                temp.append(euclidean(steady_norm[i,:], steady_norm_tran[:,j]))
            l2.append(temp)
        
        l2_mat = np.vstack(l2) 
        adaptive_adj = np.where(l2_mat > self.radius, l2_mat, 1) 
        adaptive_adj = np.where(adaptive_adj == 1, adaptive_adj, 0)
       
        degrees= [np.sum(adaptive_adj[row, :]) for row in range(adaptive_adj.shape[0])]    

        for row in range(len(degrees)):
            if degrees[row]==1.0:
                adaptive_adj[row,:] = fixed_adj[row, :]
            else:
                pass
        
        return adaptive_adj


def compile_data(norm_data_, embed_data_):
    
    norm_data = norm_data_.copy()
    embed_data = embed_data_.copy()
    
    norm_data['Embed'] = embed_data
    norm_data = norm_data.drop(columns= ['Pumping Rates (m^3/s)', 'Heads (m)'])
    group_data = norm_data.groupby(['Timestep (day)'])

    group_size = group_data.size()
    obs_well_len = group_size.iloc[0]

    temp = []

    for obs in range(obs_well_len):
        temp_data = np.array(group_data.nth(obs))

        temp.append(temp_data)
        
        out_data = np.array(temp)
        
    out_data = np.transpose(np.hstack(out_data)) # This is for Stellargraphs T-GCN 
    
    return out_data

def denorm_head_pred(predicted_train_data_, predicted_test_data_, train_rate, seq_len, pre_len):
    # Denormalized back to the standization values in physical model 
    # physical model min, max elevation above sea level are used
    
    minz = -46.330953
    maxz = 69.816757

    predicted_train_data = predicted_train_data_
    predicted_test_data = predicted_test_data_
    
    train_pred_denormal = predicted_train_data * (maxz-minz) + minz
  
    test_pred_denormal = predicted_test_data * (maxz-minz) + minz

    return train_pred_denormal, test_pred_denormal

def train_test_split(df, train_percent, val_percent):
    #df = pd.DataFrame(data)
    time_len = df.shape[1]
    
    train_size = int(time_len * train_percent)
    val_size = int(time_len * (train_percent+val_percent))-int(time_len * train_percent)

    train_data = np.array(df.iloc[:, :train_size])
    val_data = np.array(df.iloc[:, train_size:train_size+val_size])
    test_data = np.array(df.iloc[:, train_size+val_size:])
    
    return train_data, val_data, test_data

def sequence_data_preparation(seq_len, pre_len, train_data, val_data, test_data):
    trainX, trainY, valX, valY, testX, testY = [], [], [], [], [], []
    for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
        a = train_data[:, i : i + seq_len + pre_len]
        trainX.append(a[:, :seq_len])
        trainY.append(a[:, -1])

    for i in range(val_data.shape[1] - int(seq_len + pre_len - 1)):
        b = val_data[:, i : i + seq_len + pre_len]
        valX.append(b[:, :seq_len])
        valY.append(b[:, -1])

    for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
        c = test_data[:, i : i + seq_len + pre_len]
        testX.append(c[:, :seq_len])
        testY.append(c[:, -1])

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    valX = np.array(valX)
    valY = np.array(valY)
    testX = np.array(testX)
    testY = np.array(testY)

    return trainX, trainY, valX, valY, testX, testY

