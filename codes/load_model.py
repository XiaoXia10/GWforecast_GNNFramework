# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:32:20 2023

@author: Xiao Xia Liang
"""
import pandas as pd
import numpy as np
from os import chdir
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

chdir(r".\")
from plot import plot_results
from utils import train_test_split, sequence_data_preparation
from get_gnn_prediction import get_model_predictions
from gnn_model import gnn_model
from pca_inverse_transform import prediction_pca_inverse, denormalization_

#%%
################## Model parameters and hyperparameters #######################

af1 = "tanh"
af2 = "tanh"
batch_size=15 
learning_rate = 0.001
gc_layer_sizes = [5,5]
lstm_layer_sizes = [600, 600]
train_rate = 0.7
val_rate = 0.15
seq_len = 4
pre_len = 1

######CHANGE simulation number of simulation
sim_num = 1 # 1 or 2. These are the two pumping scenarios
inter = 0 # 0 to 9. These are 10 model training results
well_num = 37 # 0 to 43 wells 
#########################################################

compiled_pca_head = np.load(r'.\sim'+str(sim_num)+'_resampled30_ObsWells44_pca_head.npy')

compiled_pca_pumping_rate = np.load(r'.\sim'+str(sim_num)+'_resampled30_ObsWells44_pca_pump.npy')

resampled_data = pd.read_csv(r'.\sim'+str(sim_num)+'_resampled30_ObsWells44.csv', index_col=0)

adj = np.load(r'.\adaptive_fixed_adj_44wells_025.npy')

#### get GNN predicted head levels
load_model_path = r'.\sim'+str(sim_num)+'_adj025_models\sim'+str(sim_num)+'_training'+str(inter)+'.hdf5'

###############################################################################
gnn = gnn_model(adj, seq_len, learning_rate, af1, af2, gc_layer_sizes, lstm_layer_sizes, batch_size, train_rate)

model = gnn.create_model()

train_pred_pca, val_pred_pca, test_pred_pca = get_model_predictions(
    model, load_model_path, compiled_pca_head, seq_len, pre_len, learning_rate, train_rate, val_rate, batch_size
)

denorm_train_pred_pca, denorm_val_pred_pca, denorm_test_pred_pca = denormalization_(
    compiled_pca_head, train_pred_pca, val_pred_pca, test_pred_pca
)


head_train, head_val, head_test, pred_head_train, pred_head_val, pred_head_test = prediction_pca_inverse(
    resampled_data, compiled_pca_pumping_rate, compiled_pca_head, train_pred_pca, val_pred_pca, test_pred_pca, train_rate, val_rate, seq_len, pre_len
)

##################compiled pca normalized head level###########################

norm_compiled_pca_head  = (compiled_pca_head-compiled_pca_head.min())/(compiled_pca_head.max()-compiled_pca_head.min())
norm_compiled_pca_head = pd.DataFrame(norm_compiled_pca_head)

train_head_pca, val_head_pca, test_head_pca = train_test_split(norm_compiled_pca_head , train_rate, val_rate)

_, norm_trainY_head_pca, _, norm_valY_head_pca, _, norm_testY_head_pca = sequence_data_preparation(
    seq_len, pre_len, train_head_pca, val_head_pca, test_head_pca
)

######################compiled pca head levels#################################
compiled_pca_head = pd.DataFrame(compiled_pca_head)

train_head_pca, val_head_pca, test_head_pca = train_test_split(compiled_pca_head, train_rate, val_rate)

_, trainY_head_pca, _, valY_head_pca, _, testY_head_pca = sequence_data_preparation(
    seq_len, pre_len, train_head_pca, val_head_pca, test_head_pca
)


##################### compiled pca pumping rates ##############################
compiled_pca_pumping_rate = pd.DataFrame(compiled_pca_pumping_rate)
train_pr_pca, val_pr_pca, test_pr_pca = train_test_split(compiled_pca_pumping_rate, train_rate, val_rate)

trainX, trainY, valX, valY, testX, testY = sequence_data_preparation(
    seq_len, pre_len, train_pr_pca, val_pr_pca, test_pr_pca
) 

#%%
############## Well y predictions and y true ################

y_pred_pca = np.hstack((train_pred_pca[:,well_num], val_pred_pca[:,well_num], test_pred_pca[:,well_num]))
y_pca = np.hstack((norm_trainY_head_pca[:,well_num], norm_valY_head_pca[:,well_num], norm_testY_head_pca[:,well_num]))

y_pred_head = np.hstack((pred_head_train[:,well_num], pred_head_val[:,well_num], pred_head_test[:,well_num]))
y_head = np.hstack((head_train[:,well_num], head_val[:,well_num], head_test[:,well_num]))

#%%
############# Metrics RMSE, MEA, and R_squared calculations ################

#simulated y in normalized PCA space and GNN predicted y in PCA space

rmse_pca = mean_squared_error(y_pca, y_pred_pca)
mae_pca = mean_absolute_error(y_pca, y_pred_pca)
r2_pca = r2_score(y_pca, y_pred_pca)

# head y and y-pred
rmse_head = mean_squared_error(y_head, y_pred_head)
mae_head = mean_absolute_error(y_head, y_pred_head)
r2_head= r2_score(y_head, y_pred_head)

print("rmse_pca: {:5.6f}".format(rmse_pca))
print("mae_pca: {:5.6f}".format(mae_pca))

# Metrics for testing set only to observe the predictive performance of GNN model
rmse_pca_test = mean_squared_error(norm_testY_head_pca, test_pred_pca)
mae_pca_test = mean_absolute_error(norm_testY_head_pca, test_pred_pca)

print("rmse_pca_test: {:5.7f}".format(rmse_pca_test))
print("mae_pca_test: {:5.7f}".format(mae_pca_test))

print("r2_pca : {:5.6f}".format(r2_pca))






