# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:21:07 2024

@author: Xiao Xia Liang
"""

import argparse
import os 
from durbango import pickle_save

def get_shared_arg_parser():

    seq_length_x = 4 
    seq_length_y = 1 
    shift = 1
    
    # These are hyperparameters that you must tune
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='The max num of training epoches')
    parser.add_argument('--num_nodes', type=int, default=200, help='Number of nodes')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="X Sequence Length, this is the input length")
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Y Sequence Length, this is the output length")
    parser.add_argument("--shift", type=int, default=shift, help="Y Sequence Length, this is the output length")
    parser.add_argument("--loss", type=str, default="mae", help = "The loss function")
    parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate')
    
    # Data preparation
    parser.add_argument('--train_percent', type=float, default=0.7, help='The percentage of data used for model training')
    parser.add_argument('--val_percent', type=float, default=0.15, help='The percentage of data used for model validation')
    
    # Paths of input and output data
    parser.add_argument("--df", type=str, default="data/sim1_resampled30_ObsWells44_pca_head", help="Water level readings.")
    parser.add_argument("--save_dir", type=str, default="experiment_LSTM_"+str(timestep)+"_"+str(seq_length_x)+str(seq_length_y)+str(shift), help="Output prediction data directory.")
    
    
    parser.add_argument('--n_iters', default=None, help='quit after this many iterations')
    parser.add_argument('--es_patience', type=int, default=20, help='quit if no improvement after this many iterations')
    parser.add_argument("--save_model_name", type=str, default="trained_lstm_model.h5", help="Save trained model name")
    
    return parser
    
    