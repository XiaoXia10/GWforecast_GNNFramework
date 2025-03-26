# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:56:59 2024

@author: Xiao Xia Liang
"""
from tensorflow.keras import Sequential, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout


def create_model(train_X, train_y, args):
    """
    Build a two layer squential LSTM model for multivariate input and output
    Input shape = (timestep, number of features)
    Output shape = number of features
    
    """
    model = Sequential()

    # Add an Input layer explicitly for clarity, with shape as (timesteps, features)
    model.add(Input(shape=(train_X.shape[1], train_X.shape[2]), dtype=tf.float32))

    ## Add two LSTM layers with return sequences enabled to maintain the temporal dimension across the network
    model.add(LSTM(units=args.num_nodes, dropout=args.dropout, return_sequences=True))
    model.add(Dropout(args.dropout))
    
    ## Final Dense layer that outputs vectors of the same size as the number of features in the input
    model.add(Dense(train_y.shape[2]))

    adam = Adam(learning_rate=args.learning_rate)
    model.compile(loss=args.loss, optimizer=adam)
    
    return model
