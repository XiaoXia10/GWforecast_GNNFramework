# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:07:32 2024

@author: Xiao Xia Liang
"""
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Reshape, TimeDistributed, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers, callbacks
import tensorflow as tf
from tensorflow.compat.v1.ragged import RaggedTensorValue
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
import argparse
from get_parser import get_shared_arg_parser
import time

from util_lstm import train_val_split, sequence_data_preparation, standardize_df
from create_lstm_model import create_model

def main(args):
    
    df_std = standardize_df(args.df)
    
    train, val, test, time_dic = train_val_split(df_std, args.train_percent, args.val_percent)

    time_for_train = time_dic["train_time"]
    time_for_val = time_dic["val_time"]
    time_for_test = time_dic["test_time"]

    x_train, y_train, train_time_y = sequence_data_preparation(train, time_for_train, args.seq_length_x, args.seq_length_y, args.shift)
    x_val, y_val, val_time_y = sequence_data_preparation(val, time_for_val, args.seq_length_x, args.seq_length_y, args.shift)
    x_test, y_test, test_time_y = sequence_data_preparation(test, time_for_test, args.seq_length_x, args.seq_length_y, args.shift)
    
    np.save(join( args.save_dir, "x_train.npy"), x_train)
    np.save(join( args.save_dir, "y_train.npy"), y_train)
    np.save(join( args.save_dir, "x_val.npy"), x_val)
    np.save(join( args.save_dir, "y_val.npy"), y_val)
    np.save(join( args.save_dir, "x_test.npy"), x_test)
    np.save(join( args.save_dir, "y_test.npy"), y_test)
    np.save(join(args.save_dir, "test_time_y.npy"), test_time_y)
    np.save(join(args.save_dir, "val_time_y.npy"), val_time_y)
    np.save(join(args.save_dir, "train_time_y.npy"), train_time_y)
    
    
    model = create_model(x_train, y_train, args)
    
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', 
                                             min_delta=0, 
                                             patience=args.es_patience, 
                                             verbose=1, 
                                             mode='min')
    
    save_model = join(args.save_dir, args.save_model_name)
    model_checkpoint =  callbacks.ModelCheckpoint(filepath=save_model, 
                                                  monitor='val_loss', 
                                                  save_best_only=True, 
                                                  verbose=1)

    list_callback = [early_stopping, model_checkpoint]

    history = model.fit(x=x_train, 
                        y=y_train, 
                        epochs=args.epochs, 
                        batch_size= args.batch_size, 
                        validation_data=(x_val, y_val),
                        callbacks=list_callback,
                        verbose=2, 
                        shuffle=True)
    
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss', fontsize= 15)
    plt.ylabel('MSE Loss', fontsize= 15)
    plt.xlabel('Epoch', fontsize= 15)
    plt.legend()
    

if __name__ == "__main__":
      
    parser = get_shared_arg_parser()
    args = parser.parse_args()
    
    if os.path.exists(args.save_dir):
        reply = str(input( 'output directory exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.save_dir)
    
    t1 = time.time()
    main(args)
    t2 = time.time()
    
    mins = (t2 - t1) / 60
    print(f"Total time spent: {mins:.2f} minutes")