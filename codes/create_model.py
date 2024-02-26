# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:53:28 2023

@author: Xiao Xia Liang
"""

from stellargraph.layer import GCN_LSTM
from tensorflow.keras import Model
import tensorflow as tf

def create_model(adj, seq_len, learning_rate, af, af2, gc_layer_sizes, lstm_layer_sizes):
    gcn_lstm = GCN_LSTM(
        seq_len=seq_len,
        adj=adj,
        gc_layer_sizes= gc_layer_sizes,
        gc_activations=[af, af],
        lstm_layer_sizes=lstm_layer_sizes,
        lstm_activations=[af2, af2],
        dropout=0.5,
    )

    x_input, y_output = gcn_lstm.in_out_tensors()

    model = Model(inputs=x_input, outputs=y_output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=['mae'],
    )
    
    print("activation GCN "+str(af))
    print("activation LSTM "+str(af2))
    print("Learnning rate "+str(learning_rate))
    print("gc_layer_sizes"+str(gc_layer_sizes))
    print("lstm_layer_sizes"+str(lstm_layer_sizes))
    
    return model
