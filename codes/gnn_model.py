# -*- coding: utf-8 -*-
"""
Xiao Xia Liang
INRS - PhD 
Dec 8, 2023

"""

from stellargraph.layer import GCN_LSTM
import stellargraph as sg
#import networkx as nx
#from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras import Model, callbacks
# from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
#from knockknock import email_sender
from os import chdir
import pickle 

# from sklearn.manifold._locally_linear import barycenter_kneighbors_graph

chdir(r"G:\My Drive\PhD\Mercier_model\final_codes\github")
from utils import train_test_split, sequence_data_preparation

class gnn_model():
    
    def __init__(self, adj, seq_len, learning_rate, af, af2, gc_layer_sizes, lstm_layer_sizes, batch_size, train_rate):
        
        self.adj = adj
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.af = af
        self.af2 = af2
        self.gc_layer_sizes = gc_layer_sizes
        self.lstm_layer_sizes = lstm_layer_sizes
        self.batch_size = batch_size
        self.train_rate = train_rate

        
    def create_model(self):
        gcn_lstm = GCN_LSTM(
            seq_len=self.seq_len,
            adj=self.adj,
            gc_layer_sizes= self.gc_layer_sizes,
            gc_activations=[self.af, self.af],
            lstm_layer_sizes=self.lstm_layer_sizes,
            lstm_activations=[self.af2, self.af2],
            dropout=0.5,
        )
    
        x_input, y_output = gcn_lstm.in_out_tensors()
    
        model = Model(inputs=x_input, outputs=y_output)
    
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=['mae'],
        )
        
        # print("activation GCN "+str(af))
        # print("activation LSTM "+str(af2))
        # print("Learnning rate "+str(learning_rate))
        # print("gc_layer_sizes"+str(gc_layer_sizes))
        # print("lstm_layer_sizes"+str(lstm_layer_sizes))
        
        model.summary()
        return model

    def train_gnn_model(self, model, main_path, model_name, data_path):
 
        batch_size = self.batch_size
        train_rate = self.train_rate
        val_rate = (1-train_rate)/2
        seq_len = self.seq_len
        pre_len = 1 # can only forecast one timestep

        
        # Compiled data for training and forecasting
        # Normalize data for PCA embedded head levels
        compiled_pca = np.load(data_path)
    
        norm_data = (compiled_pca-compiled_pca.min())/(compiled_pca.max()-compiled_pca.min())
        norm_data = pd.DataFrame(norm_data)
    
        # Train, Test data split
        train_data, val_data, test_data = train_test_split(norm_data, train_rate, val_rate)
    
    
        trainX, trainY, valX, valY, testX, testY = sequence_data_preparation(
            seq_len, pre_len, train_data, val_data, test_data
        )
    

        # regulate the model to prevent overfitting
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', 
            min_delta=0, 
            patience=20, 
            verbose=1, 
            mode='min'
            )
    
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=main_path + model_name, monitor='val_loss', 
            save_best_only=True, 
        #    save_weights_only=True,
            mode='min', 
            verbose=1
            )
    
        list_callback = [early_stopping, model_checkpoint]
    
        history = model.fit(
            trainX,
            trainY,
            epochs=2000,
            batch_size=batch_size,
            shuffle=True,
            callbacks=list_callback,
            verbose=1,
            validation_data=[valX, valY],
            )
        
        print(
             "Last history train loss: ",
             history.history["loss"][-1],
             "\n Last history train mae:",
             history.history["mae"][-1],
         )
    
        sg.utils.plot_history(history)
        
        # path = r'G:\My Drive\PhD\Mercier_model\final_callback_models\sim1_adj025_2\history\history_sim9.pkl'
        
        # with open(path, 'wb') as file:
        #     pickle.dump(history.history, file)
        
        