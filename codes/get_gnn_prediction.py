# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:02:58 2023

Reload saved GNN model head predictions 

@author: Xiao Xia Liang
"""


import pandas as pd
from os import chdir
from pathlib import Path

chdir(r'G:\My Drive\PhD\Mercier_model\final_codes\github')
from utils import train_test_split, sequence_data_preparation


def get_model_predictions(model, load_model_path, compiled_data_, seq_len, pre_len, learning_rate, train_rate, val_rate, batch_size):
    compiled_data = compiled_data_
    
    norm_data = (compiled_data-compiled_data.min())/(compiled_data.max()-compiled_data.min())
    norm_data = pd.DataFrame(norm_data)

    train_data, val_data, test_data = train_test_split(norm_data, train_rate, val_rate)

    trainX, trainY, valX, valY, testX, testY = sequence_data_preparation(
        seq_len, pre_len, train_data, val_data, test_data
    )

    model.load_weights(Path(load_model_path))

    loss_train, mae_train = model.evaluate(trainX, trainY, batch_size=batch_size, verbose=1)
    loss_val, mae_val = model.evaluate(valX, valY, batch_size=batch_size, verbose=1)
    loss_test, mae_test = model.evaluate(testX, testY, batch_size=batch_size, verbose=1)


    print("Restored model, train loss: {:5.6f}".format(loss_train))
    print("Restored model, train mae: {:5.6f}".format(mae_train))
    print("Restored model, validation loss: {:5.7f}".format(loss_val))
    print("Restored model, validation mae: {:5.7f}".format(mae_val))
    print("Restored model, test loss: {:5.7f}".format(loss_test))
    print("Restored model, test mae: {:5.7f}".format(mae_test))
    
    pred_train = model.predict(trainX, verbose=1)
    pred_val = model.predict(valX, verbose=1)
    pred_test = model.predict(testX, verbose=1)

    return pred_train, pred_val, pred_test

