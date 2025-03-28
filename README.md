# Graph neural network framework for spatio-temporal groundwater level forecasting

By: Xiao Xia Liang, Erwan Gloaguen, Maxime Claprood, Daniel Paradis and Dany Lauzon

Framework is designed for monintoring and pumping well groundwater level forecasting.
Data used for training the framework for this mauscript is simulated from a FEFLOW hydrogeological model.
Field measured data can also be used to train this network

The GNN model used contains 2 GCN and 2 LSTM layers followed by a dropout layer and a dense layer. The GNN model is packaged by [StellarGraph](https://stellargraph.readthedocs.io/en/stable/index.html). The model was inspired by [Zhao et al. 2018](https://arxiv.org/abs/1811.05320).

# Framework
![image](https://github.com/XiaoXia10/GWforecast_GNNFramework/assets/130078715/368435db-757b-4873-8583-55941a5e20a1)

# Data Download

Training data and trained models can be download from Google Drive.

[Training data](https://drive.google.com/drive/folders/1kKwdkPtquwoJcdkP0wFbwgiAl0O2Kbhu?usp=drive_link)

[GNN trained models](https://drive.google.com/drive/folders/12q9jJSRvitxYqb1GcEfJYw1b8LPu-MRz?usp=drive_link)

# Python Packages
Tensorflow

Tensorflow-Keras

Pandas

Numpy

Matplotlib

StellarGraph

Scikit-learn

