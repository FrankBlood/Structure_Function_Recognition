#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Network
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-9-16下午8:10
@copyright: "Copyright (c) 2017 Guoxiu He. All Rights Reserved"
"""

from __future__ import print_function
from __future__ import division

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(curdir)
sys.path.insert(0, os.path.dirname(curdir))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import time
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, LSTM, GRU, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import TimeDistributed, RepeatVector, Permute, Lambda, Bidirectional, Dropout
from keras.layers.merge import concatenate, add, dot, multiply
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import Activation
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.layers.advanced_activations import PReLU

import numpy as np
import json

class Network(object):
    def __init__(self, maxlen=150, units=5, nb_words=2000000,
                 embedding_dims=200, rnn_dim=256, dropout_rate=0.5,
                 loss='categorical_crossentropy', optimizer='nadam'):
        self.nb_words = nb_words
        self.embedding_dims = embedding_dims
        self.maxlen = maxlen
        self.rnn_dim = rnn_dim
        self.dropout_rate = dropout_rate
        self.units = units
        self.loss = loss
        self.optimizer = optimizer

    def set_from_json(self, network_config):
        # self.embedding_dims = network_config['embedding_dims']
        # self.maxlen = network_config['maxlen']
        self.rnn_dim = network_config['rnn_dim']
        # self.dropout_rate = network_config['dropout_rate']
        # self.units = network_config['units']
        # self.loss = network_config['loss']
        # self.optimizer = network_config['optimizer']
            

    def set_name(self, model_name):
        self.model_name = model_name

    def build(self, embedding_matrix=None):
        self.model = Model()

    def inference(self, x, batch_size=None):
        # self.model.load_weights(filepath=model_path)
        y = self.model.predict(x, batch_size=batch_size)
        # y_class = self.model.predict_classes(x, batch_size=batch_size)
        # y_proba = self.model.predict_proba(x, batch_size=batch_size)
        return y

    def load_model(self, model_path):
        self.model.load_weights(model_path)
        print("successfully loaded!")


    def evaluate(self, test_feature, test_target, batch_size=None):
        # self.model.load_weights(filepath=model_path)
        return self.model.evaluate(test_feature, test_target, batch_size=batch_size)

    def train(self, train_feature, train_target, dev_feature=np.array([None]), dev_target=np.array([None])):
        print('Begin to train...')

        early_stopping = EarlyStopping(monitor='val_acc', patience=3)
        now_time = '_'.join(time.asctime(time.localtime(time.time())).split(' '))
        bst_model_path = curdir + '/save/' + self.model_name + '_' + now_time + '.h5'
        print('bst_model_path:', bst_model_path)
        model_checkpoint = ModelCheckpoint(bst_model_path, monitor='val_acc', save_best_only=True,
                                           save_weights_only=True)

        tb_cb = TensorBoard(log_dir=parentdir+'/tensorboard/'+self.model_name, histogram_freq=1, write_graph=True, write_images=False,
                            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

        if os.path.exists(bst_model_path):
            self.model.load_weights(bst_model_path)

        if dev_feature.any() == None or dev_target.any()==None:
            self.model.fit(train_feature, train_target,
                           batch_size=50,
                           nb_epoch=15, shuffle=True,
                           validation_split=0.2,
                           # callbacks=[model_checkpoint])
                           # callbacks=[early_stopping, model_checkpoint])
                           callbacks=[early_stopping, model_checkpoint, tb_cb])
        else:
            self.model.fit(train_feature, train_target,
                           batch_size=50,
                           nb_epoch=15, shuffle=True,
                           validation_data=(dev_feature, dev_target),
                           # callbacks=[model_checkpoint])
                           # callbacks=[early_stopping, model_checkpoint])
                           callbacks=[early_stopping, model_checkpoint, tb_cb])
        return bst_model_path
           

    def set_optimizer(self, optimizer_name='nadam' ,lr=0.001):
        if optimizer_name == 'sgd':
            self.optimizer = SGD(lr=lr)
        elif optimizer_name == 'rmsprop':
            self.optimizer = RMSprop(lr=lr)
        elif optimizer_name == 'adagrad':
            self.optimizer = Adagrad(lr=lr)
        elif optimizer_name == 'adadelta':
            self.optimizer = Adadelta(lr=lr)
        elif optimizer_name == 'adam':
            self.optimizer = Adam(lr=lr)
        elif optimizer_name == 'adamax':
            self.optimizer = Adamax(lr=lr)
        elif optimizer_name == 'nadam':
            self.optimizer = Nadam(lr=lr)
        else:
            print("What the F**K!")

def func():
    pass


if __name__ == "__main__":
    pass
