#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
CnnBasedGru
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-9-20上午9:59
@copyright: "Copyright (c) 2017 Guoxiu He. All Rights Reserved"
"""

from __future__ import print_function
from __future__ import division

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

from Network import Network

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import TimeDistributed, RepeatVector, Permute, Lambda, Bidirectional, Dropout, Flatten
from keras.layers.merge import concatenate, add, dot, multiply
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import Activation
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.layers.advanced_activations import PReLU
import numpy as np

class CnnBasedGru(Network):
    def __init__(self):
        Network.__init__(self)
        self.filters = 250
        self.kernel_size = 3

    def build(self, embedding_matrix=np.array([None])):
        print('Build CNN Based GRU model...')
        self.set_name("CnnBasedGru")

        if embedding_matrix.any() == None:
            # # embedding_matrix = np.zeros((config.max_features, config.embedding_dims))
            # numpy_rng = np.random.RandomState(4321)
            # embedding_matrix = numpy_rng.uniform(low=-0.05, high=0.05, size=(config.max_features, config.embedding_dims))
            embedding_layer = Embedding(self.nb_words,
                                        self.embedding_dims,
                                        input_length=self.maxlen)

        else:
            embedding_layer = Embedding(self.nb_words,
                                        self.embedding_dims,
                                        weights=[embedding_matrix],
                                        input_length=self.maxlen,
                                        trainable=False)

        ########################################
        ## All Used Layers
        ########################################
        rnn_layer = Bidirectional(GRU(self.rnn_dim,
                                      dropout=self.dropout_rate,
                                      recurrent_dropout=self.dropout_rate))

        cnn_layer = Conv1D(activation="relu",
                           padding="valid",
                           strides=1,
                           filters=self.filters,
                           kernel_size=self.kernel_size)

        pooling_layer = GlobalMaxPooling1D()
        cnn_dense = Dense(self.rnn_dim, activation='relu')
        cnn_dropout1 = Dropout(self.dropout_rate)
        cnn_dropout2 = Dropout(self.dropout_rate)
        cnn_batchnormalization = BatchNormalization()
        cnn_dense1 = Dense(self.rnn_dim, activation='tanh')

        ########################################
        ## Input and Embedding
        ########################################
        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        ########################################
        ## Attention Layer: CNN
        ########################################
        cnn = cnn_layer(embedded_sequences)
        cnn = pooling_layer(cnn)
        cnn = cnn_dropout1(cnn)
        cnn = cnn_dense(cnn)
        cnn = cnn_dropout2(cnn)
        cnn = cnn_batchnormalization(cnn)

        ########################################
        ## Attention Action
        ########################################
        cnn_t = cnn_dense1(cnn)
        a = multiply([cnn_t, embedded_sequences])
        a = Permute([2, 1])(a)
        a = Lambda(lambda x: K.sum(x, axis=1))(a)
        a = Activation('sigmoid')(a)
        embedded_sequences = Permute([2, 1])(embedded_sequences)
        x = multiply([a, embedded_sequences])
        x = Permute([2, 1])(x)

        ########################################
        ## Output Layers
        ########################################
        x = rnn_layer(x)
        x = Dense(self.rnn_dim, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = BatchNormalization()(x)

        preds = Dense(self.units, activation='softmax')(x)

        ########################################
        ## train the model
        ########################################
        model = Model(inputs=sequence_input, outputs=preds)
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['acc'])
        model.summary()
        self.model = model


def func():
    network = CnnBasedGru()
    network.rnn_dim = 200
    network.build()


if __name__ == "__main__":
    func()
