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
sys.path.insert(0, os.path.dirname(curdir))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, LSTM, GRU, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import TimeDistributed, RepeatVector, Permute, Lambda, Bidirectional, Dropout
from keras.layers.merge import concatenate, add, dot, multiply
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import Activation
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.layers.advanced_activations import PReLU

class Network(object):
    def __init__(self):
        self.nb_words = 200000
        self.embedding_dims = 200
        self.maxlen = 200
        self.rnn_dims = 200
        self.dropout_rate = 0.5

    def bidirectional_lstm(self, embedding_matrix=None):
        print('Build Bidirectional LSTM model...')

        if embedding_matrix == None:
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

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        x = Bidirectional(LSTM(self.rnn_dims))(embedded_sequences)
        x = Dropout(self.dropout_rate)(x)
        preds = Dense(5, activation='softmax')(x)
        model = Model(inputs=sequence_input, outputs=preds)

        model.compile(loss='categorical_crossentropy',
                      optimizer='nadam',
                      metrics=['acc'])
        model.summary()
        return model
        
def func():
    pass


if __name__ == "__main__":
    pass
