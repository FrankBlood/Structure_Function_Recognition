#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
BiGruPool
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-9-22下午7:13
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
from keras.layers import Input, Embedding, Dense, LSTM, GRU, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import TimeDistributed, RepeatVector, Permute, Lambda, Bidirectional, Dropout
from keras.layers.merge import concatenate, add, dot, multiply
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import Activation
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.layers.advanced_activations import PReLU

import numpy as np

class BiGruPool(Network):
    def __init__(self):
        Network.__init__(self)

    def build(self, embedding_matrix=np.array([None])):
        print('Build Bidirectional GRU add Pooling model...')
        self.set_name("BiGruPool")

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

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        x = Bidirectional(GRU(self.rnn_dim, return_sequences=True))(embedded_sequences)
        x = Dropout(self.dropout_rate)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(200, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        preds = Dense(self.units, activation='softmax')(x)
        model = Model(inputs=sequence_input, outputs=preds)

        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['acc'])
        model.summary()
        self.model = model


def func():
    network = BiGruPool()
    network.build()


if __name__ == "__main__":
    func()
