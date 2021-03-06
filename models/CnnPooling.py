#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
CNNPooling
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

class CnnPooling(Network):
    def __init__(self):
        Network.__init__(self)
        self.filters = 100
        self.kernel_size = 4

    def build(self, embedding_matrix=np.array([None])):
        print('Build CNN Pooling model...')
        self.set_name("CnnPooling")

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

        cnn1 = Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='Valid', activation='relu')(embedded_sequences)
        cnn1 = Dropout(self.dropout_rate)(cnn1)
        cnn1 = BatchNormalization()(cnn1)
        cnn1 = GlobalMaxPooling1D()(cnn1)

        cnn2 = Conv1D(filters=self.filters, kernel_size=self.kernel_size-1, padding='Valid', activation='relu')(embedded_sequences)
        cnn2 = Dropout(self.dropout_rate)(cnn2)
        cnn2 = BatchNormalization()(cnn2)
        cnn2 = GlobalMaxPooling1D()(cnn2)

        cnn3 = Conv1D(filters=self.filters, kernel_size=self.kernel_size-2, activation='relu')(embedded_sequences)
        cnn3 = Dropout(self.dropout_rate)(cnn3)
        cnn3 = BatchNormalization()(cnn3)
        cnn3 = GlobalMaxPooling1D()(cnn3)

        cnn = concatenate([cnn1, cnn2, cnn3])

        preds = Dense(self.units, activation='softmax')(cnn)
        model = Model(inputs=sequence_input, outputs=preds)

        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['acc'])
        model.summary()
        self.model = model


def func():
    network = CnnPooling()
    network.build()


if __name__ == "__main__":
    func()
