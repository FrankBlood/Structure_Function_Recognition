#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
BasicCNN
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

class BasicCNN(Network):
    def __init__(self):
        Network.__init__(self)
        self.filters = 250
        self.kernel_size = 3

    def build(self, embedding_matrix=None):
        print('Build basic CNN model...')
        self.set_name("BasicCNN")

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

        cnn1 = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(embedded_sequences)
        cnn1 = MaxPooling1D()(cnn1)
        cnn1 = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(cnn1)
        cnn1 = Dropout(self.dropout_rate)(cnn1)
        cnn1 = BatchNormalization()(cnn1)
        # cnn1 = Permute([2, 1])(cnn1)
        cnn1 = MaxPooling1D()(cnn1)
        cnn1 = Dense(200, activation='relu')(cnn1)
        cnn1 = Dropout(self.dropout_rate)(cnn1)
        cnn1 = BatchNormalization()(cnn1)
        cnn1 = Flatten()(cnn1)

        cnn2 = Conv1D(filters=self.filters, kernel_size=self.kernel_size-1, activation='relu')(embedded_sequences)
        cnn2 = MaxPooling1D()(cnn2)
        cnn2 = Conv1D(filters=self.filters, kernel_size=self.kernel_size-1, activation='relu')(cnn2)
        cnn2 = Dropout(self.dropout_rate)(cnn2)
        cnn2 = BatchNormalization()(cnn2)
        # cnn2 = Permute([2, 1])(cnn2)
        cnn2 = MaxPooling1D()(cnn2)
        cnn2 = Dense(200, activation='relu')(cnn2)
        cnn2 = Dropout(self.dropout_rate)(cnn2)
        cnn2 = BatchNormalization()(cnn2)
        cnn2 = Flatten()(cnn2)

        cnn3 = Conv1D(filters=self.filters, kernel_size=self.kernel_size-2, activation='relu')(embedded_sequences)
        cnn3 = MaxPooling1D()(cnn3)
        cnn3 = Conv1D(filters=self.filters, kernel_size=self.kernel_size-2, activation='relu')(cnn3)
        cnn3 = Dropout(self.dropout_rate)(cnn3)
        cnn3 = BatchNormalization()(cnn3)
        # cnn3 = Permute([2, 1])(cnn3)
        cnn3 = MaxPooling1D()(cnn3)
        cnn3 = Dense(200, activation='relu')(cnn3)
        cnn3 = Dropout(self.dropout_rate)(cnn3)
        cnn3 = BatchNormalization()(cnn3)
        cnn3 = Flatten()(cnn3)

        cnn = concatenate([cnn1, cnn2, cnn3])
        cnn = Dense(300, activation='relu')(cnn)
        cnn = Dropout(self.dropout_rate)(cnn)

        preds = Dense(self.units, activation='softmax')(cnn)
        model = Model(inputs=sequence_input, outputs=preds)

        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['acc'])
        model.summary()
        self.model = model


def func():
    network = BasicCNN()
    network.build()


if __name__ == "__main__":
    func()
