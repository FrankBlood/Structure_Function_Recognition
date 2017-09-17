#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
train
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-9-16下午8:11
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

from tools.utils import *
from tools.get_data import get_data
from models.Network import Network
from keras.callbacks import EarlyStopping, ModelCheckpoint
import codecs
import time

def train():
    network = Network()
    model = network.bidirectional_lstm()
    model_name = 'BidirectionalLstm'
    early_stopping = EarlyStopping(monitor='val_acc', patience=3)
    now_time = '_'.join(time.asctime(time.localtime(time.time())).split(' '))
    bst_model_path = './models/save/' + model_name+ '_' + now_time + '.h5'
    print('bst_model_path:', bst_model_path)
    model_checkpoint = ModelCheckpoint(bst_model_path, monitor='val_acc', save_best_only=True, save_weights_only=True)
    
    paded_sequences, labels = get_data()

    model.fit(paded_sequences, labels,
              batch_size = 50,
              nb_epoch=10, shuffle=True,
              validation_split = 0.2,
              # callbacks=[model_checkpoint])
              callbacks=[early_stopping, model_checkpoint])

    if os.path.exists(bst_model_path):
        model.load_weights(bst_model_path)

    # print('test:', model.evaluate(X_test, y_test, batch_size=config.batch_size))

if __name__ == "__main__":
    train()
