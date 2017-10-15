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

from tools.get_data import get_data
from models.BiLSTM import BiLSTM
from models.BiGRU import BiGRU
from models.BasicCNN import BasicCNN
from models.BiLstmPool import BiLstmPool
from models.BiGruPool import BiGruPool
from models.BiGruConv import BiGruConv
from models.BiLstmConv import BiLstmConv
from models.BiLstmConv3 import BiLstmConv3
from models.CnnPooling import CnnPooling
from models.CnnLstmPooling import CnnLstmPooling
from models.CnnBasedGru import CnnBasedGru
from models.GruInnerGru import GruInnerGru
from tools.utils import get_embedding_matrix, split_data, to_categorical, get_a_p_r_f

import codecs
import json
import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def train():
    
    config_path = sys.argv[1]
    with codecs.open(config_path, encoding='utf8') as fp:
        config = json.loads(fp.read().strip())
    
    model_name = config['model_name']
    print(model_name)
    if model_name == 'BiGRU':
        network = BiGRU()
    elif model_name == 'BiLSTM':
        network = BiLSTM()
    elif model_name == 'BasicCNN':
        network = BasicCNN()
    elif model_name == 'BiLstmPool':
        network = BiLstmPool()
    elif model_name == 'BiGruPool':
        network = BiGruPool()
    elif model_name == 'BiGruConv':
        network = BiGruConv()
    elif model_name == 'BiLstmConv':
        network = BiLstmConv()
    elif model_name == 'BiLstmConv3':
        network = BiLstmConv3()
    elif model_name == 'CnnPooling':
        network = CnnPooling()
    elif model_name == 'CnnLstmPooling':
        network = CnnLstmPooling()
    elif model_name == 'CnnBasedGru':
        network = CnnBasedGru()
    elif model_name == 'GruInnerGru':
        network = GruInnerGru()
    else:
        print("What the F**K!")
        return

    with open(config['word_index_path'], 'r') as fp:
        word_index = json.load(fp)

    with open(config['filter_json_path'], 'r') as fp:
        filter_json = json.load(fp)

    network.nb_words = min(len(word_index), config['network_config']['num_words'])+1
    # network.set_from_json(config['network_config'])

    network.set_optimizer(optimizer_name=config['network_config']['optimizer_name'], lr=config['network_config']['lr'])

    embedding_matrix = get_embedding_matrix(config['embedding_path'],
                                            word_index,
                                            max_features=config['network_config']['num_words'],
                                            embedding_dims=config['network_config']['embedding_dims'])
    
    network.build(embedding_matrix)

    paded_sequences, labels, _, _ = get_data(config['data_path'],
                                             filter_json = filter_json,
                                             num_words=config['network_config']['num_words'],
                                             maxlen=config['network_config']['maxlen'])
    categ_labels = to_categorical(labels)

    # Data splite
    train_feature, train_target, dev_feature, dev_target, test_feature, test_target = split_data(paded_sequences, categ_labels, 0.1)

    # train model with training data and evaluate with development data
    bst_model_path = network.train(train_feature, train_target, dev_feature, dev_target)

    network.load_model(bst_model_path)

    # test model with test model
    loss, accu = network.evaluate(test_feature, test_target, batch_size=512)
    print('The total loss is %s and the total accuracy is %s'
          % (loss, accu))

    # get the accuracy, precision, recall and f1score

    # get the predicted label
    y = network.inference(test_feature, batch_size=512)
    # get the label from softmax
    y_class = np.array(map(np.argmax, y))
    # get the label from target
    target_class = np.array(map(np.argmax, test_target))
    for category in range(5):
        accuracy, precision, recall, f1score = get_a_p_r_f(target_class, y_class, category)
        print('For category %s: the accuracy is %s, the precision is %s, the recall is %s and the f1score is %s'
              % (category, accuracy, precision, recall, f1score))

if __name__ == "__main__":
    train()
