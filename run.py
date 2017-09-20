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
from tools.utils import get_embedding_matrix

import codecs
import json

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
    else:
        print("What the FUCK!")
        return

    with open(config['word_index_path'], 'r') as fp:
        word_index = json.load(fp)

    network.nb_words = min(len(word_index), config['network_config']['num_words'])+1

    embedding_matrix = get_embedding_matrix(config['embedding_path'],
                                            word_index,
                                            max_features=config['network_config']['num_words'],
                                            embedding_dims=config['network_config']['embedding_dims'])
    
    network.build(embedding_matrix)

    paded_sequences, labels, _ = get_data(config['data_path'],
                                          num_words=config['network_config']['num_words'],
                                          maxlen=config['network_config']['maxlen'])

    network.train(paded_sequences, labels)

if __name__ == "__main__":
    train()
