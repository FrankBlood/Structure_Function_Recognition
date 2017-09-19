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
from tools.utils import get_embedding_matrix

import codecs
import json

def train():
    
    config_path = sys.argv[1]
    config = {}
    with codecs.open(config_path, encoding='utf8') as fp:
        config = json.loads(fp.read().strip())
    
    model_name = config['model_name']
    print(model_name)
    if model_name == 'BiGRU':
        network = BiGRU()
    elif model_name == 'BiLSTM':
        network = BiLSTM()
    else:
        print("What the FUCK!")
        return
    
    paded_sequences, labels, word_index = get_data(config['data_path'])

    embedding_matrix = get_embedding_matrix(config['embedding_path'], word_index, max_features=2000000, embedding_dims=200)
    
    network.build(embedding_matrix)

    network.train(paded_sequences, labels)

if __name__ == "__main__":
    train()
