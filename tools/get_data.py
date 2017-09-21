#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
get data
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

from utils import get_categorical
from Data_Loader import Data_Loader
import codecs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

def get_data(file_path, num_words=None, maxlen=150, filter_json=None):
    data_loader = Data_Loader()
    paras, labels = data_loader.get_para_label(file_path, filter_json)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(paras)
    sequences = tokenizer.texts_to_sequences(paras)
    paded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return paded_sequences, get_categorical(np.array(labels)), tokenizer.word_index, tokenizer.word_counts

if __name__ == "__main__":
    data, label, word_index, word_counts = get_data(sys.argv[1], num_words=260000)
    # print(data)
    # print(label)
    with open('../data/word_counts.json', 'w') as fw:
        fw.write(json.dumps(word_counts))
    # with open('aaa.dict', 'w') as fw:
    #     fw.write(json.dumps(word_index))
