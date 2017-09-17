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

def get_data():
    file_path = sys.argv[1]
    data_loader = Data_Loader()
    paras, labels = data_loader.get_para_label(file_path)
    tokenizer = Tokenizer(num_words=200000)
    tokenizer.fit_on_texts(paras)
    sequences = tokenizer.texts_to_sequences(paras)
    paded_sequences = pad_sequences(sequences, maxlen=200)
    return paded_sequences, get_categorical(np.array(labels))

if __name__ == "__main__":
    data, label = get_data()
    print(data)
    print(label)
