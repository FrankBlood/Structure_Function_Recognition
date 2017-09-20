#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Data_Loader
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

from utils import *
import codecs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Data_Loader(object):
    def __init__(self):
        print("Data loading...")

    def get_para_label(self, file_path):
        paras = []
        labels = []
        with codecs.open(file_path, encoding='utf8') as fp:
            while True:
                line = fp.readline()
                if not line:
                    return paras, labels
                tmp = line.strip().split('\t')[-2:]
                label, para = int(tmp[0]), tmp[1]
                # paras.append(para.strip().encode('utf8'))
                paras.append(convert_sequence(para.strip().encode('utf8')))
                labels.append(label)
    
    def get_section_label(self, file_path):
        # TODO: section
        pass

def func():
    file_path = sys.argv[1]
    data_loader = Data_Loader()
    paras, labels = data_loader.get_para_label(file_path)
    tokenizer = Tokenizer(nb_words=200000)
    print(type(paras))
    print(len(paras))
    tokenizer.fit_on_texts(paras[:10])
    sequences = tokenizer.texts_to_sequences(paras[:10])
    paded_sequence = pad_sequences(sequences, maxlen=400)
    print(paded_sequence)
    for para in paras:
        print(para)
        break

if __name__ == "__main__":
    func()
