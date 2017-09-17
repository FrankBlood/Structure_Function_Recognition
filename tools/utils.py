#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Utils
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

import codecs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical

def convert(line):
    line = line.strip().split()
    return line

def get_len(line):
    return len(line)

def get_tokenizer(text, num_words=None):
    if num_words == None:
        tokenizer = Tokenizer(num_words=200000)
    else:
        tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(text)
    return tokenizer

def get_sequences(tokenizer, text):
    sequences = tokenizer.texts_to_sequences(text)
    return sequences

def pad_sequences(sequences, maxlen=None):
    if maxlen == None:
        return sequence.pad_sequences(sequences, maxlen=100)
    else:
        return sequence.pad_sequences(sequences, maxlen)

def get_categorical(labels, num_classes=5):
    return to_categorical(labels, num_classes)

def func():
    pass

if __name__ == "__main__":
    func()
