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
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import random
random.seed(1234)


def get_a_p_r_f(target, predict, category):
    idx = np.array(range(len(target)))
    _target = set(idx[target == category])
    _predict = set(idx[predict == category])
    true = _target & _predict
    true_No = set(idx) - _target - _predict
    accuracy = (len(true) + len(true_No)) / len(idx)
    precision = len(true) / float(len(_predict))
    recall = len(true) / float(len(_target))
    f1_score = precision * recall * 2 / (precision + recall + 0.00001)
    return accuracy, precision, recall, f1_score

def split_data(feature, target, split_rate=0.2):
    num = len(feature)
    idx = range(num)
    random.shuffle(idx)
    test_feature = feature[idx[:int(num*split_rate)]]
    test_target = target[idx[:int(num*split_rate)]]
    dev_feature = feature[idx[int(num*split_rate): int(num*2*split_rate)]]
    dev_target = target[idx[int(num*split_rate): int(num*2*split_rate)]]
    train_feature = feature[idx[int(num*2*split_rate):]]
    train_target = target[idx[int(num*2*split_rate):]]
    print('splited successfully!')
    return train_feature, train_target, dev_feature, dev_target, test_feature, test_target
    

def get_embedding_matrix(embedding_path, word_index, max_features, embedding_dims):
    print('Preparing embedding matrix')

    # nb_words = min(max_features, len(word_index)) + 1
    nb_words = len(word_index) + 1
    # embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    numpy_rng = np.random.RandomState(4321)
    embedding_matrix = numpy_rng.uniform(low=-0.05, high=0.05, size=(nb_words, embedding_dims))
    embeddings_from_file = {}
    miss_count = 0
    with codecs.open(embedding_path, encoding='utf8') as embedding_file:
        embedding_file.readline()
        while True:
            try:
                line = embedding_file.readline()
                if not line:
                    break
                fields = line.strip().split(' ')
                word = fields[0]
                vector = np.array(fields[1:], dtype='float32')
                embeddings_from_file[word] = vector
            except:
                miss_count += 1
            # print(line)

    print('num of embedding file is ', len(embeddings_from_file))
    count = 0
    for word, i in word_index.iteritems():
        if word in embeddings_from_file:
            embedding_matrix[i] = embeddings_from_file[word]
            count += 1
        # else:
            # print(word)
    print('miss word embedding is', miss_count)
    print('nb words is', nb_words)
    print('num of word embeddings:', count)
    print('Null word embeddings: %d' % (nb_words - count))

    return embedding_matrix

def convert_sequence(text, filter_json=None):
    sequence = text_to_word_sequence(text)
    if filter_json == None:
        return ' '.join(sequence)
    new_sequence = [term for term in sequence if term in filter_json]
    return ' '.join(new_sequence)

def get_all_para(file_path, save_path):
    fw = open(save_path, 'a')
    with codecs.open(file_path, encoding='utf8') as fp:
        while True:
            line = fp.readline()
            if not line:
                fw.close()
                return
            text = line.strip().split('\t')[-1]
            text = convert_sequence(text.encode('utf8'))
            fw.write(text+' ')
        
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

def get_padded_sequences(sequences, maxlen=None):
    if maxlen == None:
        return pad_sequences(sequences, maxlen=100)
    else:
        return pad_sequences(sequences, maxlen)

def get_categorical(labels, num_classes=5):
    print("start to categorical...")
    return to_categorical(labels, num_classes)

def func():
    pass

if __name__ == "__main__":
    get_all_para(sys.argv[1], sys.argv[2])
