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

os.environ['CUDA_VISIBLE_DEVICES']=sys.argv[1]

def train():
    network = BiLSTM()
    network.build()
    
    paded_sequences, labels = get_data()

    network.train(paded_sequences, labels)

if __name__ == "__main__":
    train()
