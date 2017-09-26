# -*- coding:utf8 -*-

from utils import get_embedding_matrix
from get_data import get_data

import sys
import json

with open('../data/word_index.top.260000.json', 'r') as fp:
    filter_json = json.load(fp)

_, _, word_index, word_count = get_data(sys.argv[1], filter_json=filter_json)

print('len word_index', len(word_index))

with open('word_index.dict', 'w') as fw:
    fw.write(json.dumps(word_index))

with open('word_count.dict', 'w') as fw:
    fw.write(json.dumps(word_count))

matrix = get_embedding_matrix(sys.argv[2], word_index, max_features=260000, embedding_dims=200)
