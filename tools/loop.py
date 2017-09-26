# -*- coding:utf8 -*-
import codecs
import json
with open('bbb.dict', 'r') as fp:
    dict_data = json.load(fp)
print(len(dict_data))
while True:
    a = raw_input('Please enter a word: ')
    if a == "EXIT":
        break
    else:
        try:
            print dict_data[a]
        except:
            print('key error')
