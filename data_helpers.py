#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : pureoym
# @Contact : pureoym@163.com
# @TIME    : 2018/9/7 13:56
# @File    : data_helpers.py
# Copyright 2017 pureoym. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import urllib
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import json
import jieba

# label_index = {'恐怖':1,'暴力':2,'脏话':3,'自杀':4,'色情':5}
label_index = {'恐怖': 1, '正常': 0}

# 正例接口
URL_1 = 'http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01276/search' \
        '?fields=id,wcaption&filters=EQS_ifCompare,1|EQS_resourceState,4|EQS_newsLabelSecond,' \
        '%E6%97%B6%E6%94%BF%E6%BB%9A%E5%8A%A8&orders=wpubTime_desc' \
        '&pagestart=1&fetchsize=10000'

# 反例接口
URL_0 = 'http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01344/search' \
        '?fields=id,wcaption&filters=EQS_resourceState,4' \
        '|EQS_newsLabel,%E6%81%90%E6%80%96&pagestart=1&fetchsize=10000'

DATA_1 = '/application/search/ming/data_1.txt'
DATA_0 = '/application/search/ming/data_0.txt'
DATA_1_SEG = '/application/search/ming/data_1_seg.txt'
DATA_0_SEG = '/application/search/ming/data_0_seg.txt'


def get1():
    x1 = []
    with urllib.request.urlopen(URL_1) as response:
        resp = response.read()
        j1 = json.loads(resp)
        results = j1['value']
        for result in results:
            x = result.get('wcaption').replace(',', '，')
            x1.append(x)
    df1 = DataFrame(columns=['doc'], data=x1)
    df1['label'] = 1
    return df1


def get0():
    x0 = []
    with urllib.request.urlopen(URL_0) as response:
        resp = response.read()
        j0 = json.loads(resp)
        results = j0['value']
        for result in results:
            x = result.get('wcaption').replace(',', '，')
            x0.append(x)
    df0 = DataFrame(columns=['doc'], data=x0)
    df0['label'] = 0
    return df0


def segment(df):
    df['tokens'] = df['doc'].map(lambda x: ' '.join(jieba.cut(x, cut_all=False)))
    return df[['tokens', 'label']]


def test_analyse():
    from collections import Counter
    import jieba.analyse
    import time

    bill_path = r'bill.txt'
    bill_result_path = r'bill_result.txt'
    car_path = 'car.txt'
    with open(bill_path, 'r') as fr:
        data = jieba.cut(fr.read())
    data = dict(Counter(data))
    with open(bill_result_path, 'w') as fw:
        for k, v in data.items():
            fw.write("%s,%d\n" % (k.encode('utf-8'), v))


def get_data_0():
    """
    通过接口获得反例数据，并保存至文件
    :return:
    """
    with open(DATA_0, 'w') as f:
        with urllib.request.urlopen(URL_0) as response:
            resp = response.read()
            j1 = json.loads(resp)
            results = j1['value']
            for result in results:
                x = result.get('wcaption').replace(',', '，').replace('免费订阅精彩鬼故事，微信号：guidayecom', '')
                f.write(x + '\n')


def get_data_1():
    """
    通过接口获取正例数据，并保存至文件
    :return:
    """
    with open(DATA_1, 'w') as f:
        with urllib.request.urlopen(URL_1) as response:
            resp = response.read()
            j1 = json.loads(resp)
            results = j1['value']
            for result in results:
                x = result.get('wcaption').replace(',', '，')
                f.write(x + '\n')


def get_data_0_seg():
    with open(DATA_0, 'r') as f1:
        with open(DATA_0_SEG, 'w') as f2:
            lines = f1.readlines()
            for line in lines:
                output_line = ' '.join(jieba.cut(line, cut_all=False))
                f2.write(output_line + '\n')


def test_get_tfidf():
    import jieba
    import jieba.analyse

    # text = "故宫的著名景点包括乾清宫、太和殿和午门等。其中乾清宫非常精美，午门是紫禁城的正门，午门居中向阳。"
    text = ''
    # jieba.load_userdict("jieba_dict.txt")  # 用户自定义词典 （用户可以自己在这个文本文件中，写好自定制词汇）
    f = open('/application/search/ming/data_0_small.txt', 'r', encoding='utf8')  # 要进行分词处理的文本文件 (统统按照utf8文件去处理，省得麻烦)
    lines = f.readlines()
    for line in lines:
        text += line

    # seg_list = jieba.cut(text, cut_all=False)  #精确模式（默认是精确模式）
    seg_list = jieba.cut(text, cut_all=False) # 精确模式（默认是精确模式）
    # print("[精确模式]: ", "/ ".join(seg_list))

    # seg_list2 = jieba.cut(text, cut_all=True)    #全模式
    # print("[全模式]: ", "/ ".join(seg_list2))

    # seg_list3 = jieba.cut_for_search(text)    #搜索引擎模式
    # print("[搜索引擎模式]: ","/ ".join(seg_list3))

    tags = jieba.analyse.extract_tags(text, topK=100, withWeight=True)
    # print("关键词:    ", " / ".join(tags))
    return tags


def get_word_index(d0,d1):
    word_index={}
    for tokens in d0['tokens']:
        words=tokens.split(' ')
        for word in words:
            if word in word_index.keys():
                count = word_index[word]
                word_index[word] = count+1
            else:
                word_index[word] = 1
    for tokens in d1['tokens']:
        words=tokens.split(' ')
        for word in words:
            if word in word_index.keys():
                count = word_index[word]
                word_index[word] = count+1
            else:
                word_index[word] = 1
    word_index = sorted(word_index.items(), key=lambda x: x[1], reverse=True)
    return word_index


def pre_process():
    d1 = pd.read_csv(DATA_1, header=None, names=['doc'])
    d1['label'] = 1
    d1['tokens'] = d1['doc'].map(lambda x: ' '.join(jieba.cut(x, cut_all=False)))

    d0 = pd.read_csv(DATA_1, header=None, names=['doc'])
    d0['label'] = 0
    d0['tokens'] = d0['doc'].map(lambda x: ' '.join(jieba.cut(x, cut_all=False)))



if __name__ == '__main__':
    d1 = pd.read_csv(DATA_1, header=None, names=['doc'])
    d1['label'] = 1
    d1['tokens'] = d1['doc'].map(lambda x: ' '.join(jieba.cut(x, cut_all=False)))

    d0 = pd.read_csv(DATA_1, header=None, names=['doc'])
    d0['label'] = 0
    d0['tokens'] = d0['doc'].map(lambda x: ' '.join(jieba.cut(x, cut_all=False)))

    x1 = get1()
    x1[['doc']].to_csv('/application/search/ming/x1.csv', encoding='utf-8')
    x1_seg = segment(x1)
    x1_seg.to_csv('/application/search/ming/x1_seg.csv', encoding='utf-8')
