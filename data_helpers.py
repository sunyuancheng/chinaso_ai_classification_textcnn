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
import os

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

# 数据保存地址
BASE_DIR = '/application/search/ming/'
DATA_1 = os.path.join(BASE_DIR, 'data_1.txt')
DATA_0 = os.path.join(BASE_DIR, 'data_0.txt')
DATA_1_SEG = os.path.join(BASE_DIR, 'data_1_seg.csv')
DATA_0_SEG = os.path.join(BASE_DIR, 'data_0_seg.csv')
WORD_INDEX = os.path.join(BASE_DIR, 'word_index.csv')


def pre_process():
    """
    数据预处理。具体步骤如下：
    1 通过接口获取数据，保存至文件。
    2 分词，获取词典word_index。
    :return:
    """
    # 通过接口下载数据
    get_data_0_from_api()
    get_data_1_from_api()


    # 处理数据，分词
    d1 = pd.read_csv(DATA_1, header=None, names=['doc'])
    d1['label'] = 1
    d1['tokens'] = d1['doc'].map(lambda x: ' '.join(jieba.cut(x, cut_all=False)))

    d0 = pd.read_csv(DATA_1, header=None, names=['doc'])
    d0['label'] = 0
    d0['tokens'] = d0['doc'].map(lambda x: ' '.join(jieba.cut(x, cut_all=False)))

    # 获取word_index
    word_index = get_word_index(d0, d1)


def segment(df):
    """
    将正文doc分词，保存于tokens。返回tokens,label
    :param df:
    :return:
    """
    df['tokens'] = df['doc'].map(lambda x: ' '.join(jieba.cut(x, cut_all=False)))
    return df[['tokens', 'label']]


def get_data_0_from_api():
    """
    通过接口获得反例数据，并保存至文件
    含有简单文本过滤，并替换CSV文件分隔符：英文逗号
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


def get_data_1_from_api():
    """
    通过接口获取正例数据，并保存至文件
    含有简单文本过滤，并替换CSV文件分隔符：英文逗号
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


def test_get_tfidf():
    """
    测试jieba生成tfidf
    :return:
    """
    import jieba
    import jieba.analyse

    # text = "故宫的著名景点包括乾清宫、太和殿和午门等。其中乾清宫非常精美，午门是紫禁城的正门，午门居中向阳。"
    text = ''
    # jieba.load_userdict("jieba_dict.txt")  # 用户自定义词典 （用户可以自己在这个文本文件中，写好自定制词汇）
    f = open('DATA_0', 'r', encoding='utf8')  # 要进行分词处理的文本文件 (统统按照utf8文件去处理，省得麻烦)
    lines = f.readlines()
    for line in lines:
        text += line

    # seg_list = jieba.cut(text, cut_all=False)  #精确模式（默认是精确模式）
    # seg_list = jieba.cut(text, cut_all=False) # 精确模式（默认是精确模式）
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
    # 通过接口下载数据，保存结果至csv
    get_data_0_from_api()
    get_data_1_from_api()

    # 处理数据，分词，并保存结果至csv
    d1 = pd.read_csv(DATA_1, header=None, names=['doc'])
    d1['label'] = 1
    d1['tokens'] = d1['doc'].map(lambda x: ' '.join(jieba.cut(x, cut_all=False)))
    d1[['tokens','label']].to_csv(DATA_1_SEG, encoding='utf-8')

    d0 = pd.read_csv(DATA_1, header=None, names=['doc'])
    d0['label'] = 0
    d0['tokens'] = d0['doc'].map(lambda x: ' '.join(jieba.cut(x, cut_all=False)))
    d0[['tokens', 'label']].to_csv(DATA_0_SEG, encoding='utf-8')

    # 获取word_index，并保存结果至csv
    word_index = get_word_index(d0, d1)
    df_word_index = DataFrame(word_index,columns=['word','tf'])
    df_word_index = df_word_index[['word']][1:]
    df_word_index.to_csv(WORD_INDEX, encoding='utf-8')



if __name__ == '__main__':
    pre_process()
