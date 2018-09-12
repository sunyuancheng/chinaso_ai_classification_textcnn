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
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import json
import jieba
import os

# LABEL_INDEX = {'恐怖':1,'暴力':2,'脏话':3,'自杀':4,'色情':5}
LABEL_INDEX = {'恐怖': 1, '正常': 0}

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
BASE_DIR = '/data0/search/textcnn/data/'
DATA_1 = os.path.join(BASE_DIR, 'data_1.txt')  # 正例语料
DATA_0 = os.path.join(BASE_DIR, 'data_0.txt')  # 反例语料
SEG_DATA = os.path.join(BASE_DIR, 'seg_data.csv')  # 分词后数据
WORD_INDEX = os.path.join(BASE_DIR, 'word_index.csv')  # word_index
NUMERIC_DATA = os.path.join(BASE_DIR, 'numeric_data.csv')  # 序号化后数据
WORD2VEC = os.path.join(BASE_DIR, 'sgns.merge.bigram')  # word2vec词典地址
EMBEDDING_MATRIX = os.path.join(BASE_DIR, 'embedding_matrix.npy')  # embedding_matrix

# 超参
EMBEDDING_DIM = 300  # 词向量维数
MAX_NUM_WORDS = 157000  # 词典最大词数，若语料中含词数超过该数，则取前MAX_NUM_WORDS个

# 分隔符
SEG_SPLITTER = ' '
# CSV_SPLITTER = ','

word_index = {}


def pre_process(skip_download=False):
    """
    数据预处理。具体步骤如下：
    1 如果需要下载，通过接口下载数据，并保存结果至csv
    2 处理正例反例数据，添加标签
    3 合并正例反例，分词，打乱顺序，并保存结果至csv:SEG_DATA
    4 获取word_index，并保存结果至csv:WORD_INDEX
    5 语料数字化：将分词列表替换成序号列表，并保存结果至csv:NUMERIC_DATA
    6 获取embeddings_index（加载预训练好的word2vec词典）
    7 通过获取embeddings_index以及word_index，生成embedding_matrix
    :param skip_download: 是否跳过数据下载
    :return:
    """
    # 如果需要下载，通过接口下载数据，并保存结果至csv
    if not skip_download:
        get_data_0_from_api()
        get_data_1_from_api()
        print('download and save to ' + DATA_0 + ',' + DATA_1)

    # 处理正例数据，添加标签
    df1 = pd.read_csv(DATA_1, header=None, names=['doc'])
    df1['label'] = 1

    # 处理反例数据，添加标签
    df0 = pd.read_csv(DATA_0, header=None, names=['doc'])
    df0['label'] = 0

    # 合并正例反例，分词，打乱顺序，并保存结果至csv:SEG_DATA
    df10 = df1.append(df0, ignore_index=True)
    df10['tokens'] = df10['doc'].map(segment)
    seg_data = df10[['tokens', 'label']]
    seg_data = seg_data.sample(frac=1)
    seg_data.to_csv(SEG_DATA, encoding='utf-8')
    print('segmentation and save to ' + SEG_DATA)

    # 获取word_index，并保存结果至csv:WORD_INDEX
    word_index = get_word_index(seg_data)
    word_index_df = DataFrame(list(word_index.items()), columns=['word', 'index'])
    word_index_df.to_csv(WORD_INDEX, encoding='utf-8')
    print('getting word_index and save to ' + WORD_INDEX)

    # 语料数字化：将分词列表替换成序号列表，并保存结果至csv:NUMERIC_DATA
    seg_data['indexes'] = seg_data['tokens'].map(word2index)
    numeric_data = seg_data[['indexes', 'label']]
    numeric_data.to_csv(NUMERIC_DATA, encoding='utf-8')
    print('word2index and save to ' + NUMERIC_DATA)

    # 获取embeddings_index（加载预训练好的word2vec词典）
    embeddings_index = get_embeddings_index()

    # 通过获取embeddings_index以及word_index，生成embedding_matrix
    embedding_matrix = generate_embedding_matrix(embeddings_index)
    np.save(EMBEDDING_MATRIX, embedding_matrix)


def get_embeddings_index():
    """
    加载预训练word2vec模型，返回字典embeddings_index
    :return: embeddings_index
    """
    embeddings_index = {}
    with open(WORD2VEC) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def generate_embedding_matrix(embeddings_index):
    """
    prepare embedding matrix
    使用embeddings_index，word_index生成预训练矩阵embedding_matrix。
    :param embeddings_index:
    :param word_index:
    :return:
    """
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# def word2index(tokens):
#     """
#     将输入的tokens转换成word_index中的序号
#     :param tokens:
#     :return:
#     """
#     word_list = tokens.split(SEG_SPLITTER)
#     indexes = []
#     for word in word_list:
#         if word is not None:
#             if word in word_index.keys():
#                 indexes.append(str(word_index[word]))
#     return SEG_SPLITTER.join(indexes).strip()

def word2index(tokens):
    """
    将输入的tokens转换成word_index中的序号
    :param tokens:
    :return:
    """
    word_list = tokens.split(SEG_SPLITTER)
    indexes = []
    for word in word_list:
        if word is not None:
            if word in word_index.keys():
                index = word_index[word]
                if index > MAX_NUM_WORDS:
                    indexes.append('0')
                else:
                    indexes.append(str(index))
    return SEG_SPLITTER.join(indexes).strip()


def segment(input_string):
    """
    分词
    :param input_string:
    :return:
    """
    return SEG_SPLITTER.join(jieba.cut(input_string, cut_all=False))


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
                line = result.get('wcaption') \
                    .replace(',', '，') \
                    .replace('免费订阅精彩鬼故事，微信号：guidayecom', '')
                f.write(line + '\n')


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
                line = result.get('wcaption') \
                    .replace(',', '，') \
                    .replace('|', '')
                f.write(line + '\n')


def get_word_index(df):
    """
    统计语料分词词典，按照词频由大到小排序
    :param d0:
    :param d1:
    :return:
    """
    word_dict = {}
    for tokens in df['tokens']:
        words = tokens.split(SEG_SPLITTER)
        for word in words:
            if word in word_dict.keys():
                count = word_dict[word]
                word_dict[word] = count + 1
            else:
                word_dict[word] = 1
    word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    word_dict = word_dict[1:]  # 去除空字符
    for i, word in enumerate(word_dict):
        w = word[0]
        word_index[w] = i + 1
    return word_index


def test_jieba_extract_tags():
    """
    测试jieba生成tfidf关键词
    :return:
    """
    import jieba
    import jieba.analyse

    # text = "故宫的著名景点包括乾清宫、太和殿和午门等。其中乾清宫非常精美，午门是紫禁城的正门，午门居中向阳。"
    text = ''
    # jieba.load_userdict("jieba_dict.txt")  # 用户自定义词典 （用户可以自己在这个文本文件中，写好自定制词汇）
    f = open(DATA_0, 'r', encoding='utf8')  # 要进行分词处理的文本文件 (统统按照utf8文件去处理，省得麻烦)
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


if __name__ == '__main__':
    pre_process(skip_download=True)
