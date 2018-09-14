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

# LABEL_INDEX = {'新闻':0,'恐怖':1,'暴力':2,'脏话':3,'自杀':4,'色情':5}
LABEL_INDEX = {'news': 0, 'horror': 1, 'violence': 2, 'dirty_words': 3, 'suicide': 4, 'sex': 5}
# LABEL_INDEX = {'恐怖': 1, '正常': 0}

# 新闻数据接口
URL_0 = 'http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01276/search' \
        '?fields=id,wcaption&filters=EQS_ifCompare,1|EQS_resourceState,4|EQS_newsLabelSecond,' \
        '%E6%97%B6%E6%94%BF%E6%BB%9A%E5%8A%A8&orders=wpubTime_desc' \
        '&pagestart=1&fetchsize=10000'

# 反例数据接口：恐怖
URL_1 = 'http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01344/search' \
        '?fields=id,wcaption&filters=EQS_resourceState,4' \
        '|EQS_newsLabel,%E6%81%90%E6%80%96&pagestart=1&fetchsize=10000'

# 反例数据接口：色情
URL_5 = 'http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01344/search' \
        '?fields=id,wcaption&filters=EQS_resourceState,4' \
        '|EQS_newsLabel,%E8%89%B2%E6%83%85&pagestart=1&fetchsize=10000'

# 数据保存地址
BASE_DIR = '/data0/search/textcnn/data/'
DATASET = os.path.join(BASE_DIR, 'dataset/')  # 数据集文件夹
DATA_0 = os.path.join(BASE_DIR, 'dataset/news/data_0.txt')  # 新闻语料
DATA_1 = os.path.join(BASE_DIR, 'dataset/horror/data_1.txt')  # 反例恐怖语料
DATA_5 = os.path.join(BASE_DIR, 'dataset/sex/data_5.txt')  # 反例色情语料
SEG_DATA = os.path.join(BASE_DIR, 'seg_data.csv')  # 分词后数据
SEG_DATA_SHUFFLED = os.path.join(BASE_DIR, 'seg_data_shuffled.csv')  # 分词后数据
WORD_INDEX = os.path.join(BASE_DIR, 'word_index.npy')  # word_index
NUMERIC_DATA = os.path.join(BASE_DIR, 'numeric_data.csv')  # 序号化后数据
WORD2VEC = os.path.join(BASE_DIR, 'sgns.merge.bigram')  # word2vec词典地址
EMBEDDING_MATRIX = os.path.join(BASE_DIR, 'embedding_matrix.npy')  # embedding_matrix

# 超参
EMBEDDING_DIM = 300  # 词向量维数
MAX_NUM_WORDS = 150000  # 词典最大词数，若语料中含词数超过该数，则取前MAX_NUM_WORDS个

# 分隔符
SEG_SPLITTER = ' '
# CSV_SPLITTER = ','

word_index = {}

# 打开停用词表并做处理
STOP_WORDS_LIST = os.path.join(BASE_DIR, 'stop_list.txt')  # 停用词表
with open(STOP_WORDS_LIST, 'r') as f:
    stop_words = f.readlines()
del stop_words[0]  # 删除txt文件第一行的特殊字符
for word in stop_words:  # 删除每行最后的回车
    stop_words[stop_words.index(word)] = word.replace('\n', '')


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
        get_data_5_from_api()
    print('download and save')

    # 添加标签
    df0 = pd.read_csv(DATA_0, header=None, names=['doc'])
    df0['label'] = 0
    print('get data 0' + str(len(df0)))
    df1 = pd.read_csv(DATA_1, header=None, names=['doc'])
    df1['label'] = 1
    print('get data 1' + str(len(df1)))
    df5 = pd.read_csv(DATA_5, header=None, names=['doc'])
    df5['label'] = 5
    print('get data 5' + str(len(df5)))
    all_data = df0.append(df1, ignore_index=True).append(df5, ignore_index=True)
    print('all data size=' + str(len(all_data)))

    # all_data = get_labeled_data()
    print('get data, size = '+str(len(all_data)))
    print('set labels')

    # 合并正例反例，分词，打乱顺序，并保存结果至csv:SEG_DATA
    all_data['tokens'] = all_data['doc'].map(segment)
    all_data.to_csv(SEG_DATA, encoding='utf-8')
    print('segmentation and save to ' + SEG_DATA)

    seg_data = all_data[['tokens', 'label']]
    seg_data_shuffled = seg_data.sample(frac=1)
    seg_data_shuffled.to_csv(SEG_DATA_SHUFFLED, encoding='utf-8')
    print('shuffle and save to ' + SEG_DATA_SHUFFLED)

    # 获取word_index，并保存结果至csv:WORD_INDEX
    word_index = get_word_index(seg_data_shuffled)

    # word_index_df = DataFrame(list(word_index.items()), columns=['word', 'index'])
    # word_index_df.to_csv(WORD_INDEX, encoding='utf-8')
    np.save(WORD_INDEX, word_index)
    # 读取
    # word_index = np.load(WORD_INDEX)[()]
    print('getting word_index and save to ' + WORD_INDEX)

    # 语料数字化：将分词列表替换成序号列表，并保存结果至csv:NUMERIC_DATA
    seg_data['indexes'] = seg_data['tokens'].map(word2index)
    numeric_data = seg_data[['indexes', 'label']]
    numeric_data.to_csv(NUMERIC_DATA, encoding='utf-8')
    print('word2index and save to ' + NUMERIC_DATA)

    # 获取embeddings_index（加载预训练好的word2vec词典）
    embeddings_index = get_embeddings_index()
    print('get embeddings_index (or word2vec dict)')

    # 通过获取embeddings_index以及word_index，生成embedding_matrix
    embedding_matrix = generate_embedding_matrix(embeddings_index)
    np.save(EMBEDDING_MATRIX, embedding_matrix)
    print('generate_embedding_matrix and save to' + EMBEDDING_MATRIX)


def get_labeled_data():
    """
    从文档中读取dataframe，并增加标签
    :return:
    """
    all_data = pd.DataFrame()
    for name in sorted(os.listdir(DATASET)):
        path = os.path.join(DATASET, name)
        if name in LABEL_INDEX:
            label = LABEL_INDEX[name]
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                single_data_path = os.path.join(path, fname)
                print(single_data_path)
                df = pd.read_csv(single_data_path, header=None, names=['doc'])[:10]
                df['label'] = label
                all_data = all_data.append(df, ignore_index=True)
    return all_data


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
    seg_origin = SEG_SPLITTER.join(jieba.cut(input_string, cut_all=False))
    seg_origin_list = seg_origin.split(SEG_SPLITTER)
    seg_stop_list = [word for word in seg_origin_list if word not in stop_words]
    return SEG_SPLITTER.join(seg_stop_list)


def get_data_0_from_api():
    """
    通过接口获取新闻数据，并保存至文件
    含有简单文本过滤，并替换CSV文件分隔符：英文逗号
    :return:
    """
    with open(DATA_0, 'w') as f:
        with urllib.request.urlopen(URL_1) as response:
            resp = response.read()
            j1 = json.loads(resp)
            results = j1['value']
            for result in results:
                line = result.get('wcaption') \
                    .replace(',', '，') \
                    .replace('|', '')
                f.write(line + '\n')


def get_data_1_from_api():
    """
    通过接口获得反例恐怖数据，并保存至文件
    含有简单文本过滤，并替换CSV文件分隔符：英文逗号
    :return:
    """
    with open(DATA_1, 'w') as f:
        with urllib.request.urlopen(URL_0) as response:
            resp = response.read()
            j1 = json.loads(resp)
            results = j1['value']
            for result in results:
                line = result.get('wcaption') \
                    .replace(',', '，') \
                    .replace('免费订阅精彩鬼故事，微信号：guidayecom', '')
                f.write(line + '\n')


def get_data_5_from_api():
    """
    通过接口获取正例色情数据，并保存至文件
    含有简单文本过滤，并替换CSV文件分隔符：英文逗号
    :return:
    """
    with open(DATA_5, 'w') as f:
        with urllib.request.urlopen(URL_5) as response:
            resp = response.read()
            j1 = json.loads(resp)
            results = j1['value']
            for result in results:
                line = result.get('wcaption') \
                    .replace(',', '，')
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


if __name__ == '__main__':
    pre_process(skip_download=True)
