#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : pureoym
# @Contact : pureoym@163.com
# @TIME    : 2018/9/14 9:42
# @File    : utils.py
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
import os
import jieba
import numpy as np

# 打开停用词表并做处理
BASE_DIR = '/data0/search/textcnn/data/'
STOP_WORDS_LIST = os.path.join(BASE_DIR, 'stop_list.txt')  # 停用词表
WORD_INDEX = os.path.join(BASE_DIR, 'word_index.npy')  # word_index
with open(STOP_WORDS_LIST, 'r') as f:
    stop_words = f.readlines()
del stop_words[0]  # 删除txt文件第一行的特殊字符
for word in stop_words:  # 删除每行最后的回车
    stop_words[stop_words.index(word)] = word.replace('\n', '')

SEG_SPLITTER = ' '
word_index = np.load(WORD_INDEX)[()]
MAX_NUM_WORDS = 150000


def get_index_sequence_from_text(text):
    """
    获取词序列
    :param text:文本
    :return: 词序列
    """
    # 分词
    tokens = segment(text)
    # 转换成词序号序列
    indexes = word2index(tokens)
    return indexes


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
