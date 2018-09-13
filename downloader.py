#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : pureoym
# @Contact : pureoym@163.com
# @TIME    : 2018/9/13 14:41
# @File    : downloader.py
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
import os
import json

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
        '|EQS_newsLabel,%E8%89%B2%E6%83%85|EQS_newsLabelSecond,色情小说&pagestart=1&fetchsize=10000'

# 数据保存地址
BASE_DIR = '/data0/search/textcnn/data/'
DATA_0 = os.path.join(BASE_DIR, 'data_0.txt')  # 新闻语料
DATA_1 = os.path.join(BASE_DIR, 'data_1.txt')  # 反例恐怖语料
DATA_5 = os.path.join(BASE_DIR, 'data_5.txt')  # 反例色情语料


def download():
    get_data_0_from_api()
    get_data_1_from_api()
    get_data_5_from_api()


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


