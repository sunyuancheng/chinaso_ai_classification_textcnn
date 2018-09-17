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
import urllib.request
import os
import json

# 新闻数据接口
NEWS_DATA_API = 'http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01276/search' \
                '?fields=id,wcaption&filters=EQS_ifCompare,1|EQS_resourceState,4|EQS_newsLabelSecond,' \
                '%E6%97%B6%E6%94%BF%E6%BB%9A%E5%8A%A8&orders=wpubTime_desc' \
                '&pagestart=1&fetchsize=10000'

# 反例数据接口：恐怖
HORROR_DATA_API = 'http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01344/search' \
                  '?fields=id,wcaption&filters=EQS_resourceState,4' \
                  '|EQS_newsLabel,%E6%81%90%E6%80%96&pagestart=1&fetchsize=10000'

# 反例数据接口：色情
SEX_DATA_API = 'http://data.mgt.chinaso365.com/datasrv/1.0/resources/01344/search' \
               '?fields=id,wcaption&filters=EQS_resourceState,4|EQS_newsLabel,%E8%89%B2%E6%83%85%7C' \
               'NES_newsLabelSecond,%E8%89%B2%E6%83%85%E5%B0%8F%E8%AF%B4&pagestart=1&fetchsize=10000'

# 反例数据接口：色情2
SEX_DATA_API_2 = 'http://data.mgt.chinaso365.com/datasrv/1.0/resources/01344/search' \
                 '?fields=id,wcaption,picSet&filters=EQS_resourceState,4|EQS_newsLabel,%E8%89%B2%E6%83%85%7C' \
                 'EQS_newsLabelSecond,%E8%89%B2%E6%83%85%E5%B0%8F%E8%AF%B4&pagestart=1&fetchsize=10000'

label_index = {'news': 0, 'horror': 1, 'violence': 2, 'dirty_words': 3, 'suicide': 4, 'sex': 5}

# 数据保存地址
BASE_DIR = '/data0/search/textcnn/data/'
DATASET = os.path.join(BASE_DIR, 'dataset/')  # 数据集文件夹
NEWS_DATA = os.path.join(DATASET, 'news/data_0.txt')  # 新闻语料
HORROR_DATA = os.path.join(DATASET, 'horror/data_1.txt')  # 反例恐怖语料
SEX_DATA = os.path.join(DATASET, 'sex/data_5.txt')  # 反例色情语料
SEX_DATA2 = os.path.join(DATASET, 'sex/data_55.txt')  # 反例色情语料


def download():
    get_data_from_api(NEWS_DATA, NEWS_DATA_API, 'news')
    get_data_from_api(HORROR_DATA, HORROR_DATA_API, 'horror')
    get_data_from_api(SEX_DATA, SEX_DATA_API, 'sex')
    get_data_from_api_2(SEX_DATA2, SEX_DATA_API_2, 'sex2')


def get_data_from_api(data_path, api, pre_process_type):
    """
    通过接口获取数据，并保存至文件
    :param data_path: 保存路径
    :param api: 接口
    :param pre_process_type: 预处理类型
    :return:
    """
    with open(data_path, 'w') as f:
        with urllib.request.urlopen(api) as response:
            resp = response.read()
            j1 = json.loads(resp)
            results = j1['value']
            for result in results:
                line = result.get('wcaption')
                line = pre_process(line, pre_process_type)
                f.write(line + '\n')


def get_data_from_api_2(data_path, api, pre_process_type):
    """
    通过接口2获取数据，并保存至文件
    :param data_path: 保存路径
    :param api: 接口
    :param pre_process_type: 预处理类型
    :return:
    """
    with open(data_path, 'w') as f:
        with urllib.request.urlopen(api) as response:
            resp = response.read()
            captions = json.loads(resp)
            for item in captions:
                line = item.get('wcaption')
                if item.get('picSet') != None:
                    for pic in item.get('picSet'):
                        line = line + pic.get('caption')
                line = pre_process(line, pre_process_type)
                if len(line) > int(100):
                    f.write(line + '\n')


def pre_process(input_line, pre_process_type):
    """
    数据预处理
    :param input_line: 输入数据行
    :param pre_process_type: 预处理类型
    :return: 输出数据行
    """
    output_line = input_line.replace(',', '，')  # 英文逗号为默认csv分隔符
    if pre_process_type == 'news':
        output_line = output_line.replace('|', '')
    elif pre_process_type == 'horror':
        output_line = output_line.replace('免费订阅精彩鬼故事，微信号：guidayecom', '')
    elif pre_process_type == 'sex2':
        output_line = output_line.replace('(转载请注明来源cna5两性网www.cna5.cc)', '').replace('性爱 CNA5两性健康网', '').replace(
            'CNA5两性健康网', '')
    return output_line


if __name__ == '__main__':
    download()
