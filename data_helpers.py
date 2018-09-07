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
import json

# label_index = {'恐怖':1,'暴力':2,'脏话':3,'自杀':4,'色情':5}
label_index = {'恐怖': 1, '正常': 0}

url_1 = 'http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01344/search' \
        '?fields=id,wcaption&filters=EQS_resourceState,4' \
        '|EQS_newsLabel,%E6%81%90%E6%80%96&pagestart=1&fetchsize=10'

url_0 = 'http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01276/search' \
        '?fields=id,wcaption&filters=EQS_ifCompare,1|EQS_resourceState,4|EQS_newsLabelSecond,' \
        '%E6%97%B6%E6%94%BF%E6%BB%9A%E5%8A%A8&orders=wpubTime_desc' \
        '&pagestart=1&fetchsize=10'


def get_x1():
    x1 = []
    with urllib.request.urlopen(url_1) as response:
        resp = response.read()
        j1 = json.loads(resp)
        results = j1['value']
    for result in results:
        label = '1'
        x = result.get('wcaption')
        x1.append(x + '&&&' + '1')
    return x1

def get_x0():
    x0 = []
    with urllib.request.urlopen(url_0) as response:
        resp = response.read()
        j0 = json.loads(resp)
        results = j0['value']
    for result in results:
        label = '1'
        x = result.get('wcaption')
        x0.append(x + '&&&' + '0')
    return x0



with urllib.request.urlopen(url_0) as response:
    json0 = response.read()
df0 = pd.DataFrame(pd.read_json(json0))
