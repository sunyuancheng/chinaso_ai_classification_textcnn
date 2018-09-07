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

# label_index = {'恐怖':1,'暴力':2,'脏话':3,'自杀':4,'色情':5}
label_index = {'恐怖': 1, '正常': 0}

url_1 = 'http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01344/search' \
        '?fields=id,wcaption&filters=EQS_resourceState,4' \
        '|EQS_newsLabel,%E6%81%90%E6%80%96&pagestart=1&fetchsize=10'

url_0 = 'http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01276/search' \
        '?fields=id,wcaption&filters=EQS_ifCompare,1|EQS_resourceState,4|&orders=wpubTime_desc' \
        '&pagestart=1&fetchsize=10'

with urllib.request.urlopen(url_1) as response:
    json1= response.read()
df1 = pd.DataFrame(pd.read_json(json1))

with urllib.request.urlopen(url_0) as response:
    json0= response.read()
df0 = pd.DataFrame(pd.read_json(json0))
