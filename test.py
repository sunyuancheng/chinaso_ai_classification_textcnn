#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : pureoym
# @Contact : pureoym@163.com
# @TIME    : 2018/9/5 10:33
# @File    : model.py
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

# transformer 模型
# 带attention的lstm

from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate, Embedding
from keras.models import Model
from keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
import pandas as pd
import jieba
import os
import numpy as np

# 数据保存地址
BASE_DIR = '/data0/search/ai_challenger/2/'
WORD2VEC = '/data0/search/textcnn/data/sgns.merge.bigram'
DATA_PATH = os.path.join(BASE_DIR, 'data/train/sentiment_analysis_trainingset.csv')
WORD_INDEX = os.path.join(BASE_DIR, 'data/word_index.npy')
EMBEDDING_MATRIX = os.path.join(BASE_DIR, 'data/embedding_matrix.npy')
SEG_DATA = os.path.join(BASE_DIR, 'data/seg_data.csv')


MODEL_DIR = os.path.join(BASE_DIR, 'text2label123/')
NUMERIC_DATA = os.path.join(MODEL_DIR, 'numeric_data.csv')
MODEL = os.path.join(MODEL_DIR, 'model.h5')

SEG_SPLITTER = ' '
word_index = {}

# Model Hyperparameters
EMBEDDING_DIM = 300  # 词向量维数
NUM_FILTERS = 100  # 滤波器数目
FILTER_SIZES = [2, 3, 4, 5]  # 卷积核
DROPOUT_RATE = 0.5
HIDDEN_DIMS = 64
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2

# Training parameters
BATCH_SIZE = 64
NUM_EPOCHS = 1

# Prepossessing parameters
MAX_SEQUENCE_LENGTH = 300
MAX_NUM_WORDS = 150000  # 词典最大词数，若语料中含词数超过该数，则取前MAX_NUM_WORDS个
NUM_LABELS = 64  # 分类数目

#
columns = ['id', 'content', 'location_traffic_convenience',
           'location_distance_from_business_district', 'location_easy_to_find',
           'service_wait_time', 'service_waiters_attitude',
           'service_parking_convenience', 'service_serving_speed', 'price_level',
           'price_cost_effective', 'price_discount', 'environment_decoration',
           'environment_noise', 'environment_space', 'environment_cleaness',
           'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
           'others_overall_experience', 'others_willing_to_consume_again']

label_dict = {'location_traffic_convenience': 'l1',
              'location_distance_from_business_district': 'l2',
              'location_easy_to_find': 'l3',
              'service_wait_time': 'l4',
              'service_waiters_attitude': 'l5',
              'service_parking_convenience': 'l6',
              'service_serving_speed': 'l7',
              'price_level': 'l8',
              'price_cost_effective': 'l9',
              'price_discount': 'l10',
              'environment_decoration': 'l11',
              'environment_noise': 'l12',
              'environment_space': 'l13',
              'environment_cleaness': 'l14',
              'dish_portion': 'l15',
              'dish_taste': 'l16',
              'dish_look': 'l17',
              'dish_recommendation': 'l18',
              'others_overall_experience': 'l19',
              'others_willing_to_consume_again': 'l20'}

# 打开停用词表并做处理
STOP_WORDS_LIST = os.path.join(BASE_DIR, 'stop_list.txt')  # 停用词表
with open(STOP_WORDS_LIST, 'r') as f:
    stop_words = f.readlines()
del stop_words[0]  # 删除txt文件第一行的特殊字符
for word in stop_words:  # 删除每行最后的回车
    stop_words[stop_words.index(word)] = word.replace('\n', '')


def prepare_data():
    """
    数据准备
    1 读取CSV
    2 分词，并保存结果
    3 获取词字典 word_index 按照词频倒排
    4 统计词频 按照顺序倒排
    5
    :return:
    """

    # 分词
    data = pd.read_csv(DATA_PATH)
    seg_data = data
    data['tokens'] = data['content'].map(segment)

    # 标签处理与统计
    # 将标签列名转换成['l1','l2',...,'l20']
    # 将[-2,-1,0,1]转换成[0,1,2,3]
    data.rename(columns=label_dict, inplace=True)
    for i in range(20):
        name = 'l' + str(i + 1)
        data[name] = data[name].map(lambda x: x + 2)
        series_i = pd.Series(data[name])
        print('l1 value counts :\n')
        print(series_i.value_counts())

    # 获取word_index并保存
    word_index = get_word_index(data)
    np.save(WORD_INDEX, word_index)

    # 序列化输入
    data['indexes'] = data['tokens'].map(word2index)

    # 保存处理后的结果
    data.to_csv(SEG_DATA)

    # 处理标签
    # one_label_data = data[['tokens', 'l1']]
    # numeric_data = one_label_data[['indexes', 'l1']]
    # numeric_data.to_csv(NUMERIC_DATA, encoding='utf-8')
    three_label_data = data[['indexes', 'l1', 'l2', 'l3']]
    three_label_data['labels'] = three_label_data['l1'].map(lambda x: x * 16) + \
                                 three_label_data['l2'].map(lambda x: x * 4) + \
                                 three_label_data['l3']
    print(pd.Series(three_label_data['labels']).value_counts())
    numeric_data = three_label_data[['indexes', 'labels']]

    # 保存结果
    numeric_data.to_csv(NUMERIC_DATA, encoding='utf-8')

    # 获取embeddings_index（加载预训练好的word2vec词典）
    embeddings_index = get_embeddings_index()
    print('get embeddings_index (or word2vec dict)')

    # 通过获取embeddings_index以及word_index，生成embedding_matrix
    embedding_matrix = generate_embedding_matrix(embeddings_index)
    np.save(EMBEDDING_MATRIX, embedding_matrix)
    print('generate_embedding_matrix and save to' + EMBEDDING_MATRIX)


def pre_processing_multi_class():
    """
    预处理。获取训练集，测试集。
    :return:
    """

    # 获取数字化的数据集
    d1 = pd.read_csv(NUMERIC_DATA)
    d1['index_array'] = d1['indexes'].map(lambda x: x.split(SEG_SPLITTER))
    sequences = d1['index_array']
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    labels = d1['labels'].values.reshape(-1, 1)
    labels = to_categorical(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # 切分训练集和测试集
    data_size = data.shape[0]
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    train_test_samples = int(TEST_SPLIT * data_size)

    x_train = data[:-train_test_samples]
    y_train = labels[:-train_test_samples]
    x_test = data[-train_test_samples:]
    y_test = labels[-train_test_samples:]
    print('Shape of data x_train:', x_train.shape)
    print('Shape of label y_train:', y_train.shape)
    print('Shape of data x_test:', x_test.shape)
    print('Shape of label y_test:', y_test.shape)
    return x_train, y_train, x_test, y_test


def text_cnn_multi_class():
    """
    构建多分类text_cnn模型
    :return:
    """
    # Inputs
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    # Embeddings layers
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_matrix = np.load(EMBEDDING_MATRIX)
    num_words = embedding_matrix.shape[0] + 1
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(sequence_input)

    # conv layers
    convs = []
    for filter_size in FILTER_SIZES:
        l_conv = Conv1D(filters=NUM_FILTERS,
                        kernel_size=filter_size,
                        activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(MAX_SEQUENCE_LENGTH - filter_size + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    x = Dropout(DROPOUT_RATE)(merge)
    x = Dense(HIDDEN_DIMS, activation='relu')(x)

    preds = Dense(units=NUM_LABELS, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss="categorical_crossentropy",
                  optimizer="rmsprop",
                  metrics=['acc'])

    return model


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
    for i, word in enumerate(word_dict):
        w = word[0]
        word_index[w] = i + 1
    return word_index


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
            else:
                indexes.append('0')
    return SEG_SPLITTER.join(indexes).strip()


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


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = pre_processing_multi_class()
    model = text_cnn_multi_class()
    model.summary()
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_split=VALIDATION_SPLIT,
              shuffle=True)
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))

    model.save(MODEL)
