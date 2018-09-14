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
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate, Embedding
from keras.models import Model
from keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
import os
import pandas as pd

# Word2Vec source, data source
BASE_DIR = '/data0/search/textcnn/data/'
EMBEDDING_MATRIX = os.path.join(BASE_DIR, 'embedding_matrix.npy')  # embedding_matrix
NUMERIC_DATA = os.path.join(BASE_DIR, 'numeric_data.csv')  # 序号化后数据
MODEL = os.path.join(BASE_DIR, '/backup/model_textcnn.h5')

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
MAX_NUM_WORDS = 150000  # 词典最大词数，若语料中含词数超过该数，则取前MAX_NUM_WORDS个
MAX_SEQUENCE_LENGTH = 2000  # 每篇文章最长词数
NUM_LABELS = 6 # 分类数目


def text_cnn():
    """
    构建text_cnn模型
    :return:
    """
    # Inputs
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    # Embeddings layers
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_matrix = np.load(EMBEDDING_MATRIX)
    num_words = embedding_matrix.shape[0]+1
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

    preds = Dense(units=1, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

    return model


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
    num_words = embedding_matrix.shape[0]+1
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


def pre_processing():
    """
    预处理。获取训练集，测试集。
    :return:
    """

    # 获取数字化的数据集
    d1 = pd.read_csv(NUMERIC_DATA)
    d1['index_array'] = d1['indexes'].map(lambda x: x.split(' '))
    sequences = d1['index_array']
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    labels = d1['label'].values.reshape(-1, 1)
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


def pre_processing_multi_class():
    """
    预处理。获取训练集，测试集。
    :return:
    """

    # 获取数字化的数据集
    d1 = pd.read_csv(NUMERIC_DATA)
    d1['index_array'] = d1['indexes'].map(lambda x: x.split(' '))
    sequences = d1['index_array']
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    labels = d1['label'].values.reshape(-1, 1)
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


def test():
    # 获取text set
    DATA_TEST = '/data0/search/textcnn/data/data_55.txt'
    df = pd.read_csv(DATA_TEST, header=None, names=['doc'])
    df['label'] = 5
    df['tokens'] = df['doc'].map(segment)
    df['indexes'] = df['tokens'].map(word2index)
    df['index_array'] = df['indexes'].map(lambda x: x.split(' '))
    sequences = df['index_array']
    x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    labels = df['label'].values.reshape(-1, 1)
    y_test = to_categorical(labels)

    # 加载模型，评估
    model_path = '/data0/search/textcnn/data/backup/model_textcnn_015_20180913.h5'
    model = load_model(model_path)
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))


if __name__ == "__main__":
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

    # model.save(MODEL)
    # model = load_model(MODEL)
