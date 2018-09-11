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
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate,Embedding
from keras.models import Model
from keras.initializers import Constant
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import os
import sys
import pandas as pd
import jieba

# Word2Vec source, data source
BASE_DIR = '/data0/search/textcnn/data/'
EMBEDDING_MATRIX = os.path.join(BASE_DIR, 'embedding_matrix.npy')  # embedding_matrix

# Model Hyperparameters
EMBEDDING_DIM = 300  # 词向量维数
NUM_FILTERS = 100  # 滤波器数目
FILTER_SIZES = [2, 3, 4, 5]  # 卷积核
DROPOUT_RATE = 0.5
HIDDEN_DIMS = 64
VALIDATION_SPLIT = 0.2

# Training parameters
BATCH_SIZE = 64
NUM_EPOCHS = 10

# Prepossessing parameters
MAX_NUM_WORDS = 157000  # 词典最大词数，若语料中含词数超过该数，则取前MAX_NUM_WORDS个
MAX_SEQUENCE_LENGTH = 2000  # 每篇文章最长词数


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
    # embeddings_index = get_embeddings_index()
    word_index = get_word_index()
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.load(EMBEDDING_MATRIX)
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



def get_word_index():
    """
    加载计算好的word_index
    :return:
    """
    df = pd.read_csv(WORD_INDEX)



def test_get_word_index(texts):
    """
    vectorize the text samples into a 2D integer tensor
    :param texts:
    :return:
    """
    sentence = ''
    for text in texts:
        sentence = sentence+text
        # seg_list = jieba.cut(text, cut_all=False)
        # output = ' '.join(list(seg_list))
    tags = jieba.analyse.extract_tags(sentence, topK=10, withWeight=False, allowPOS=())
    d1 = {}
    for i,tag in enumerate(tags):
        d1[tag]=i+1

    return d1



def pre_process():
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # 切分成测试集和验证集
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    return word_index, x_train, y_train, x_val, y_val


def get_texts_and_labels():
    """
    prepare text samples and their labels
    :return:
    """
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    args = {} if sys.version_info < (3,) else {'encoding': 'utf-8'}
                    with open(fpath, **args) as f:
                        t = f.read()
                        i = t.find('\n\n')  # skip header
                        if 0 < i:
                            t = t[i:]
                        texts.append(t)
                    labels.append(label_id)
    print('Found %s texts.' % len(texts))
    return texts, labels_index, labels


def text_to_index_array(word2vec_dict, text):  # 文本转为索引数字模式
    new_sentences = []
    for sen in text:
        new_sen = []
        for word in sen:
            try:
                new_sen.append(word2vec_dict[word])  # 单词转索引数字
            except:
                new_sen.append(0)  # 索引字典里没有的词转为数字0
        new_sentences.append(new_sen)

    return np.array(new_sentences)


def get_training_set_and_validation_set(data, labels):
    """
    split the data into a training set and a validation set
    :return:
    """
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    return x_train, y_train, x_val, y_val


def preprocessing(train_texts, train_labels, test_texts, test_labels):
    tokenizer = Tokenizer(num_words=2000)  # 建立一个2000个单词的字典
    tokenizer.fit_on_texts(train_texts)
    # 对每一句影评文字转换为数字列表，使用每个词的编号进行编号
    x_train_seq = tokenizer.texts_to_sequences(train_texts)
    x_test_seq = tokenizer.texts_to_sequences(test_texts)
    x_train = sequence.pad_sequences(x_train_seq, maxlen=150)
    x_test = sequence.pad_sequences(x_test_seq, maxlen=150)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing()
    model = text_cnn()
    model.summary()
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=(x_val, y_val),
              shuffle=True)
    scores = model.eveluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
