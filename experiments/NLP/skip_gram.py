#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: skip_gram.py
@time: 2020/10/22 11:30 上午
@desc:
"""

import io
import os
import sys
import requests
from collections import OrderedDict
import math
import random
import numpy as np
import paddle
import paddle.fluid as fluid

from paddle.fluid.dygraph.nn import Embedding
from collections import Counter

from data import build_dict, preprocess_data


# 使用二次采样算法（subsampling）处理语料，强化训练效果
def subsampling(corpus, word2id_freq):
    # 这个discard函数决定了一个词会不会被替换，这个函数是具有随机性的，每次调用结果不同
    # 如果一个词的频率很大，那么它被遗弃的概率就很大
    def discard(word_id):
        return random.uniform(0, 1) < 1 - math.sqrt(
            1e-4 / word2id_freq[word_id] * len(corpus))

    print(f"all {len(corpus)} tokens in the corpus")
    corpus = [word for word in corpus if not discard(word)]
    return corpus


# 构造数据，准备模型训练
# max_window_size代表了最大的window_size的大小，程序会根据max_window_size从左到右扫描整个语料
# negative_sample_num代表了对于每个正样本，我们需要随机采样多少负样本用于训练，
# 一般来说，negative_sample_num的值越大，训练效果越稳定，但是训练速度越慢。
def build_data(corpus, vocab_size, max_window_size=3, negative_sample_num=4):
    dataset = []

    for center_word_idx in range(len(corpus)):
        # 以max_window_size为上限，随机采样一个window_size，这样会使得训练更加稳定
        window_size = random.randint(1, max_window_size)
        # 当前的中心词就是center_word_idx所指向的词
        center_word = corpus[center_word_idx]

        # 以当前中心词为中心，左右两侧在window_size内的词都可以看成是正样本
        positive_word_range = (
            max(0, center_word_idx - window_size), min(len(corpus) - 1, center_word_idx + window_size))
        positive_word_candidates = [corpus[idx] for idx in range(positive_word_range[0], positive_word_range[1] + 1) if
                                    idx != center_word_idx]

        # 对于每个正样本来说，随机采样negative_sample_num个负样本，用于训练
        for positive_word in positive_word_candidates:
            # 首先把（中心词，正样本，label=1）的三元组数据放入dataset中，
            # 这里label=1表示这个样本是个正样本
            dataset.append((center_word, positive_word, 1))

            # 开始负采样
            i = 0
            while i < negative_sample_num:
                negative_word_candidate = random.randint(0, vocab_size - 1)

                if negative_word_candidate not in positive_word_candidates:
                    # 把（中心词，正样本，label=0）的三元组数据放入dataset中，
                    # 这里label=0表示这个样本是个负样本
                    dataset.append((center_word, negative_word_candidate, 0))
                    i += 1

        yield dataset
        dataset = []


def build_batch(batch_size, epoch_num):
    # center_word_batch缓存batch_size个中心词
    center_word_batch = []
    # target_word_batch缓存batch_size个目标词（可以是正样本或者负样本）
    target_word_batch = []
    # label_batch缓存了batch_size个0或1的标签，用于模型训练
    label_batch = []

    dataset = []
    first = True
    for epoch in range(epoch_num):
        # 每次开启一个新epoch之前，都对数据进行一次随机打乱，提高训练效果
        if first:
            for tmp in build_data(corpus, vocab_size):
                dataset.extend(tmp)
                for center_word, target_word, label in tmp:
                    # 遍历dataset中的每个样本，并将这些数据送到不同的tensor里
                    center_word_batch.append([center_word])
                    target_word_batch.append([target_word])
                    label_batch.append(label)

                    if len(center_word_batch) == batch_size:
                        yield np.array(center_word_batch).astype("int64"), \
                              np.array(target_word_batch).astype("int64"), \
                              np.array(label_batch).astype("float32")
                        center_word_batch = []
                        target_word_batch = []
                        label_batch = []

        else:
            random.shuffle(dataset)

            for center_word, target_word, label in dataset:
                # 遍历dataset中的每个样本，并将这些数据送到不同的tensor里
                center_word_batch.append([center_word])
                target_word_batch.append([target_word])
                label_batch.append(label)

                if len(center_word_batch) == batch_size:
                    yield np.array(center_word_batch).astype("int64"), \
                          np.array(target_word_batch).astype("int64"), \
                          np.array(label_batch).astype("float32")
                    center_word_batch = []
                    target_word_batch = []
                    label_batch = []

    if len(center_word_batch) > 0:
        yield np.array(center_word_batch).astype("int64"), \
              np.array(target_word_batch).astype("int64"), \
              np.array(label_batch).astype("float32")


class SkipGram(fluid.dygraph.Layer):
    def __init__(self, vocab_size, embedding_size, init_scale=0.1):
        # vocab_size定义了这个skipgram这个模型的词表大小
        # embedding_size定义了词向量的维度是多少
        # init_scale定义了词向量初始化的范围，一般来说，比较小的初始化范围有助于模型训练
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # 使用paddle.fluid.dygraph提供的Embedding函数，构造一个词向量参数
        # 这个参数的大小为：[self.vocab_size, self.embedding_size]
        # 数据类型为：float32
        # 这个参数的名称为：embedding_para
        # 这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        self.embedding = Embedding(
            size=[self.vocab_size, self.embedding_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5 / embedding_size, high=0.5 / embedding_size)))

        # 使用paddle.fluid.dygraph提供的Embedding函数，构造另外一个词向量参数
        # 这个参数的大小为：[self.vocab_size, self.embedding_size]
        # 数据类型为：float32
        # 这个参数的名称为：embedding_para_out
        # 这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        # 跟上面不同的是，这个参数的名称跟上面不同，因此，
        # embedding_para_out和embedding_para虽然有相同的shape，但是权重不共享
        self.embedding_out = Embedding(
            size=[self.vocab_size, self.embedding_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_out_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5 / embedding_size, high=0.5 / embedding_size)))

    # 定义网络的前向计算逻辑
    # center_words是一个tensor（mini-batch），表示中心词
    # target_words是一个tensor（mini-batch），表示目标词
    # label是一个tensor（mini-batch），表示这个词是正样本还是负样本（用0或1表示）
    # 用于在训练中计算这个tensor中对应词的同义词，用于观察模型的训练效果
    def forward(self, center_words, target_words, label):
        # 首先，通过embedding_para（self.embedding）参数，将mini-batch中的词转换为词向量
        # 这里center_words和eval_words_emb查询的是一个相同的参数
        # 而target_words_emb查询的是另一个参数
        center_words_emb = self.embedding(center_words)
        target_words_emb = self.embedding_out(target_words)

        # center_words_emb = [batch_size, embedding_size]
        # target_words_emb = [batch_size, embedding_size]
        # 我们通过点乘的方式计算中心词到目标词的输出概率，并通过sigmoid函数估计这个词是正样本还是负样本的概率。
        word_sim = fluid.layers.elementwise_mul(center_words_emb, target_words_emb)
        word_sim = fluid.layers.reduce_sum(word_sim, dim=-1)
        word_sim = fluid.layers.reshape(word_sim, shape=[-1])
        pred = fluid.layers.sigmoid(word_sim)

        # 通过估计的输出概率定义损失函数，注意我们使用的是sigmoid_cross_entropy_with_logits函数
        # 将sigmoid计算和cross entropy合并成一步计算可以更好的优化，所以输入的是word_sim，而不是pred

        loss = fluid.layers.sigmoid_cross_entropy_with_logits(word_sim, label)
        loss = fluid.layers.reduce_mean(loss)

        # 返回前向计算的结果，飞桨会通过backward函数自动计算出反向结果。
        return pred, loss


corpus = preprocess_data()
word2id_dict, word2id_freq, id2word_dict = build_dict(corpus)
corpus = [word2id_dict[word] for word in corpus]
vocab_size = len(word2id_dict)
print(f"vocab size: {vocab_size}")
corpus = subsampling(corpus, word2id_freq)
print("after subsampling %d tokens in the corpus" % len(corpus))
# print(f"finish create dateset: {len(dataset)} sample")


step = 0
learning_rate = 0.001


# 定义一个使用word-embedding查询同义词的函数
# 这个函数query_token是要查询的词，k表示要返回多少个最相似的词，embed是我们学习到的word-embedding参数
# 我们通过计算不同词之间的cosine距离，来衡量词和词的相似度
# 具体实现如下，x代表要查询词的Embedding，Embedding参数矩阵W代表所有词的Embedding
# 两者计算Cos得出所有词对查询词的相似度得分向量，排序取top_k放入indices列表
def get_similar_tokens(query_token, k, embed):
    W = embed.numpy()
    x = W[word2id_dict[query_token]]
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    flat = cos.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    for i in indices:
        print('for word %s, the similar word is %s' % (query_token, str(id2word_dict[i])))


# 将模型放到GPU上训练（fluid.CUDAPlace(0)），如果需要指定CPU，则需要改为fluid.CPUPlace()
with fluid.dygraph.guard(fluid.CUDAPlace(0)):
    epoch_num = 3
    batch_size = 512
    embedding_size = 200
    # 通过我们定义的SkipGram类，来构造一个Skip-gram模型网络
    skip_gram_model = SkipGram(vocab_size, embedding_size)
    # 构造训练这个网络的优化器
    adam = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate, parameter_list=skip_gram_model.parameters())

    # 使用build_batch函数，以mini-batch为单位，遍历训练数据，并训练网络
    for center_words, target_words, label in build_batch(batch_size, epoch_num):
        # 使用fluid.dygraph.to_variable函数，将一个numpy的tensor，转换为飞桨可计算的tensor
        center_words_var = fluid.dygraph.to_variable(center_words)
        target_words_var = fluid.dygraph.to_variable(target_words)
        label_var = fluid.dygraph.to_variable(label)

        # 将转换后的tensor送入飞桨中，进行一次前向计算，并得到计算结果
        pred, loss = skip_gram_model(
            center_words_var, target_words_var, label_var)

        # 通过backward函数，让程序自动完成反向计算
        loss.backward()
        # 通过minimize函数，让程序根据loss，完成一步对参数的优化更新
        adam.minimize(loss)
        # 使用clear_gradients函数清空模型中的梯度，以便于下一个mini-batch进行更新
        skip_gram_model.clear_gradients()

        # 每经过100个mini-batch，打印一次当前的loss，看看loss是否在稳定下降
        step += 1
        if step % 100 == 0:
            print("step %d, loss %.3f" % (step, loss.numpy()[0]))

        # 经过10000个mini-batch，打印一次模型对eval_words中的10个词计算的同义词
        # 这里我们使用词和词之间的向量点积作为衡量相似度的方法
        # 我们只打印了5个最相似的词
        if step % 10000 == 0:
            get_similar_tokens('one', 5, skip_gram_model.embedding.weight)
            get_similar_tokens('she', 5, skip_gram_model.embedding.weight)
            get_similar_tokens('chip', 5, skip_gram_model.embedding.weight)
