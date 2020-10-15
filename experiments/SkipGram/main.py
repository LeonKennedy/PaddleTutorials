#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: main.py
@time: 2020/10/14 11:30 上午
@desc:
"""
import numpy as np
from paddle import fluid
from data import word2id_dict, dataset, id2word_dict, vocab_size
import random
from net import SkipGram


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


def build_batch(dataset, batch_size, epoch_num):
    # center_word_batch缓存batch_size个中心词
    center_word_batch = []
    # target_word_batch缓存batch_size个目标词（可以是正样本或者负样本）
    target_word_batch = []
    # label_batch缓存了batch_size个0或1的标签，用于模型训练
    label_batch = []

    for epoch in range(epoch_num):
        # 每次开启一个新epoch之前，都对数据进行一次随机打乱，提高训练效果
        random.shuffle(dataset)

        for center_word, target_word, label in dataset:
            # 遍历dataset中的每个样本，并将这些数据送到不同的tensor里
            center_word_batch.append([center_word])
            target_word_batch.append([target_word])
            label_batch.append(label)

            # 当样本积攒到一个batch_size后，我们把数据都返回回来
            # 在这里我们使用numpy的array函数把list封装成tensor
            # 并使用python的迭代器机制，将数据yield出来
            # 使用迭代器的好处是可以节省内存
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


def main():
    batch_size = 512
    epoch_num = 3
    embedding_size = 200
    step = 0
    learning_rate = 0.001
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        skip_gram_model = SkipGram(vocab_size, embedding_size)
        # 构造训练这个网络的优化器
        adam = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate, parameter_list=skip_gram_model.parameters())

        # 使用build_batch函数，以mini-batch为单位，遍历训练数据，并训练网络
        for center_words, target_words, label in build_batch(dataset, batch_size, epoch_num):
            # 使用fluid.dygraph.to_variable函数，将一个numpy的tensor，转换为飞桨可计算的tensor
            center_words_var = fluid.dygraph.to_variable(center_words)
            target_words_var = fluid.dygraph.to_variable(target_words)
            label_var = fluid.dygraph.to_variable(label)

            # 将转换后的tensor送入飞桨中，进行一次前向计算，并得到计算结果
            pred, loss = skip_gram_model(center_words_var, target_words_var, label_var)

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


if __name__ == '__main__':
    main()
