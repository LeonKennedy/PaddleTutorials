#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: data.py
@time: 2020/10/14 10:55 上午
@desc:
"""
import requests
import os
import random
import math

corpus_filename = "text8.txt"


def download():
    corpus_url = "https://dataset.bj.bcebos.com/word2vec/text8.txt"
    web_request = requests.get(corpus_url)
    corpus = web_request.content
    with open(corpus_filename, "wb") as f:
        f.write(corpus)
    f.close()


def load_text8():
    if not os.path.exists(corpus_filename):
        download()
    with open(corpus_filename, "r") as f:
        corpus = f.read().strip("\n")
    return corpus


def data_preprocess(corpus):
    # 由于英文单词出现在句首的时候经常要大写，所以我们把所有英文字符都转换为小写，
    # 以便对语料进行归一化处理（Apple vs apple等）
    corpus = corpus.strip().lower()
    corpus = corpus.split(" ")
    return corpus


def build_dict(corpus):
    # 首先统计每个不同词的频率（出现的次数），使用一个词典记录
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1

    # 将这个词典中的词，按照出现次数排序，出现次数越高，排序越靠前
    # 一般来说，出现频率高的高频词往往是：I，the，you这种代词，而出现频率低的词，往往是一些名词，如：nlp
    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

    # 构造3个不同的词典，分别存储，
    # 每个词到id的映射关系：word2id_dict
    # 每个id出现的频率：word2id_freq
    # 每个id到词的映射关系：id2word_dict
    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = dict()

    # 按照频率，从高到低，开始遍历每个单词，并为这个单词构造一个独一无二的id
    for word, freq in word_freq_dict:
        curr_id = len(word2id_dict)
        word2id_dict[word] = curr_id
        word2id_freq[word2id_dict[word]] = freq
        id2word_dict[curr_id] = word

    return word2id_freq, word2id_dict, id2word_dict


def convert_corpus_to_id(corpus, word2id_dict):
    # 使用一个循环，将语料中的每个词替换成对应的id，以便于神经网络进行处理
    corpus = [word2id_dict[word] for word in corpus]
    return corpus


def subsampling(corpus, word2id_freq):
    # 这个discard函数决定了一个词会不会被替换，这个函数是具有随机性的，每次调用结果不同
    # 如果一个词的频率很大，那么它被遗弃的概率就很大
    def discard(word_id):
        return random.uniform(0, 1) < 1 - math.sqrt(
            1e-4 / word2id_freq[word_id] * len(corpus))

    corpus = [word for word in corpus if not discard(word)]
    return corpus


# max_window_size代表了最大的window_size的大小，程序会根据max_window_size从左到右扫描整个语料
# negative_sample_num代表了对于每个正样本，我们需要随机采样多少负样本用于训练，
# 一般来说，negative_sample_num的值越大，训练效果越稳定，但是训练速度越慢。
def build_data(corpus, word2id_dict, word2id_freq, max_window_size=3, negative_sample_num=4):
    # 使用一个list存储处理好的数据
    dataset = []
    vocab_size = len(word2id_freq)
    # 从左到右，开始枚举每个中心点的位置
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

    return dataset


corpus = load_text8()
# 对语料进行预处理（分词）
corpus = data_preprocess(corpus)
# 构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
word2id_freq, word2id_dict, id2word_dict = build_dict(corpus)
vocab_size = len(word2id_freq)
print("there are totoally %d different words in the corpus" % len(word2id_freq))
corpus = convert_corpus_to_id(corpus, word2id_dict)
print("%d tokens in the corpus" % len(corpus))
# 使用二次采样算法（subsampling）处理语料，强化训练效果
corpus = subsampling(corpus, word2id_freq)
print("%d tokens in the corpus after subsample" % len(corpus))
# 构造数据，准备模型训练
dataset = build_data(corpus, word2id_dict, word2id_freq)
