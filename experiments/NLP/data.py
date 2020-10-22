#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: data.py
@time: 2020/10/22 11:31 上午
@desc:
"""
from collections import Counter

import requests
import os


def download():
    # 可以从百度云服务器下载一些开源数据集（dataset.bj.bcebos.com）
    corpus_url = "https://dataset.bj.bcebos.com/word2vec/text8.txt"
    web_request = requests.get(corpus_url)
    corpus = web_request.content
    with open("./text8.txt", "wb") as f:
        f.write(corpus)
    f.close()


def load_text8():
    filename = 'text8.txt'
    if not os.path.exists(filename):
        download()
    with open("./text8.txt", "r") as f:
        corpus = f.read().strip("\n")
    f.close()

    return corpus


def preprocess_data():
    corpus = load_text8()
    return corpus.strip().lower().split(" ")


def build_dict(corpus):
    word_dict = Counter(corpus)
    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = list()
    cur_id = 0
    for word, freq in word_dict.most_common():
        word2id_dict[word] = cur_id
        id2word_dict.append(word)
        word2id_freq[cur_id] = freq
        cur_id += 1
    return word2id_dict, word2id_freq, id2word_dict
