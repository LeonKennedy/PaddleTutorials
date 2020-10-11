#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: data.py
@time: 2020/10/10 11:26 下午
@desc:
"""

import os
import random
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# import gzip
import json

IMG_ROWS = 28
IMG_COLS = 28
INPUT_DIM = IMG_COLS * IMG_COLS


# 定义数据集读取器
def load_data(mode='train'):
    # 数据文件
    # data = json.load(gzip.open(datafile))
    # train_set, val_set, eval_set = data
    train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=16)
    image_data = np.array([x[0] for x in data]).astype('float32')
    label_data = np.array([x[1] for x in data]).astype('float32').reshape(-1, 1)
    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS

    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]

    imgs_length = len(imgs)

    assert len(imgs) == len(labels), \
        "length of train_imgs({}) should be the same as train_labels({})".format(
            len(imgs), len(labels))

    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('float32')
            imgs_list.append(img)
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator
