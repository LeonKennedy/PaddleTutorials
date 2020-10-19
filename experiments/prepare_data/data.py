#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: data.py
@time: 2020/10/16 1:33 下午
@desc:
"""
from paddle import fluid
import numpy as np


def fluid_data():
    # Creates a variable with fixed size [3, 2, 1]
    # User can only feed data of the same shape to x
    x = fluid.data(name='x', shape=[3, 2, 1], dtype='float32')

    # Creates a variable with changable batch size -1.
    # Users can feed data of any batch size into y,
    # but size of each data sample has to be [2, 1]
    y = fluid.data(name='y', shape=[-1, 2, 1], dtype='float32')

    z = x + y

    # In this example, we will feed x and y with np-ndarry "1"
    # and fetch z, like implementing "1 + 1 = 2" in PaddlePaddle
    feed_data = np.ones(shape=[3, 2, 1], dtype=np.float32)

    exe = fluid.Executor(fluid.CPUPlace())
    out = exe.run(fluid.default_main_program(),
                  feed={
                      'x': feed_data,
                      'y': feed_data
                  },
                  fetch_list=[z.name])

    # np-ndarray of shape=[3, 2, 1], dtype=float32, whose elements are 2
    print(out)


def fluid_data2():
    x = fluid.data(name='x', shape=[2, 2], dtype='float32')
    y = fluid.data(name='y', shape=[2, 2], dtype='float32')
    place = fluid.CPUPlace()
    np_data = np.array([[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3], [4, 4], [4, 4]]).astype('float32')
    print(np_data.shape)
    x_lod_tensor = fluid.create_lod_tensor(np_data, [[2, 2], [2, 2, 2, 2]], place)

    print(x_lod_tensor)


fluid_data2()
