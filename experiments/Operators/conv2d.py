#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: conv2d.py
@time: 2020/10/17 10:49 上午
@desc:
"""
from paddle import fluid
import numpy as np

place = fluid.CPUPlace()
exe = fluid.Executor(place)
x = fluid.layers.data('audio', shape=(1, 40, 63), dtype='float32', append_batch_size=True)
param_attr = fluid.ParamAttr(name='conv2d.weight', initializer=fluid.initializer.Xavier(uniform=False),
                             learning_rate=0.001)
z = fluid.layers.conv2d(input=x,
                        filter_size=(11, 1),
                        num_filters=5,
                        stride=(3, 1),
                        padding=(3, 2),
                        param_attr=param_attr
                        )
data = np.random.uniform(-1, 1, size=(8, 1, 40, 63)).astype("float32")
print(data.shape)
exe.run(fluid.default_startup_program())
out = exe.run(
    feed={'audio': data},
    fetch_list=[z.name])
print(out)

# data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
# param_attr = fluid.ParamAttr(name='conv2d.weight', initializer=fluid.initializer.Xavier(uniform=False), learning_rate=0.001)
# res = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu", param_attr=param_attr)
# place = fluid.CPUPlace()
# exe = fluid.Executor(place)
# exe.run(fluid.default_startup_program())
# x = np.random.rand(1, 3, 32, 32).astype("float32")
# output = exe.run(feed={"data": x}, fetch_list=[res])
# print(output[0].shape)
