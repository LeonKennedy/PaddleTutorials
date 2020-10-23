#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: cos_similarity.py
@time: 2020/10/23 4:08 下午
@desc:
"""
import paddle.fluid as fluid
import numpy as np

x = fluid.layers.data(name='x', shape=[3, 7], dtype='float32', append_batch_size=False)
# y维度必须与x维度一样 ，或者为1
y = fluid.layers.data(name='y', shape=[1, 7], dtype='float32', append_batch_size=False)
out = fluid.layers.cos_sim(x, y)
# 如果范围为[0, 5],可以将原来[0, 1]进行缩放
z = fluid.layers.scale(out, 5)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
np_x = np.random.random(size=(3, 7)).astype('float32')
np_y = np.random.random(size=(1, 7)).astype('float32')
output = exe.run(feed={"x": np_x, "y": np_y}, fetch_list=[z])
print(output)
