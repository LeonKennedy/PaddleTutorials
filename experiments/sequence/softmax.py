#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: softmax.py
@time: 2020/10/18 3:31 下午
@desc:
"""

from paddle import fluid
import numpy as np

place = fluid.CPUPlace()
# a = fluid.create_random_int_lodtensor([[1, 2, 3]], base_shape=[5], place=place, low=0, high=1)
x = fluid.data(name='x', shape=[5])
out = fluid.layers.sequence_softmax(x)
seq_lens = [2, 4, 4]
a = np.random.rand(sum(seq_lens)).astype('float32')
x1 = fluid.create_lod_tensor(a, recursive_seq_lens=[seq_lens], place=place)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
res = exe.run(fluid.default_main_program(),
              feed={'x': x1},
              fetch_list=[out.name], return_numpy=False)
print(res[0])
