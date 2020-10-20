#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: pad.py
@time: 2020/10/19 2:28 下午
@desc:
"""

from paddle import fluid
import numpy as np

place = fluid.CPUPlace()

exe = fluid.Executor(place)


def pad_2_dim(exe):
    seq_lens = [2, 4, 4]
    a = np.random.rand(sum(seq_lens), 5).astype('float32')
    x1 = fluid.create_lod_tensor(a, recursive_seq_lens=[seq_lens], place=place)
    x = fluid.data(name='y', shape=[10, 5], dtype='float32', lod_level=1)
    pad_value = fluid.layers.assign(input=np.array([0.0], dtype=np.float32))
    Out, Length = fluid.layers.sequence_pad(x, pad_value)

    y = fluid.layers.sequence_unpad(Out, Length)
    fluid.layers.Print(y)
    print(y.shape)
    exe.run(fluid.default_startup_program())
    res = exe.run(fluid.default_main_program(),
                  feed={'y': x1}, fetch_list=[Out.name, Length.name, y.name], return_numpy=False)

    return res


# res = pad_2_dim(exe)

def pad_3_dim(exe):
    x = fluid.data(name='audio', shape=[8, None, 29])
    l = fluid.data(name='l', shape=[8], dtype='int64')
    y = fluid.layers.sequence_unpad(x, l)

    x1 = np.random.rand(8, 81, 29).astype("float32")
    l1 = np.array([81, 81, 81, 81, 81, 81, 81, 81]).astype('int64')
    exe.run(fluid.default_startup_program())
    res = exe.run(fluid.default_main_program(),
                  feed={'audio': x1, 'l': l1}, fetch_list=[y.name], return_numpy=False)
    return res


res2 = pad_3_dim(exe)
