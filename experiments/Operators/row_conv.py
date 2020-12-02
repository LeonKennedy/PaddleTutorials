#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: row_conv.py
@time: 2020/10/16 12:02 下午
@desc:
"""
import paddle.fluid as fluid
import numpy as np

# LoDTensor input


# Tensor input
xx = fluid.layers.data(name='x', shape=[9, 4, 16],
                       dtype='float32',
                       append_batch_size=False)
out = fluid.layers.row_conv(input=xx, future_context_size=2)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
data = np.ones(shape=[9, 4, 16], dtype=np.float32)
print(data.shape)
out_main = exe.run(fluid.default_main_program(),
                   feed={'x': data},
                   fetch_list=[out], return_numpy=False)
print(out_main)
