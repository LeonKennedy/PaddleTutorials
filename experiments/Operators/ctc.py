#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: ctc.py
@time: 2020/10/16 11:46 上午
@desc:
"""
# using LoDTensor
import paddle.fluid as fluid
import numpy as np

# lengths of logit sequences
class_num = 28
seq_lens = [2, 4]
# lengths of label sequences
label_lens = [2, 3]
# class num

batch_size = 3

logits = fluid.data(name='logits', shape=[None, class_num + 1], dtype='float32', lod_level=1)
label = fluid.data(name='label', shape=[None, 1], dtype='int32', lod_level=1)
cost = fluid.layers.warpctc(input=logits, label=label)

place = fluid.CPUPlace()
xnp = np.random.rand(np.sum(seq_lens), class_num + 1).astype("float32")
x = fluid.create_lod_tensor(xnp, [seq_lens], place)
print(xnp.shape)
y = fluid.create_lod_tensor(
    np.random.randint(0, class_num, [np.sum(label_lens), 4, 1]).astype("int32"),
    [label_lens], place)
print(y)
exe = fluid.Executor(place)
output = exe.run(fluid.default_main_program(),
                 feed={"logits": x, "label": y},
                 fetch_list=[cost.name])
print(output)
