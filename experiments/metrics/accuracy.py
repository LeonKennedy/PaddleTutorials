#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: accuracy.py
@time: 2020/12/2 4:01 下午
@desc:
"""
import paddle.fluid as fluid
import numpy as np

data = fluid.layers.data(name="input", shape=[-1, 100], dtype="float32")
label = fluid.layers.data(name="label", shape=[-1, 1], dtype="int")
predict = fluid.layers.softmax(input=data)
result = fluid.layers.accuracy(input=predict, label=label)

place = fluid.CPUPlace()
exe = fluid.Executor(place)

exe.run(fluid.default_startup_program())
x = np.array([[0.2, 0.2, 0.2, 0.2, 0.2],
              [3, 3, 3, 3, 3],
              [4, 4, 4, 4, 4],
              [5, 5, 5, 5, 5]]).astype("float32")
x = np.random.rand(4, 100).astype("float32")
print(x)
y = np.array([[0], [0], [0], [0]])
output = exe.run(feed={"input": x, "label": y},
                 fetch_list=[result[0]])
print(output)
true_y = x.argmax(axis=1).reshape(4, 1)
print(true_y)
output = exe.run(feed={"input": x, "label": true_y},
                 fetch_list=[result[0]])
print(output)
