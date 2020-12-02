#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: sequence_unpad.py
@time: 2020/10/18 10:59 上午
@desc:
"""
import paddle.fluid as fluid
import numpy

# example 1:
train_pro = fluid.Program()
start_pro = fluid.Program()
with fluid.program_guard(train_pro, start_pro):
    x = fluid.data(name='x', shape=[10, 5], dtype='float32')
    len = fluid.data(name='length', shape=[10], dtype='int64')
    out = fluid.layers.sequence_unpad(x=x, length=len)
# # example 2:
# # 使用sequence_pad填充数据
# # input = fluid.data(name='input', shape=[10, 5], dtype='float32', lod_level=1)
# # pad_value = fluid.layers.assign(input=numpy.array([0.0], dtype=numpy.float32))
# # pad_data, len = fluid.layers.sequence_pad(x=input, pad_value=pad_value)
#
# #使用sequence_unpad移除填充数据
# # unpad_data = fluid.layers.sequence_unpad(x=pad_data, length=len)
#
a = numpy.random.rand(10, 5).astype("float32")
b = numpy.random.rand(10).astype("int64")
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(start_pro)

s = exe.run(train_pro,
            feed={"x": a, "length":b},
            fetch_list=[out])

print(s.numpy())

# train_program = fluid.Program()
# startup_program = fluid.Program()
# with fluid.program_guard(train_program, startup_program):
#     data = fluid.data(name='X', shape=[None, 1], dtype='float32')
#     hidden = fluid.layers.fc(input=data, size=10)
#     loss = fluid.layers.mean(hidden)
#     sgd = fluid.optimizer.SGD(learning_rate=0.001)
#     sgd.minimize(loss)
#
# use_cuda = True
# place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
# exe = fluid.Executor(place)
#
# # Run the startup program once and only once.
# # Not need to optimize/compile the startup program.
# startup_program.random_seed = 1
# exe.run(startup_program)
#
# # Run the main program directly without compile.
# x = numpy.random.random(size=(10, 1)).astype('float32')
# loss_data, = exe.run(train_program,
#                      feed={"X": x},
#                      fetch_list=[loss.name])
