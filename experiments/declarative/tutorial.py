#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: tutorial.py
@time: 2020/10/16 9:50 上午
@desc:
"""

import paddle.fluid as fluid

"""

# 定义一个数据类型为int64的二维数据变量x，x第一维的维度为3，第二个维度未知，要在程序执行过程中才能确定，因此x的形状可以指定为[3, None]
x = fluid.data(name="x", shape=[3, None], dtype="int64")

# 大多数网络都会采用batch方式进行数据组织，batch大小在定义时不确定，因此batch所在维度（通常是第一维）可以指定为None
batched_x = fluid.data(name="batched_x", shape=[None, 3, None], dtype='int64')

#创建常量
data = fluid.layers.fill_constant(shape=[3, 4], value=16, dtype='int64')
"""

def print_data():
    data = fluid.layers.fill_constant(shape=[3, 4], value=16, dtype='int64')
    data = fluid.layers.Print(data, message="Print data:")

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    ret = exe.run()


if __name__ == '__main__':
    print_data()