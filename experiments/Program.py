#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: Program.py
@time: 2020/10/15 10:46 上午
@desc:
    start_program 初始化参数 只执行一次
    main_program  随着batch 更新梯度参数 执行多次
"""

from paddle import fluid

train_program = fluid.Program()
start_program = fluid.Program()

with fluid.program_guard(train_program, start_program):
    data = fluid.data("data_1", shape=(2, 4), dtype='float32')
    data2 = fluid.data("data_2", shape=(4, 5), dtype='float32')
    res = fluid.layers.mul(data, data2)
    print(fluid.default_main_program())
