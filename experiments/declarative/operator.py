#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: operator.py
@time: 2020/10/16 9:58 上午
@desc:
"""
import numpy
from paddle import fluid


def add():
    a = fluid.data(name="a", shape=[None, 1], dtype='int64')
    b = fluid.data(name="b", shape=[None, 1], dtype='int64')

    # 组建网络（此处网络仅由一个操作构成，即elementwise_add）
    result = fluid.layers.elementwise_add(a, b)

    # 准备运行网络
    cpu = fluid.CPUPlace()  # 定义运算设备，这里选择在CPU下训练
    exe = fluid.Executor(cpu)  # 创建执行器
    exe.run(fluid.default_startup_program())  # 网络参数初始化

    # 读取输入数据
    data_1 = int(input("Please enter an integer: a="))
    data_2 = int(input("Please enter an integer: b="))
    x = numpy.array([[data_1]])
    y = numpy.array([[data_2]])

    # 运行网络
    outs = exe.run(
        feed={'a': x, 'b': y},  # 将输入数据x, y分别赋值给变量a，b
        fetch_list=[result]  # 通过fetch_list参数指定需要获取的变量结果
    )

    # 输出计算结果
    print("%d+%d=%d" % (data_1, data_2, outs[0][0]))


def fc_net():
    train_data = numpy.array([[1.0], [2.0], [3.0], [4.0]]).astype('float32')
    y_true = numpy.array([[2.0], [4.0], [6.0], [8.0]]).astype('float32')

    x = fluid.data(name="x", shape=[None, 1], dtype='float32')
    y = fluid.data(name="y", shape=[None, 1], dtype='float32')
    # 搭建全连接网络
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    sgd_optimizer.minimize(avg_cost)

    # 网络参数初始化
    cpu = fluid.CPUPlace()
    exe = fluid.Executor(cpu)
    exe.run(fluid.default_startup_program())

    # 开始训练，迭代100次
    for i in range(100):
        outs = exe.run(
            feed={'x': train_data, 'y': y_true},
            fetch_list=[y_predict, avg_cost])
        print(outs)

if __name__ == '__main__':
    # add()
    fc_net()