#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: linear.py
@time: 2020/10/10 6:30 下午
@desc:
"""

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
import numpy as np
import os
from PIL import Image

trainset = paddle.dataset.mnist.train()
train_reader = paddle.batch(trainset, batch_size=8)


class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义一层全连接层，输出维度是1，激活函数为None，即不使用激活函数
        self.fc = Linear(input_dim=784, output_dim=1, act=None)

    # 定义网络结构的前向计算过程
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs

def run():
    with fluid.dygraph.guard():
        # 声明网络结构
        model = MNIST()
        # 启动训练模式
        model.train()
        # 定义数据读取函数，数据读取batch_size设置为16
        train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=16)
        # 定义优化器，使用随机梯度下降SGD优化器，学习率设置为0.001
        optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=model.parameters())

        EPOCH_NUM = 10
        for epoch_id in range(EPOCH_NUM):
            for batch_id, data in enumerate(train_loader()):
                # 准备数据，格式需要转换成符合框架要求
                image_data = np.array([x[0] for x in data]).astype('float32')
                label_data = np.array([x[1] for x in data]).astype('float32').reshape(-1, 1)
                # 将数据转为飞桨动态图格式
                image = fluid.dygraph.to_variable(image_data)
                label = fluid.dygraph.to_variable(label_data)

                # 前向计算的过程
                predict = model(image)

                # 计算损失，取一个批次样本损失的平均值
                loss = fluid.layers.square_error_cost(predict, label)
                avg_loss = fluid.layers.mean(loss)

                # 每训练了1000批次的数据，打印下当前Loss的情况
                if batch_id != 0 and batch_id % 1000 == 0:
                    print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))

                # 后向传播，更新参数的过程
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                model.clear_gradients()

        # 保存模型
        fluid.save_dygraph(model.state_dict(), 'mnist')


if __name__ == '__main__':
    run()