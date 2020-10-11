#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: ConvolutionNatualNetwork.py
@time: 2020/10/11 9:37 上午
@desc:
"""
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from data import IMG_ROWS, IMG_COLS
import numpy as np


class ConvolutionNatualNetwork(fluid.dygraph.Layer):
    def __init__(self):
        super(ConvolutionNatualNetwork, self).__init__()
        self.conv1 = Conv2D(num_channels=1, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
        # 定义池化层，池化核pool_size=2，池化步长为2，选择最大池化方式
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # 定义卷积层，输出特征通道num_filters设置为20，卷积核的大小filter_size为5，卷积步长stride=1，padding=2
        self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
        # 定义池化层，池化核pool_size=2，池化步长为2，选择最大池化方式
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # 定义一层全连接层，输出维度是1，不使用激活函数
        self.fc = Linear(input_dim=980, output_dim=10, act="softmax")

    def forward(self, inputs):
        inputs = fluid.layers.reshape(inputs, [inputs.shape[0], 1, IMG_ROWS, IMG_COLS])
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = fluid.layers.reshape(x, [x.shape[0], 980])
        x = self.fc(x)
        return x


def run():
    with fluid.dygraph.guard():
        model = ConvolutionNatualNetwork()
        model.train()
        # 使用SGD优化器，learning_rate设置为0.01
        optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())
        # 训练5轮
        train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=32)
        EPOCH_NUM = 5
        for epoch_id in range(EPOCH_NUM):
            for batch_id, data in enumerate(train_loader()):
                # 准备数据
                image_data = np.array([x[0] for x in data]).astype('float32')
                label_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)
                image = fluid.dygraph.to_variable(image_data)
                label = fluid.dygraph.to_variable(label_data)

                # 前向计算的过程
                predict = model(image)

                # 计算损失，取一个批次样本损失的平均值
                loss = fluid.layers.cross_entropy(predict, label)
                avg_loss = fluid.layers.mean(loss)

                # 每训练了200批次的数据，打印下当前Loss的情况
                if batch_id % 200 == 0:
                    print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))

                # 后向传播，更新参数的过程
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                model.clear_gradients()

        # 保存模型参数
        # fluid.save_dygraph(model.state_dict(), 'cnn')

if __name__ == '__main__':
    run()

