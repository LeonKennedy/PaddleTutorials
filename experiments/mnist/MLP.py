#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: MLP.py
@time: 2020/10/10 11:25 下午
@desc:
"""

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
from data import INPUT_DIM
import numpy as np

class MultilayerPerception(fluid.dygraph.Layer):
    def __init__(self):
        super(MultilayerPerception, self).__init__()
        self.fc1 = Linear(input_dim=INPUT_DIM, output_dim=10, act="relu")
        self.fc2 = Linear(input_dim=10, output_dim=10, act="relu")
        self.fc3 = Linear(input_dim=10, output_dim=1, act=None)

    def forward(self, inputs, labels=None):
        inputs = fluid.layers.reshape(inputs, [inputs.shape[0], 784])
        outputs1 = self.fc1(inputs)
        outputs2 = self.fc2(outputs1)
        outputs_final = self.fc3(outputs2)
        return outputs_final


def run():
    with fluid.dygraph.guard():
        model = MultilayerPerception()
        model.train()
        # 使用SGD优化器，learning_rate设置为0.01
        optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())
        # 训练5轮
        train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=16)
        EPOCH_NUM = 5
        for epoch_id in range(EPOCH_NUM):
            for batch_id, data in enumerate(train_loader()):
                # 准备数据
                image_data = np.array([x[0] for x in data]).astype('float32')
                label_data = np.array([x[1] for x in data]).astype('float32').reshape(-1, 1)
                image = fluid.dygraph.to_variable(image_data)
                label = fluid.dygraph.to_variable(label_data)

                # 前向计算的过程
                predict = model(image)

                # 计算损失，取一个批次样本损失的平均值
                loss = fluid.layers.square_error_cost(predict, label)
                avg_loss = fluid.layers.mean(loss)

                # 每训练了200批次的数据，打印下当前Loss的情况
                if batch_id % 200 == 0:
                    print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))

                # 后向传播，更新参数的过程
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                model.clear_gradients()

        # 保存模型参数
        fluid.save_dygraph(model.state_dict(), 'mnist')


if __name__ == '__main__':
    run()