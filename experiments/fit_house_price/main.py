#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: linear.py
@time: 2020/10/10 5:42 下午
@desc:
"""

import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
import paddle.fluid.dygraph as dygraph
import numpy as np


def preprocess(ratio=0.8):
    data = np.fromfile('housing.data', sep=' ')
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                     'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    # split train and test
    data = normalize(data)
    offset = int(data.shape[0] * ratio)
    print(f"train: {offset}, test {data.shape[0] -  offset}")
    return data[:offset], data[offset:]


def normalize(data):
    maximum, minimum = data.max(axis=0), data.min(axis=0)
    return (data - minimum) / (maximum - minimum)


class Regressor(fluid.dygraph.Layer):
    def __init__(self):
        super(Regressor, self).__init__()

        self.fc = Linear(input_dim=13, output_dim=1, act=None)

    def forward(self, inputs):
        return self.fc(inputs)


def run():
    with dygraph.guard():
        model = Regressor()
        model.train()  # 开启训练模式
        training_data, test_data = preprocess()
        opt = fluid.optimizer.SGD(learning_rate=0.01, parameter_list=model.parameters())

        EPOCH_NUM = 100  # 设置外层循环次数
        BATCH_SIZE = 33  # 设置batch大小

        # 定义外层循环
        for epoch_id in range(EPOCH_NUM):
            # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个batch包含10条数据
            mini_batches = [training_data[k:k + BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
            # 定义内层循环
            for iter_id, mini_batch in enumerate(mini_batches):
                x = np.array(mini_batch[:, :-1]).astype('float32')  # 获得当前批次训练数据
                y = np.array(mini_batch[:, -1:]).astype('float32')  # 获得当前批次训练标签（真实房价）
                # 将numpy数据转为飞桨动态图variable形式
                house_features = dygraph.to_variable(x)
                prices = dygraph.to_variable(y)

                # 前向计算
                predicts = model(house_features)

                # 计算损失
                loss = fluid.layers.square_error_cost(predicts, label=prices)
                avg_loss = fluid.layers.mean(loss)
                if iter_id % BATCH_SIZE == 0:
                    print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))

                # 反向传播
                avg_loss.backward()
                # 最小化loss,更新参数
                opt.minimize(avg_loss)
                # 清除梯度
                model.clear_gradients()

        fluid.save_dygraph(model.state_dict(), 'LR_model')


def eval():
    with dygraph.guard():
        model = Regressor()
        model_dict, _ = fluid.load_dygraph("LR_model")
        model.load_dict(model_dict)
        model.eval()

        training_data, test_data = preprocess()
        x = np.array(test_data[:, :-1]).astype('float32')
        y = np.array(test_data[:, -1:]).astype('float32')
        # 将numpy数据转为飞桨动态图variable形式
        house_features = dygraph.to_variable(x)
        prices = dygraph.to_variable(y)
        results = model(house_features)
        comparison = np.concatenate([results.numpy(), y], axis=1)
        loss = np.square(results.numpy() - y).mean()
        print(f"Inference result and  corresponding label: {loss}")
        print(comparison)


if __name__ == '__main__':
    # run()
    eval()