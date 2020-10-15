#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: main.py
@time: 2020/10/12 5:15 下午
@desc:
"""

# !/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: read_data.py
@time: 2020/10/12 4:58 下午
@desc:
"""
import numpy as np
import paddle.fluid as fluid

BATCH_NUM = 10
BATCH_SIZE = 16
MNIST_IMAGE_SIZE = 784
MNIST_LABLE_SIZE = 1


# 伪数据生成函数，服务于下述三种不同的生成器
def get_random_images_and_labels(image_shape, label_shape):
    image = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return image, label


# 每次生成一个Sample，使用set_sample_generator配置数据源
def sample_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM * BATCH_SIZE):
            image, label = get_random_images_and_labels([MNIST_IMAGE_SIZE], [MNIST_LABLE_SIZE])
            yield image, label

    return __reader__


# 每次生成一个Sample List，使用set_sample_list_generator配置数据源
def sample_list_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM):
            sample_list = []
            for _ in range(BATCH_SIZE):
                image, label = get_random_images_and_labels([MNIST_IMAGE_SIZE], [MNIST_LABLE_SIZE])
                sample_list.append([image, label])

            yield sample_list

    return __reader__


# 每次生成一个Batch，使用set_batch_generator配置数据源
def batch_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM):
            batch_image, batch_label = get_random_images_and_labels([BATCH_SIZE, MNIST_LABLE_SIZE],
                                                                    [BATCH_SIZE, MNIST_LABLE_SIZE])
            yield batch_image, batch_label

    return __reader__


EPOCH_NUM = 4


# 1. 构建命令式编程模式（动态图）网络
class MyLayer(fluid.dygraph.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.linear = fluid.dygraph.nn.Linear(MNIST_IMAGE_SIZE, 10)

    def forward(self, inputs, label=None):
        x = self.linear(inputs)
        if label is not None:
            loss = fluid.layers.cross_entropy(x, label)
            avg_loss = fluid.layers.mean(loss)
            return x, avg_loss
        else:
            return x


# 2. 创建网络执行对象，配置DataLoader，进行训练或预测
place = fluid.CPUPlace()  # 或者 fluid.CUDAPlace(0)
fluid.enable_imperative(place)

# 创建执行的网络对象
my_layer = MyLayer()

# 添加优化器
adam = fluid.optimizer.AdamOptimizer(
    learning_rate=0.001, parameter_list=my_layer.parameters())

# 配置DataLoader
# 使用sample数据生成器作为DataLoader的数据源
# data_loader1 = fluid.io.DataLoader.from_generator(capacity=10)
# data_loader1.set_sample_generator(sample_generator_creator(), batch_size=BATCH_SIZE, places=place)

# 使用sample list数据生成器作为DataLoader的数据源
data_loader2 = fluid.io.DataLoader.from_generator(capacity=10)
data_loader2.set_sample_list_generator(sample_list_generator_creator(), places=place)

# 使用batch数据生成器作为DataLoader的数据源
# data_loader3 = fluid.io.DataLoader.from_generator(capacity=10)
# data_loader3.set_batch_generator(batch_generator_creator(), places=place)
train_loader = fluid.io.DataLoader.from_generator(capacity=10)
train_loader.set_sample_list_generator(sample_list_generator_creator(), places=place)

# 执行训练/预测
for _ in range(EPOCH_NUM):
    for data in train_loader():
        # 拆解载入数据
        image, label = data

        # 执行前向
        x, avg_loss = my_layer(image, label)

        # 执行反向
        avg_loss.backward()

        # 梯度更新
        adam.minimize(avg_loss)
        my_layer.clear_gradients()
    print(avg_loss.numpy())