#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: VGG.py
@time: 2020/10/12 9:42 上午
@desc:
    1. VGG-16 中的 16 表示整个网络中有 trainable 参数的层数为 16 层
    2. VGG-16 大约有 138million 个参数
    3. VGG-16 中所有卷积层 filter 宽和高都是 3，步长为 1，padding 都使用 same convolution；所有池化层的 filter 宽和高都是 2，步长都是 2。
"""
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable


# 定义vgg块，包含多层卷积和1层2x2的最大池化层
class vgg_block(fluid.dygraph.Layer):
    def __init__(self, num_convs, in_channels, out_channels):
        """
        num_convs, 卷积层的数目
        num_channels, 卷积层的输出通道数，在同一个Incepition块内，卷积层输出通道数是一样的
        """
        super(vgg_block, self).__init__()
        self.conv_list = []
        for i in range(num_convs):
            conv_layer = self.add_sublayer('conv_' + str(i), Conv2D(num_channels=in_channels,
                                                                    num_filters=out_channels, filter_size=3, padding=1,
                                                                    act='relu'))
            self.conv_list.append(conv_layer)
            in_channels = out_channels
        self.pool = Pool2D(pool_stride=2, pool_size=2, pool_type='max')

    def forward(self, x):
        for item in self.conv_list:
            x = item(x)
        return self.pool(x)


class VGG(fluid.dygraph.Layer):
    def __init__(self, conv_arch=((2, 64),
                                  (2, 128), (3, 256), (3, 512), (3, 512))):
        super(VGG, self).__init__()
        self.vgg_blocks = []
        iter_id = 0
        # 添加vgg_block
        # 这里一共5个vgg_block，每个block里面的卷积层数目和输出通道数由conv_arch指定
        in_channels = [3, 64, 128, 256, 512, 512]
        for (num_convs, num_channels) in conv_arch:
            block = self.add_sublayer('block_' + str(iter_id),
                                      vgg_block(num_convs, in_channels=in_channels[iter_id],
                                                out_channels=num_channels))
            self.vgg_blocks.append(block)
            iter_id += 1
        self.fc1 = Linear(input_dim=512 * 7 * 7, output_dim=4096,
                          act='relu')
        self.drop1_ratio = 0.5
        self.fc2 = Linear(input_dim=4096, output_dim=4096,
                          act='relu')
        self.drop2_ratio = 0.5
        self.fc3 = Linear(input_dim=4096, output_dim=1)

    def forward(self, x):
        for item in self.vgg_blocks:
            x = item(x)
        x = fluid.layers.reshape(x, [x.shape[0], -1])
        x = fluid.layers.dropout(self.fc1(x), self.drop1_ratio)
        x = fluid.layers.dropout(self.fc2(x), self.drop2_ratio)
        x = self.fc3(x)
        return x
