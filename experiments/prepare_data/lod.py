#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: lod.py
@time: 2020/10/12 6:52 下午
@desc:
"""

import paddle.fluid as fluid
import numpy as np
import paddle.fluid.layers as layers


def sample():
    n_1 = np.array([[1], [1], [1],
                    [1], [1],
                    [1], [1], [1], [1],
                    [1],
                    [1], [1],
                    [1], [1], [1]]).astype('int64')

    print(n_1.shape)
    a = fluid.create_lod_tensor(data=n_1,
                                recursive_seq_lens=[[3, 1, 2], [3, 2, 4, 1, 2, 3]],
                                place=fluid.CPUPlace())

    # 查看lod-tensor嵌套层数
    print(a)
    print(a.recursive_sequence_lengths())
    print("The LoD of the result: {}.".format(a.lod()))

    n_2 = np.random.rand(15).astype('float32')
    a = fluid.create_lod_tensor(data=n_2,
                                recursive_seq_lens=[[3, 1, 2], [3, 2, 4, 1, 2, 3]],
                                place=fluid.CPUPlace())

    # 查看lod-tensor嵌套层数
    print(a)
    print(a.recursive_sequence_lengths())
    print("The LoD of the result: {}.".format(a.lod()))

    n_2 = np.random.rand(15, 3).astype('float32')
    a = fluid.create_lod_tensor(data=n_2,
                                recursive_seq_lens=[[3, 1, 2], [3, 2, 4, 1, 2, 3]],
                                place=fluid.CPUPlace())

    # 查看lod-tensor嵌套层数
    print(a)
    print(a.recursive_sequence_lengths())
    print("The LoD of the result: {}.".format(a.lod()))


def LodTensor_to_Tensor(lod_tensor):
    # 获取 LoD-Tensor 的 lod 信息
    lod = lod_tensor.lod()
    # 转换成 array
    array = np.array(lod_tensor)
    new_array = []
    # 依照原LoD-Tensor的层级信息，转换成Tensor
    for i in range(len(lod[0]) - 1):
        new_array.append(array[lod[0][i]:lod[0][i + 1]])
    return new_array


def to_lodtensor(data, place):
    # 存储Tensor的长度作为LoD信息
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    # 对待转换的 Tensor 降维
    flattened_data = np.concatenate(data, axis=0).astype("float32")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    # 为 Tensor 数据添加lod信息
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def create():
    # 创建一个 LoD-Tensor
    a = fluid.create_lod_tensor(np.array([[1.1], [2.2], [3.3], [4.4]]).astype('float32'), [[1, 3]], fluid.CPUPlace())
    new_array = LodTensor_to_Tensor(a)
    # 输出结果
    print(new_array)
    # new_array 为上段代码中转换的Tensor
    lod_tensor = to_lodtensor(new_array, fluid.CPUPlace())

    # 输出 LoD 信息
    print("The LoD of the result: {}.".format(lod_tensor.lod()))

    # 检验与原Tensor数据是否一致
    print("The array : {}.".format(np.array(lod_tensor)))


def exp_sequence_expand():
    # x = fluid.data(name='x', shape=[1], dtype='float32')
    # y = fluid.data(name='y', shape=[1], dtype='float32', lod_level=1)
    # out = layers.sequence_expand(x=x, y=y, ref_level=0)

    x = fluid.data(name='x', shape=[4, 1], dtype='float32')
    y = fluid.data(name='y', shape=[8, 1], dtype='float32', lod_level=1)
    out = layers.sequence_expand(x=x, y=y, ref_level=0)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    np_data = np.array([[1], [2], [3], [4]]).astype('float32')
    x_lod_tensor = fluid.create_lod_tensor(np_data, [[2, 2]], place)
    print(x_lod_tensor)

    y_lod_tensor = fluid.create_random_int_lodtensor([[2, 2], [3, 3, 1, 1]], [1],
                                                     place, low=0, high=1)
    y_lod_tensor2 = fluid.create_random_int_lodtensor([[2, 2], [3, 3, 1, 1]], [9, 16], place, low=0, high=1)
    print(y_lod_tensor)
    print(y_lod_tensor2)
    # lod: [[0, 2, 4][0, 3, 6, 7, 8]]
    #    dim: 8, 1
    #    layout: NCHW
    #    dtype: int64_t
    #    data: [0 0 1 1 1 1 1 0]

    out_main = exe.run(fluid.default_main_program(),
                       feed={'x': x_lod_tensor, 'y': y_lod_tensor},
                       fetch_list=[out], return_numpy=False)
    print(out_main[0])


if __name__ == '__main__':
    # sample()
    create()
    # exp_sequence_expand()
