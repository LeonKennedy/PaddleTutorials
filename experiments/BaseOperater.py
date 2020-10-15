#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: BaseOperater.py
@time: 2020/10/14 11:07 下午
@desc:
"""
import numpy as np
from paddle import fluid


def j():
    batch_size = 8
    hidden_size = embedding_size = 40
    init_hidden_data = np.zeros((1, batch_size, embedding_size), dtype='float32')
    print(init_hidden_data.shape)
    with fluid.dygraph.guard():
        init_hidden = fluid.dygraph.to_variable(init_hidden_data)
        init_hidden.stop_gradient = True
        init_h = fluid.layers.reshape(init_hidden, shape=[1, -1, hidden_size])
        print(init_h.shape)


def slice():
    input = fluid.layers.data(
        name="input", shape=[3, 4, 5, 6], dtype='float32')

    # example 1:
    # attr starts is a list which doesn't contain tensor Variable.
    axes = [0, 1, 2]
    starts = [-3, 0, 2]
    ends = [3, 2, 4]
    sliced_1 = fluid.layers.slice(input, axes=axes, starts=starts, ends=ends)
    # sliced_1 is input[:, 0:3, 0:2, 2:4].

    # example 2:
    # attr starts is a list which contain tensor Variable.
    minus_3 = fluid.layers.fill_constant([1], "int32", -3)
    sliced_2 = fluid.layers.slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends)
    print(1)


def lstm():
    emb_dim = 256
    vocab_size = 10000
    data = fluid.layers.data(name='x', shape=[-1, 100, 1],
                             dtype='int64')
    emb = fluid.layers.embedding(input=data, size=[vocab_size, emb_dim], is_sparse=True)
    batch_size = 20
    max_len = 100
    dropout_prob = 0.2
    hidden_size = 150
    num_layers = 1
    init_h = fluid.layers.fill_constant([num_layers, batch_size, hidden_size], 'float32', 0.0)
    init_c = fluid.layers.fill_constant([num_layers, batch_size, hidden_size], 'float32', 0.0)

    print(emb.shape)
    rnn_out, last_h, last_c = fluid.layers.lstm(emb, init_h, init_c, max_len, hidden_size, num_layers,
                                                dropout_prob=dropout_prob)
    print(rnn_out.shape)  # (-1, 100, 150)
    print(last_h.shape)  # (1, 20, 150)
    print(last_c.shape)  # (1, 20, 150)


if __name__ == '__main__':
    # slice()
    lstm()
