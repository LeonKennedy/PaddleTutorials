#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: imperative.py
@time: 2020/10/20 2:28 下午
@desc:
"""

from experiments.fit_house_price.main import preprocess
from paddle import fluid


def create_reader():
    def reader():
        training_data, test_data = preprocess()
        for i in training_data:
            yield i[:-1], i[-1]
    return reader


def create_batch_reader(batch_size=12):
    def reader():
        training_data, test_data = preprocess()
        mini_batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]
        # 定义内层循环
        for iter_id, mini_batch in enumerate(mini_batches):
            x = mini_batch[:, :-1].astype('float32')  # 获得当前批次训练数据
            y = mini_batch[:, -1:].astype('int64')  # 获得当前批次训练标签（真实房价）
            yield x, y

    return reader


def network():
    x = fluid.data(name='x', shape=[None, 13])
    y = fluid.data(name='y', shape=[None, 1], dtype='int64')
    loader = fluid.io.DataLoader.from_generator(feed_list=[x, y], capacity=10, iterable=False)

    fluid.layers.Print(x)
    fc = fluid.layers.fc(x, size=10)
    xe = fluid.layers.softmax_with_cross_entropy(fc, y)
    loss = fluid.layers.reduce_mean(xe)
    return loss, loader


train_prog = fluid.Program()
train_startup = fluid.Program()

with fluid.program_guard(train_prog, train_startup):
    # Use fluid.unique_name.guard() to share parameters with test network
    with fluid.unique_name.guard():
        train_loss, train_loader = network()
        adam = fluid.optimizer.Adam(learning_rate=0.01)
        adam.minimize(train_loss)

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)

exe.run(train_startup)
ITERABLE = True
places = fluid.cuda_places() if ITERABLE else None

BATCH_SIZE = 12
train_loader.set_batch_generator(create_batch_reader(BATCH_SIZE), places=places)
train_loader.start()
total = 0
iter = 0
while 1:
    total += BATCH_SIZE
    iter += 1
    try:
        loss_value = exe.run(program=train_prog, fetch_list=[train_loss])
        print(f"iter({iter}) total({total}) loss is {loss_value}")
    except fluid.core.EOFException:
        train_loader.reset()
        break

