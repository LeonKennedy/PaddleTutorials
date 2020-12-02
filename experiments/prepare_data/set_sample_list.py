#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: set_sample_list.py
@time: 2020/10/19 10:30 上午
@desc:
"""
import paddle
import paddle.fluid as fluid
import paddle.dataset.mnist as mnist


def network():
    image = fluid.data(name='image', dtype='float32', shape=[None, 784])
    label = fluid.data(name='label', dtype='int64', shape=[None, 1])
    loader = fluid.io.DataLoader.from_generator(feed_list=[image, label], capacity=64)

    # Definition of models
    fc = fluid.layers.fc(image, size=10)
    xe = fluid.layers.softmax_with_cross_entropy(fc, label)
    loss = fluid.layers.reduce_mean(xe)
    return loss, loader


# Create main program and startup program for training
train_prog = fluid.Program()
train_startup = fluid.Program()

with fluid.program_guard(train_prog, train_startup):
    # Use fluid.unique_name.guard() to share parameters with test network
    with fluid.unique_name.guard():
        train_loss, train_loader = network()
        adam = fluid.optimizer.Adam(learning_rate=0.01)
        adam.minimize(train_loss)

# 创建预测的main_program和startup_program
test_prog = fluid.Program()
test_startup = fluid.Program()

# 定义预测网络
with fluid.program_guard(test_prog, test_startup):
    # Use fluid.unique_name.guard() to share parameters with train network
    with fluid.unique_name.guard():
        test_loss, test_loader = network()

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)

# 运行startup_program进行初始化
exe.run(train_startup)
exe.run(test_startup)

# Compile programs
# train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(loss_name=train_loss.name)
# test_prog = fluid.CompiledProgram(test_prog).with_data_parallel(share_vars_from=train_prog)

ITERABLE = True
# 设置DataLoader的数据源
places = fluid.cuda_places() if ITERABLE else None

train_loader.set_sample_list_generator(fluid.io.shuffle(fluid.io.batch(mnist.train(), 512), buf_size=1024),
                                       places=places)

test_loader.set_sample_list_generator(fluid.io.batch(mnist.test(), 512), places=places)


def run_iterable(program, exe, loss, data_loader):
    for data in data_loader():
        loss_value = exe.run(program=program, feed=data, fetch_list=[loss])
        print('loss is {}'.format(loss_value))


for epoch_id in range(10):
    run_iterable(train_prog, exe, train_loss, train_loader)
    run_iterable(test_prog, exe, test_loss, test_loader)
