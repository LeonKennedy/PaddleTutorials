#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: s1.py
@time: 2020/10/19 5:50 下午
@desc:
"""
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.io import Dataset, BatchSampler, DataLoader

BATCH_NUM = 20
BATCH_SIZE = 16
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

USE_GPU = False  # whether use GPU to run model


# define a random dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


# get places
places = fluid.cuda_places() if USE_GPU else fluid.cpu_places()


# -------------------- static graph ---------------------

def simple_net(image, label):
    fluid.layers.Print(image)
    fc_tmp = fluid.layers.fc(image, size=CLASS_NUM, act='softmax')
    cross_entropy = fluid.layers.softmax_with_cross_entropy(image, label)
    loss = fluid.layers.reduce_mean(cross_entropy)
    sgd = fluid.optimizer.SGD(learning_rate=1e-3)
    sgd.minimize(loss)
    return loss


image = fluid.data(name='image', shape=[None, IMAGE_SIZE], dtype='float32')
label = fluid.data(name='label', shape=[None, 1], dtype='int64')

loss = simple_net(image, label)

exe = fluid.Executor(places[0])
exe.run(fluid.default_startup_program())

prog = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(loss_name=loss.name)

dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)

loader = DataLoader(dataset,
                    feed_list=[image, label],
                    places=places,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=True,
                    num_workers=2)

for e in range(EPOCH_NUM):
    for i, data in enumerate(loader()):
        l = exe.run(prog, feed=data, fetch_list=[loss], return_numpy=True)
        print("Epoch {} batch {}: loss = {}".format(e, i, l[0][0]))
