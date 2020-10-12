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
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm, Dropout
from data import IMG_ROWS, IMG_COLS
import numpy as np
from visualdl import LogWriter

log_writer = LogWriter("./log")


class ConvolutionNatualNetwork(fluid.dygraph.Layer):
    def __init__(self):
        super(ConvolutionNatualNetwork, self).__init__()
        self.conv1 = Conv2D(num_channels=1, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
        # 定义池化层，池化核pool_size=2，池化步长为2，选择最大池化方式
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # self.drop1 = Dropout(0.5, dropout_implementation='')
        # 定义卷积层，输出特征通道num_filters设置为20，卷积核的大小filter_size为5，卷积步长stride=1，padding=2
        self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
        # 定义池化层，池化核pool_size=2，池化步长为2，选择最大池化方式
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.bn1 = BatchNorm(20)
        # 定义一层全连接层，输出维度是1，不使用激活函数
        self.fc = Linear(input_dim=980, output_dim=10, act="softmax")

    def forward(self, inputs, label=None, check_shape=False, check_content=False):
        inputs = fluid.layers.reshape(inputs, [inputs.shape[0], 1, IMG_ROWS, IMG_COLS])
        c1 = self.conv1(inputs)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        b1 = self.bn1(p2)
        _p2 = fluid.layers.reshape(b1, [b1.shape[0], -1])
        fc1 = self.fc(_p2)

        if check_shape:
            # 打印每层网络设置的超参数-卷积核尺寸，卷积步长，卷积padding，池化核尺寸
            print("\n########## print network layer's superparams ##############")
            print("conv1-- kernel_size:{}, padding:{}, stride:{}".format(self.conv1.weight.shape, self.conv1._padding,
                                                                         self.conv1._stride))
            print("conv2-- kernel_size:{}, padding:{}, stride:{}".format(self.conv2.weight.shape, self.conv2._padding,
                                                                         self.conv2._stride))
            print("pool1-- pool_type:{}, pool_size:{}, pool_stride:{}".format(self.pool1._pool_type,
                                                                              self.pool1._pool_size,
                                                                              self.pool1._pool_stride))
            print("pool2-- pool_type:{}, poo2_size:{}, pool_stride:{}".format(self.pool2._pool_type,
                                                                              self.pool2._pool_size,
                                                                              self.pool2._pool_stride))
            print("fc-- weight_size:{}, bias_size_{}, activation:{}".format(self.fc.weight.shape, self.fc.bias.shape,
                                                                            self.fc._act))

            # 打印每层的输出尺寸
            print("\n########## print shape of features of every layer ###############")
            print("inputs_shape: {}".format(inputs.shape))
            print("conv1_outputs_shape: {}".format(c1.shape))
            print("pool1_output_shape: {}".format(p1.shape))
            print("conv2_outputs_shape: {}".format(c2.shape))
            print("pool2_outputs_shape: {}".format(p2.shape))
            print("fc_outputs_shape: {}".format(fc1.shape))

            # 选择是否打印训练过程中的参数和输出内容，可用于训练过程中的调试
        if check_content:
            # 打印卷积层的参数-卷积核权重，权重参数较多，此处只打印部分参数
            print("\n########## print convolution layer's kernel ###############")
            print("conv1 params -- kernel weights:", self.conv1.weight[0][0])
            print("conv2 params -- kernel weights:", self.conv2.weight[0][0])

            # 创建随机数，随机打印某一个通道的输出值
            idx1 = np.random.randint(0, c1.shape[1])
            idx2 = np.random.randint(0, c2.shape[1])
            # 打印卷积-池化后的结果，仅打印batch中第一个图像对应的特征
            print("\nThe {}th channel of conv1 layer: ".format(idx1), c1[0][idx1])
            print("The {}th channel of conv2 layer: ".format(idx2), c2[0][idx2])
            print("The output of last layer:", fc1[0], '\n')
        if label is not None:
            acc = fluid.layers.accuracy(input=fc1, label=label)
            return fc1, acc
        else:
            return fc1


def run():
    with fluid.dygraph.guard():
        model = ConvolutionNatualNetwork()
        model.train()
        # 使用SGD优化器，learning_rate设置为0.01
        # optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=0.01, momentum=0.9, parameter_list=model.parameters())
        # optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.01, parameter_list=model.parameters())
        batch_size = 100
        EPOCH_NUM = 5
        total_step = (int(60000/batch_size) + 1) * EPOCH_NUM
        lr = fluid.dygraph.PolynomialDecay(0.01, total_step, 0.0001)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr,
                                                  regularization=fluid.regularizer.L2Decay(regularization_coeff=0.1),
                                                  parameter_list=model.parameters())
        # 训练5轮
        train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=batch_size)

        iter_total = 0
        for epoch_id in range(EPOCH_NUM):
            for batch_id, data in enumerate(train_loader()):
                # 准备数据
                image_data = np.array([x[0] for x in data]).astype('float32')
                label_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)
                image = fluid.dygraph.to_variable(image_data)
                label = fluid.dygraph.to_variable(label_data)

                # 前向计算的过程
                if batch_id == 0 and epoch_id == 0:
                    # 打印模型参数和每层输出的尺寸
                    predict, acc = model(image, label, check_shape=True, check_content=False)
                elif batch_id == 401:
                    # 打印模型参数和每层输出的值
                    predict, acc = model(image, label, check_shape=False, check_content=True)
                else:
                    predict, acc = model(image, label)
                # 计算损失，取一个批次样本损失的平均值
                loss = fluid.layers.cross_entropy(predict, label)
                avg_loss = fluid.layers.mean(loss)

                # 每训练了200批次的数据，打印下当前Loss的情况
                if batch_id % batch_size == 0:
                    print("epoch: {}, batch: {}, loss is: {} acc: {} lr: {}".format(epoch_id, batch_id, avg_loss.numpy(),
                                                                             acc.numpy(), optimizer.current_step_lr()))
                    log_writer.add_scalar(tag='acc', step=iter_total, value=acc.numpy())
                    log_writer.add_scalar(tag='loss', step=iter_total, value=avg_loss.numpy())
                    iter_total += batch_size

                # 后向传播，更新参数的过程
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                model.clear_gradients()

        # 保存模型参数
        fluid.save_dygraph(model.state_dict(), 'cnn')


def eval():
    with fluid.dygraph.guard():
        print('start evaluation .......')
        # 加载模型参数
        model = ConvolutionNatualNetwork()
        model_state_dict, _ = fluid.load_dygraph('cnn')
        model.load_dict(model_state_dict)

        model.eval()
        train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=50)
        acc_set = []
        avg_loss_set = []
        for batch_id, data in enumerate(train_loader()):
            image_data = np.array([x[0] for x in data]).astype('float32')
            label_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)
            img = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
            prediction, acc = model(img, label)
            loss = fluid.layers.cross_entropy(input=prediction, label=label)
            avg_loss = fluid.layers.mean(loss)
            acc_set.append(float(acc.numpy()))
            avg_loss_set.append(float(avg_loss.numpy()))

        # 计算多个batch的平均损失和准确率
        acc_val_mean = np.array(acc_set).mean()
        avg_loss_val_mean = np.array(avg_loss_set).mean()

        print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))


if __name__ == '__main__':
    run()
    # eval()
