#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: main.py
@time: 2020/10/14 4:56 下午
@desc:
"""
import paddle.fluid as fluid

# 开始训练
from data import get_data, build_batch
from net import SentimentClassifier

batch_size = 50
epoch_num = 5
embedding_size = 256
max_seq_len = 128

place = fluid.CUDAPlace(0)
train_corpus, test_corpus, word2id_dict = get_data()
vocab_size = len(word2id_dict)


def train(learning_rate=0.01):
    step = 0
    with fluid.dygraph.guard(place):
        # 创建一个用于情感分类的网络实例，sentiment_classifier
        sentiment_classifier = SentimentClassifier(
            embedding_size, vocab_size, num_steps=max_seq_len)
        # 创建优化器AdamOptimizer，用于更新这个网络的参数
        adam = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate,
                                             parameter_list=sentiment_classifier.parameters())

        for sentences, labels in build_batch(
                word2id_dict, train_corpus, batch_size, epoch_num, max_seq_len):

            sentences_var = fluid.dygraph.to_variable(sentences)
            labels_var = fluid.dygraph.to_variable(labels)
            pred, loss = sentiment_classifier(sentences_var, labels_var)

            loss.backward()
            adam.minimize(loss)
            sentiment_classifier.clear_gradients()

            step += 1
            if step % 10 == 0:
                print("step %d, loss %.3f" % (step, loss.numpy()[0]))

        fluid.save_dygraph(sentiment_classifier.state_dict(), 'sentiment_classifier')
        # 我们希望在网络训练结束以后评估一下训练好的网络的效果
        # 通过eval()函数，将网络设置为eval模式，在eval模式中，网络不会进行梯度更新
        sentiment_classifier.eval()
        # 这里我们需要记录模型预测结果的准确率
        # 对于二分类任务来说，准确率的计算公式为：
        # (true_positive + true_negative) /
        # (true_positive + true_negative + false_positive + false_negative)
        tp = 0.
        tn = 0.
        fp = 0.
        fn = 0.
        for sentences, labels in build_batch(
                word2id_dict, test_corpus, batch_size, 1, max_seq_len):

            sentences_var = fluid.dygraph.to_variable(sentences)
            labels_var = fluid.dygraph.to_variable(labels)

            # 获取模型对当前batch的输出结果
            pred, loss = sentiment_classifier(sentences_var, labels_var)

            # 把输出结果转换为numpy array的数据结构
            # 遍历这个数据结构，比较预测结果和对应label之间的关系，并更新tp，tn，fp和fn
            pred = pred.numpy()
            for i in range(len(pred)):
                if labels[i][0] == 1:
                    if pred[i][1] > pred[i][0]:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if pred[i][1] > pred[i][0]:
                        fp += 1
                    else:
                        tn += 1

        # 输出最终评估的模型效果
        print("the acc in the test set is %.3f" % ((tp + tn) / (tp + tn + fp + fn)))


def eval():
    with fluid.dygraph.guard(place):
        model = SentimentClassifier(embedding_size, vocab_size, num_steps=max_seq_len)
        model_state_dict, _ = fluid.load_dygraph('sentiment_classifier')
        model.load_dict(model_state_dict)
        model.eval()
        # 这里我们需要记录模型预测结果的准确率
        # 对于二分类任务来说，准确率的计算公式为：
        # (true_positive + true_negative) /
        # (true_positive + true_negative + false_positive + false_negative)
        tp = 0.
        tn = 0.
        fp = 0.
        fn = 0.
        for sentences, labels in build_batch(word2id_dict, test_corpus, batch_size, 1, max_seq_len):

            sentences_var = fluid.dygraph.to_variable(sentences)
            labels_var = fluid.dygraph.to_variable(labels)

            # 获取模型对当前batch的输出结果
            pred, loss = model(sentences_var, labels_var)

            # 把输出结果转换为numpy array的数据结构
            # 遍历这个数据结构，比较预测结果和对应label之间的关系，并更新tp，tn，fp和fn
            pred = pred.numpy()
            for i in range(len(pred)):
                if  [0] == 1:
                    if pred[i][1] > pred[i][0]:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if pred[i][1] > pred[i][0]:
                        fp += 1
                    else:
                        tn += 1

        # 输出最终评估的模型效果
        print("the acc in the test set is %.3f" % ((tp + tn) / (tp + tn + fp + fn)))


if __name__ == '__main__':
    # train()
    eval()
