#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: create_numpy.py
@time: 2020/10/17 3:09 下午
@desc:
"""
import numpy as np

a = np.random.rand(3, 4, 5)  # uniform distribution
print(a.shape, '\n', a)

b = np.random.randn(2, 3, 5)  # standar normal
print(b.shape, '\n', b)

c = np.random.uniform(-1, 2, size=(2, 3, 5))  # uniform distributed over half-open interval [low,high)
print(c.shape, '\n', c)
