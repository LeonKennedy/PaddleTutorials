#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: s1.py
@time: 2020/10/17 3:03 下午
@desc:
"""

import numpy as np

a = np.array([1, 2, 3]).astype('int32')
print(a.shape)
print(a)

b = np.expand_dims(a, axis=-1)
print(b.shape, b)
c = np.expand_dims(b, axis=-1)
print(c.shape, c)
d = c.squeeze(axis=-1)
print(d.shape, d)

