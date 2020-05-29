#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangxing
# datetime:2020/5/26 4:53 下午
# software: PyCharm
from sklearn.feature_extraction import DictVectorizer

measurements = [{'city': 'Dubai', 'temperature': 33.},
                {'city': 'London', 'temperature': 12.},
                {'city': 'San Fransisco', 'temperature': 18.}]

dv = DictVectorizer()
measurements_vec = dv.fit_transform(measurements)
