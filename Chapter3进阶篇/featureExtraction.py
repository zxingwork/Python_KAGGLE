#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangxing
# datetime:2020/5/29 2:39 下午
# software: PyCharm
from sklearn.feature_extraction import DictVectorizer

measurements = [
    {
        'city': 'Dubai',
        'temperature': 33.
    },
    {
        'city': 'London',
        'temperature': 12.
    },
    {
        'city': 'San Fransisco',
        'temperature': 18.
    }
]
vec = DictVectorizer()
data = vec.fit_transform(measurements)
data2 = vec.fit_transform(measurements).toarray()
print(data)
print(data2)
print(vec.get_feature_names())
