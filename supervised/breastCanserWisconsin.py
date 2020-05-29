#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangxing
# datetime:2020/5/21 4:40 下午
# software: PyCharm
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

coloumns_names = [
    'Sample code numble',
    'Clump Thickness',
    'Uniformity of Cell Size',
    'Uniform of Cell Shape',
    'Marginal Adhesion',
    'Single Epithelial Cell Size',
    'Bare Nuclei',
    'Bland Chromatin',
    'Normal Nucleoli',
    'Mitoses',
    'Class'
]

# 数据预处理
data = pd.read_csv('./data/breast-cancer-wisconsin.data', names=coloumns_names)
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')
X_train, X_test, y_train, y_test = train_test_split(data[coloumns_names[1:10]], data[coloumns_names[10]],
                                                    test_size=0.25, random_state=33)

# 数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 初始化LogisticRegression与SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier()

# 训练-逻辑回归
lr.fit(X_train, y_train)
# 预测
lr_y_predict = lr.predict(X_test)
lr_score = (lr.score(X_test, y_test))

# 随机梯度下降
sgdc.fit(X_train, y_train)
sgdc_y_predict = sgdc.predict(X_test)
sdgc_score = sgdc.score(X_test, y_test)

# 性能分析
print("Accuracy of LR Classifier:", lr.score(X_test, y_test))
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))
print("Accuracy of SGD Classifier:", sgdc.score(X_test, y_test))
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))
