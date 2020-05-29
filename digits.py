#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangxing
# datetime:2020/5/22 10:07 上午
# software: PyCharm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size=0.25,
                                                    random_state=33)

# 标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 初始化训练器
lsvc = LinearSVC()
# 模型训练
lsvc.fit(X_train, y_train)
# 预测
y_predict = lsvc.predict(X_test)

print("Accuracy of SVC:", lsvc.score(X_test, y_test))
print(classification_report(y_test, y_predict))