#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangxing
# datetime:2020/5/26 6:58 下午
# software: PyCharm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

liner = LinearRegression()
liner.fit(X_train, y_train)
print(liner.score(X_train, y_train))

xx = np.linspace(0, 26, 100)
xx = xx.reshape((xx.shape[0], 1))
yy = liner.predict(xx)

# plt.scatter(X_train, y_train)
# plt.plot(xx, yy)
# plt.show()

poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)

# regressor_poly2 = LinearRegression()
# regressor_poly2.fit(X_train_poly2, y_train)

