#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangxing
# datetime:2020/5/23 4:24 下午
# software: PyCharm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


boston = load_boston()

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
# y_train = np.reshape(y_train, (y_train.shape[0], 1))
# y_test = np.reshape(y_test, (y_test.shape[0], 1))

ss_X = StandardScaler()
# ss_y = StandardScaler()
#
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
#
# y_train = ss_y.fit_transform(y_train)
# y_test = ss_y.transform(y_test)


# y_train = np.reshape(y_train, (y_train.shape[0], ))
# y_test = np.reshape(y_test, (y_test.shape[0], ))


lr = LinearRegression()
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

sgdr = SGDRegressor()
sgdr.fit(X_train, y_train)
sgdr_y_predict = sgdr.predict(X_test)
# print('The value of defult measurement of LinearRegression is: ', lr.score(X_test, y_test))
# print('The value of R-square of LinearRegression is: ', r2_score(y_test, lr_y_predict))
# print('The mean square error of LinearRegression is: ', mean_squared_error(y_test, lr_y_predict))
# print('The mean absoluate error of LinearRegression is: ', mean_absolute_error(y_test, lr_y_predict))


# 支持向量机（回归）
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# 线性核函数
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, y_train)
linear_svc_y_predict = linear_svr.predict(X_test)

# 多项式核函数
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train, y_train)
poly_svc_y_predict = poly_svr.predict(X_test)

# 径向基核函数
rbf_svc = SVR(kernel='rbf')
rbf_svc.fit(X_train, y_train)
rbf_svc_y_predict = rbf_svc.predict(X_test)

print('The value of defult measurement of SVC with linear/poly/rbf kernel is: {}/{}/{}: '.format(linear_svr.score(X_test, y_test),
                                                                                                 poly_svr.score(X_test, y_test),
                                                                                                 rbf_svc.score(X_test, y_test)))
print('The value of R_square of SVC with linear/poly/rbf kernel is: {}/{}/{}: '.format(r2_score(y_test, linear_svc_y_predict),
                                                                                       r2_score(y_test, poly_svc_y_predict),
                                                                                       r2_score(y_test, rbf_svc_y_predict)))
print('The mean square error of SVC with linear/poly/rbf kernel is: {}/{}/{}: '.format(mean_squared_error(y_test, linear_svc_y_predict),
                                                                                       mean_squared_error(y_test, poly_svc_y_predict),
                                                                                       mean_squared_error(y_test, rbf_svc_y_predict)))
print('The mean absolute error of SVC with linear/poly/rbf kernel is: {}/{}/{}: '.format(mean_absolute_error(y_test, linear_svc_y_predict),
                                                                                         mean_absolute_error(y_test, poly_svc_y_predict),
                                                                                         mean_absolute_error(y_test, rbf_svc_y_predict)))
