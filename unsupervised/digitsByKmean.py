#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangxing
# datetime:2020/5/25 5:12 下午
# software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


digits_train = pd.read_csv('/Users/zhangxing/PycharmProjects/Python_KAGGLE/data/optdigits.tra', header=None)
digits_test = pd.read_csv('/Users/zhangxing/PycharmProjects/Python_KAGGLE/data/optdigits.tes', header=None)


X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)

estimator = PCA(n_components=2)

X_pca = estimator.fit_transform(X_train)


def plot_pca_scatter():
    color = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

    for i in range(len(color)):
        px = X_pca[:, 0][y_train.values == i]
        py = X_pca[:, 1][y_train.values == i]
        plt.scatter(px, py, c=color[i], edgecolors='black')

    plt.legend(np.arange(0, 10).astype(str))
    plt.show()
