# coding:utf-8

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LinearRegression, Ridge


if __name__ == '__main__':
    data = pd.read_csv('600600_0608.txt', sep='\t\t', header=0, encoding='GBK', engine='python')
    y = data[u'成交']
    # print x

    K = 5
    weight = np.ones(K)
    weight /= weight.sum()
    print weight
    y_convolve = np.convolve(y, weight, mode='valid')
    # print y_convolve.shape
    # print y.shape

    y = y_convolve
    y = y[:-100]

    np.set_printoptions(linewidth=200)
    N = len(y)
    lag = 10
    x = np.zeros((N-lag+1, lag))
    for i in range(N-lag+1):
        x[i] = y[i:i+lag]
    y = y[lag-1:]
    # model = LinearRegression(fit_intercept=True)
    model = Ridge(alpha=1, fit_intercept=True)
    model.fit(x, y)
    y_pred = model.predict(x)

    t = np.arange(len(y))
    window = x[-1].tolist()

    m = 50
    y_next = np.zeros(m)
    for i in range(m):
        print window
        y_next[i] = model.predict(np.array(window).reshape((1, 10)))
        print i, y_next[i]
        window.pop(0)
        window.append(y_next[i])

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y, 'ro-', lw=2, ms=5, label=u'真实值')
    plt.plot(t, y_pred, 'go-', lw=2, ms=5, label=u'估计值')
    t = np.arange(len(y), len(y)+len(y_next))
    plt.plot(t, y_next, 'bo-', lw=2, ms=5, label=u'预测值')
    plt.xlabel(u'时间', fontsize=15)
    plt.ylabel(u'股票价格', fontsize=15)
    plt.title(u'600600股票价格曲线', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
