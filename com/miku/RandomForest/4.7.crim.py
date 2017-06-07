#!/usr/bin/python
# -*- encoding: utf-8

import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    pd.set_option('display.width', 200)
    np.set_printoptions(linewidth=200)
    data = pd.read_excel('..\\2.Regression\\crim.xlsx', sheetname='Sheet1', header=0)
    print 'data.head() = \n', data.head()
    columns = [c for c in data.columns]      # 列标题
    data.sort_values(by=data.columns[1], inplace=True)
    data = data.values
    x = data[:, 2:].astype(np.float)
    y = data[:, 1].astype(np.int)
    columns = columns[2:]

    model = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=10, min_samples_split=5,
                                  max_features=0.6, oob_score=True)
    model.fit(x, y)
    print 'OOB Score = ', model.oob_score_
    y_hat = model.predict(x)
    rmse = np.sqrt(np.mean((y_hat - y)**2))
    print 'RMSE = ', rmse, 'Predict Score = ', rmse / np.mean(y)
    feature_importances = np.array(zip(columns, model.feature_importances_))
    feature_importances[:, 1] = feature_importances[:, 1].astype(np.float)
    feature_importances.sort(axis=0)
    feature_importances = feature_importances[::-1]
    for fi in feature_importances:
        print fi[0], fi[1]

    plt.figure(facecolor='w')
    t = np.arange(len(y))
    plt.plot(t, y_hat, 'go', label=u'预测值')
    plt.plot(t, y, 'r-', lw=2, label=u'实际值')
    plt.grid(b=True)
    plt.legend(loc='upper left')
    plt.title(u'北京市犯罪率与特征相关性回归分析', fontsize=18)
    plt.show()
